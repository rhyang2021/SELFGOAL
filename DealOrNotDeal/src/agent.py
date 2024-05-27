import openai
import ai21
import re
import cohere
import tiktoken
import dill
import ujson as json
from openai import OpenAI
from copy import deepcopy
from pprint import pprint
from tenacity import retry, stop_after_attempt, wait_chain, wait_fixed
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.chat_models import ChatOpenAI

from lib_api import *

# from local.azure import azure_completion_with_backoff
from src.prompt_base import (INSTRUCT_DECISION_SCENE_TEMPLATE,
                             FEEDBACK_TEMPLATE,
                             REACT_TEMPLATE,
                             PARSE_RESPONSE_SYSTEM_MESSAGE, INSTRUCT_REFLEXION_TEMPLATE, INSTRUCT_LEARNING_TEMPLATE,
                             INSTRUCT_CLIN_TEMPLATE, INSTRUCT_REFINE_TEMPLATE, ADAPT_TEMPLATE)
from src.retrieve_base import NodeRetrieve
from utils import extract_jsons_from_text, wprint, assign_labels_to_last_layer, extract_numbered_list

DESIRE_DESC = {
    'selfish': "Your primary objective is to maximize your own profit in this negotiation, regardless of your partner's outcome.",
    'fair': "Your primary objective is to minimize the profit gap between yourself and your partner in this negotiation, regardless of your own profit."
}


def load_initial_instructions(path_to_instructions):
    """Load initial instructions from textual format to a python dict"""
    pattern = r"==== (SYSTEM|USER|ASSISTANT) ===="

    # Use re.split to split the string by the pattern
    with open(path_to_instructions) as f:
        content = f.read()
        content = re.split(pattern, content)
        content_ = []
        for c in content:
            if (c != ""): content_.append(c)
        content = content_
        l = len(content)
        assert (l % 2 == 0)
        initial_instruction = []
        for i in range(0, l, 2):
            instruction = {"role": content[i].strip().lower().replace("====", "").replace(" ", "").strip(),
                           "content": content[i + 1].strip()
                           }
            initial_instruction.append(instruction)
        initial_instruction[0]['role'] = 'user'
        init = [initial_instruction[0],
                {"role": 'assistant',
                 "content": 'Got it!'}
                ]
        initial_instruction = init + initial_instruction[1:]
    return initial_instruction


def involve_moderator(player_1_run, player_2_run, n=0):
    """If at least one player's response does not contain a number, involve a moderator
    The moderator determines if they players have reached an agreement, or break the 
    negotiation, or is still in negotiation.
    """
    number_pattern = r"[-+]?\d*\.\d+|\d+|one|two|three|four|five|\?"

    # Use re.search to find if the string contains a match to the pattern
    match_1 = re.search(number_pattern, player_1_run)
    # print(match_1)
    match_2 = re.search(number_pattern, player_2_run)
    # print(match_2)

    if ((match_1 is not None and match_2 is None) or
            (match_1 is None and match_2 is not None)
            or (match_1 is None and match_2 is None)
            or (n >= 6)
    ):
        return True
    else:
        return False


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-3.5-turbo-1106",
        "gpt-4-1106-preview",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
    }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    openai_cost = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>

    if model in {"gpt-3.5-turbo-0613", "gpt-3.5-turbo-1106"}:
        openai_cost = num_tokens / 1000 * 0.002
    elif model in {"gpt-4-1106-preview"}:
        openai_cost = num_tokens / 1000 * 0.03

    return num_tokens, openai_cost


class DialogAgent(object):
    """GPT Agent base class, later derived to be a seller, buyer, critic, or moderator

    TODO: add code to detect price inconsistency to seller and buyer
    TODO: release the restriction of the fixed initial price 
    """

    def __init__(self,
                 initial_dialog_history=None,
                 agent_type="",  # "seller", "buyer", "critic", "moderator"
                 system_instruction="You are a helpful AI assistant",
                 engine="gpt-3.5-turbo-0613",
                 model_path="",
                 base_url="",
                 api_key="",
                 ):
        """Initialize the agent"""
        super().__init__()

        self.agent_type = agent_type
        self.engine = engine
        self.model_path = model_path
        self.base_url = base_url
        self.api_key = api_key
        self.llm_token_count = 0
        self.openai_cost = 0

        if ("claude" in self.engine):
            self.claude = anthropic.Client(self.api_key)
        if ("cohere" in self.engine):
            assert self.engine in ["cohere-command-nightly",
                                   "cohere-command",
                                   "cohere-command-light",
                                   "cohere-command-light-nightly"
                                   ]
            self.cohere_model = self.engine[7:]
            self.co = cohere.Client(api_key)

        if (initial_dialog_history is None):
            self.dialog_history = [{"role": "system", "content": system_instruction}]
        else:
            self.initial_dialog_history = deepcopy(initial_dialog_history)
            self.dialog_history = deepcopy(initial_dialog_history)

        self.last_prompt = ""
        return

    def reset(self):
        """Reset dialog history"""
        self.dialog_history = deepcopy(self.initial_dialog_history)
        self.openai_cost = 0
        self.llm_token_count = 0
        return

    def call_engine(self, messages):
        if ("gpt" in self.engine):
            # import ipdb; ipdb.set_trace()
            response = completion_with_backoff(
                model=self.engine,
                messages=messages,
                temperature=0,
            )

            llm_token_count, openai_cost = num_tokens_from_messages(messages, self.engine)
            self.llm_token_count += llm_token_count
            self.openai_cost += openai_cost
            message = {"role": "assistant", 'content': response.choices[0].message.content}
            assert (message["role"] == 'assistant')

        elif ("claude" in self.engine):
            prompt_claude = convert_openai_to_anthropic_prompt(messages)
            response = claude_completion_with_backoff(self.claude,
                                                      prompt=prompt_claude,
                                                      stop_sequences=[anthropic.HUMAN_PROMPT],
                                                      model=self.engine,
                                                      max_tokens_to_sample=512,
                                                      )
            message = {"role": "assistant", "content": response["completion"].strip()}
        elif ("j2" in self.engine):
            prompt_ai21 = convert_openai_to_ai21_prompt_format_1(messages, self.agent_type)
            response = ai21_completion_with_backoff(model=self.engine,
                                                    prompt=prompt_ai21,
                                                    numResults=1,
                                                    maxTokens=512,
                                                    temperature=0.7,
                                                    topKReturn=0,
                                                    topP=1,
                                                    stopSequences=["##"]
                                                    )
            content = response["completions"][0]["data"]["text"]
            if (self.agent_type in ["seller", "buyer"]):
                content = content.split('\n')[0]
            message = {"role": "assistant",
                       "content": content
                       }
        elif ("cohere" in self.engine):
            prompt_cohere = convert_openai_to_cohere_prompt(messages)
            # import ipdb; ipdb.set_trace()
            response = cohere_completion_with_backoff(self.co,
                                                      prompt=prompt_cohere,
                                                      model=self.cohere_model,
                                                      max_tokens=512,
                                                      )

            # import ipdb; ipdb.set_trace()
            message = {"role": "assistant",
                       "content": response[0].text
                       }

        elif "mistral" in self.engine:
            client = OpenAI(
                # defaults to os.environ.get("OPENAI_API_KEY")
                api_key="EMPTY",
                base_url=self.base_url,
            )
            models = client.models.list()
            model = models.data[0].id
            response = client.chat.completions.create(
                temperature=0,
                model=model,
                messages=messages,
                top_logprobs=1,
                logprobs=True,
            )
            message = {"role": "assistant", 'content': response.choices[0].message.content}
            assert (message['role'] == 'assistant')


        elif 'qwen' in self.engine:
            client = OpenAI(
                api_key="EMPTY",
                base_url=self.base_url,
            )
            models = client.models.list()
            model = models.data[0].id
            response = client.chat.completions.create(
                temperature=0,
                model=model,
                messages=messages,
                top_logprobs=1,
                logprobs=True,
            )
            message = {"role": "assistant", 'content': response.choices[0].message.content}
            assert (message['role'] == 'assistant')

        else:
            raise ValueError("Unknown engine %s" % self.engine)
        return message

    def call(self, prompt):

        """Call the agent with a prompt. Handle different backend engines in this function
        """
        # TODO: refactor the code, add `remember_history` flag
        #       if yes, then add the prompt to the dialog history, else not
        prompt = {"role": "user", "content": prompt}
        self.dialog_history.append(prompt)
        self.last_prompt = prompt['content']

        messages = list(self.dialog_history)
        # messages.append(prompt)
        message = self.call_engine(messages)

        self.dialog_history.append(dict(message))
        return message["content"]

    @property
    def last_response(self):
        return self.dialog_history[-1]['content']

    @property
    def history(self):
        for h in self.dialog_history:
            print('%s:  %s' % (h["role"], h["content"]))
        return


class BuyerAgent(DialogAgent):

    def __init__(self,
                 initial_dialog_history=None,
                 agent_type="buyer",
                 engine="gpt-3.5-turbo",
                 api_key="",
                 desire='selfish',
                 bandit_tool='',
                 book_value=5,
                 hat_value=0,
                 ball_value=5,
                 book_cnt=1,
                 hat_cnt=3,
                 ball_cnt=1,
                 ):
        """Initialize the buyer agent"""
        super().__init__(initial_dialog_history=initial_dialog_history,
                         agent_type=agent_type,
                         engine=engine,
                         api_key=api_key,
                         )
        self.desire = desire
        self.bandit_tool = bandit_tool
        self.book_value = book_value
        self.hat_value = hat_value
        self.ball_value = ball_value
        self.book_cnt = book_cnt
        self.hat_cnt = hat_cnt
        self.ball_cnt = ball_cnt

        print("Initializing buyer with engine %s" % self.engine)

        for i, d in enumerate(self.initial_dialog_history):
            self.initial_dialog_history[i]["content"] = d["content"].replace("BOOK_VALUE", str(book_value))
            self.initial_dialog_history[i]["content"] = d["content"].replace("HAT_VALUE", str(hat_value))
            self.initial_dialog_history[i]["content"] = d["content"].replace("BALL_VALUE", str(ball_value))
            self.initial_dialog_history[i]["content"] = d["content"].replace("BOOK_COUNT", str(book_cnt))
            self.initial_dialog_history[i]["content"] = d["content"].replace("HAT_COUNT", str(hat_cnt))
            self.initial_dialog_history[i]["content"] = d["content"].replace("BALL_COUNT", str(ball_cnt))
            self.initial_dialog_history[i]["content"] = d["content"].replace("DESIRE_DESC", DESIRE_DESC[self.desire])
        self.dialog_history = deepcopy(self.initial_dialog_history)
        return

    def feedback(self, result):
        """Reset dialog history"""
        if result["final_proposal"]:
            missing_keys = [key for key in ["alice_items", "bob_items"] if key not in result["final_proposal"]]
            if missing_keys:
                text = ", ".join(f"Missing {key} items" for key in missing_keys)
            else:
                alice_items = " ".join(f"{cnt} {item}" for item, cnt in result["final_proposal"]["alice_items"].items())
                bob_items = " ".join(f"{cnt} {item}" for item, cnt in result["final_proposal"]["bob_items"].items())
                text = f"Alice receives: {alice_items}. Bob receives: {bob_items}."
        else:
            text = 'No parsable proposal.'

        msgs = FEEDBACK_TEMPLATE.format(
            text=text,
            errors=result["profits"],
        )
        self.initial_dialog_history[1]['content'] += '\n\n' + msgs
        self.dialog_history = deepcopy(self.initial_dialog_history)
        self.openai_cost = 0
        self.llm_token_count = 0
        self.guidance = ""
        self.guidance_history = []
        return


class SellerAgent(DialogAgent):

    def __init__(self,
                 initial_dialog_history=None,
                 agent_type="seller",
                 engine="gpt-3.5-turbo-1106",
                 model_path="",
                 base_url="",
                 api_key="",
                 desire='selfish',
                 name='',
                 learning_strategy="none",
                 retrieve_strategy="none",
                 max_depth=0.8,
                 enable_prune=False,
                 enable_react=False,
                 enable_adapt=False,
                 enable_tool=False,
                 book_value=5,
                 hat_value=0,
                 ball_value=5,
                 book_cnt=1,
                 hat_cnt=3,
                 ball_cnt=1,
                 ):
        """Initialize the seller agent"""
        super().__init__(initial_dialog_history=initial_dialog_history,
                         agent_type=agent_type,
                         engine=engine,
                         model_path=model_path,
                         base_url=base_url,
                         api_key=api_key,
                         )

        self.desire = desire
        self.name = name
        self.enable_prune = enable_prune
        self.enable_react = enable_react
        self.enable_adapt = enable_adapt
        self.enable_tool = enable_tool
        self.learning_strategy = learning_strategy
        self.retrieve_strategy = retrieve_strategy
        self.max_depth = max_depth
        self.book_value = book_value
        self.hat_value = hat_value
        self.ball_value = ball_value
        self.book_cnt = book_cnt
        self.hat_cnt = hat_cnt
        self.ball_cnt = ball_cnt

        self.learnings = ''
        self.guidance = ''
        self.guidance_history = []
        self.prune_list = []

        print("Initializing seller with engine %s" % self.engine)

        if self.retrieve_strategy != 'none':
            self.retrieve_tool = NodeRetrieve(search_model=self.retrieve_strategy,
                                              model_path=self.model_path,
                                              base_url=self.base_url,
                                              max_depth=self.max_depth,
                                              enable_prune=enable_prune)
            self.enable_tool = True

        for i, d in enumerate(self.initial_dialog_history):
            self.initial_dialog_history[i]["content"] = d["content"].replace("BOOK_VALUE", str(book_value))
            self.initial_dialog_history[i]["content"] = d["content"].replace("HAT_VALUE", str(hat_value))
            self.initial_dialog_history[i]["content"] = d["content"].replace("BALL_VALUE", str(ball_value))
            self.initial_dialog_history[i]["content"] = d["content"].replace("BOOK_COUNT", str(book_cnt))
            self.initial_dialog_history[i]["content"] = d["content"].replace("HAT_COUNT", str(hat_cnt))
            self.initial_dialog_history[i]["content"] = d["content"].replace("BALL_COUNT", str(ball_cnt))
            self.initial_dialog_history[i]["content"] = d["content"].replace("DESIRE_DESC", DESIRE_DESC[self.desire])

        self.dialog_history = deepcopy(self.initial_dialog_history)
        return

    def learning(self, past_auction_log):

        if self.learning_strategy == 'reflexion':
            instruct_learn = INSTRUCT_REFLEXION_TEMPLATE.format(
                past_auction_log=past_auction_log)
            prompt = {"role": "user", "content": instruct_learn}
            result = self.call_engine([prompt])
            self.learnings += 'Reflections:\n-' + result["content"] + '\n'

        elif self.learning_strategy == 'clin':
            instruct_learn = INSTRUCT_LEARNING_TEMPLATE.format(
                past_auction_log=past_auction_log,
                past_learnings=self.learnings)
            prompt = {"role": "user", "content": instruct_learn}
            result = self.call_engine([prompt])
            self.learnings += 'Learnings:\n-' + result["content"] + '\n'

        if self.learnings != '':
            self.dialog_history[1]['content'] = self.initial_dialog_history[1]['content'] + \
                                                f"\n\nHere are your key learning points and practical tips from a " \
                                                f"previous negotiation. You can use them to guide this negotiation:\n```\n" \
                                                f"{self.learnings}\n```"
        return self.learnings

    def parse_thinking(self, result):
        llm = ChatOpenAI(model='gpt-3.5-turbo-0613', temperature=0.)
        msgs = llm.invoke([SystemMessage(content=PARSE_RESPONSE_SYSTEM_MESSAGE),
                           HumanMessage(content=f"{result.content}\nDon't output anything else other than the JSON "
                                                f"object.")])
        response = json.loads(msgs.content)
        return response

    def adapt(self, prompt):

        prompt = {"role": "user", "content": prompt}
        self.dialog_history.append(prompt)
        dialog_history = ""
        dialog_history += f"{self.initial_dialog_history[0]['content']}\n\n{self.initial_dialog_history[1]['content']}\n\n"

        for h in self.dialog_history[2:]:
            dialog_history += '%s:  %s \n' % (
                h["role"].replace('user', 'Bob').replace('assistant', 'Alice'), h["content"])

        instruct = ADAPT_TEMPLATE.format(dialog_history=dialog_history,
                                         goal=DESIRE_DESC[self.desire]
                                         )
        if 'mistral' in self.engine:
            llm = ChatOpenAI(model=self.model_path,
                             temperature=0,
                             api_key="EMPTY",
                             base_url=self.base_url)
        elif 'qwen' in self.engine:
            llm = ChatOpenAI(model=self.model_path,
                             temperature=0,
                             api_key="EMPTY",
                             base_url=self.base_url
                             )

        else:
            llm = ChatOpenAI(model=self.engine, temperature=0.)

        result = llm.invoke([HumanMessage(content=instruct)])
        print(self.enable_adapt, result)
        response = self.parse_thinking(result)

        message = {"role": "assistant",
                   "content": response['response']
                   }
        self.dialog_history.append(dict(message))

        return message["content"]

    def react(self, prompt):

        prompt = {"role": "user", "content": prompt}
        self.dialog_history.append(prompt)
        dialog_history = ""
        dialog_history += f"{self.initial_dialog_history[0]['content']}\n\n{self.initial_dialog_history[1]['content']}\n\n"

        for h in self.dialog_history[2:]:
            dialog_history += '%s:  %s \n' % (
                h["role"].replace('user', 'Bob').replace('assistant', 'Alice'), h["content"])

        instruct = REACT_TEMPLATE.format(dialog_history=dialog_history)

        if 'mistral' in self.engine:
            llm = ChatOpenAI(model=self.model_path,
                             temperature=0,
                             api_key="EMPTY",
                             base_url=self.base_url)
        elif 'qwen' in self.engine:
            llm = ChatOpenAI(model=self.model_path,
                             temperature=0,
                             api_key="EMPTY",
                             base_url=self.base_url
                             )
        else:
            llm = ChatOpenAI(model=self.engine, temperature=0.)

        result = llm.invoke([HumanMessage(content=instruct)])
        response = self.parse_thinking(result)

        message = {"role": "assistant",
                   "content": response['response']
                   }
        self.dialog_history.append(dict(message))

        return message["content"]

    def retrieve(self, prompt):

        prompt = {"role": "user", "content": prompt}
        dialog_history = deepcopy(self.dialog_history)
        dialog_history.append(prompt)
        scene = f"{self.initial_dialog_history[0]['content']}\n\n"

        context = ""
        for h in dialog_history[3:]:
            context += '%s:  %s \n' % (
                h["role"].replace('user', 'Bob').replace('assistant', 'Alice'), h["content"])

        instruct = INSTRUCT_DECISION_SCENE_TEMPLATE.format(
            player_name='Alice',
            history_dialogue=context,
        )
        self.retrieve_tool.expansion(instruct)
        if self.retrieve_tool.search_model == 'random':
            labels = self.retrieve_tool.random_search(num_paths=5)
        elif self.retrieve_tool.search_model == 'similarity':
            labels = self.retrieve_tool.embed_search(scene=scene + instruct, beam_width=5)
        else:
            labels = self.retrieve_tool.search(scene=scene + instruct, beam_width=5)

        path = [f"{node.goal}: {node.detail}" for node in self.retrieve_tool.cur_nodes if node.chosen]

        guidance_list = []
        structured_text = f"\nHere are some sub-goals and guidance derived from your primary objective: \n"
        temp_str = ""
        for label, guidance in zip(labels, path):
            temp_str += f" - {guidance}\n"  # details must be less than 30 words
            guidance_list.append(f"{label}: {guidance}")

        structured_text += temp_str
        structured_text += "\nIn this round, You should try to improve your negotiation strategy based on these " \
                           "guidance, in order to achieve your primary objective.\n "

        self.guidance = structured_text
        self.guidance_history.append(guidance_list)
        self.openai_cost += self.retrieve_tool.openai_cost
        self.dialog_history[1]['content'] = self.initial_dialog_history[1]['content'] + '\n\n' + self.guidance
        self.dialog_history[2]['content'] = self.initial_dialog_history[2][
                                                'content'] + '\n And I will try to improve my negotiation strategy based on the guidance, in order to minimize the profit gap between myself and my partner.'
        wprint(self.guidance, color='magenta')

    def feedback(self, result):
        """Reset dialog history and provide feedback based on the result."""
        if result["final_proposal"]:
            missing_keys = [key for key in ["alice_items", "bob_items"] if key not in result["final_proposal"]]
            if missing_keys:
                text = ", ".join(f"Missing {key} items" for key in missing_keys)
            else:
                alice_items = " ".join(f"{cnt} {item}" for item, cnt in result["final_proposal"]["alice_items"].items())
                bob_items = " ".join(f"{cnt} {item}" for item, cnt in result["final_proposal"]["bob_items"].items())
                text = f"Alice receives: {alice_items}. Bob receives: {bob_items}."
        else:
            text = 'No parsable proposal.'

        msgs = FEEDBACK_TEMPLATE.format(text=text,
                                        errors=result["profits"])

        self.initial_dialog_history[1]['content'] += '\n\n' + msgs
        self.dialog_history = deepcopy(self.initial_dialog_history)
        self.openai_cost = 0
        self.llm_token_count = 0
        self.guidance = ""
        self.guidance_history = []


class ModeratorAgent(DialogAgent):
    """NOTE: initial experiments shows that the moderator is much better at recognizing deal than not deal
    Do not know why but interesting 
    """

    def __init__(self,
                 initial_dialog_history=None,
                 agent_type="moderator",
                 engine="gpt-3.5-turbo",
                 api_key="",
                 trace_n_history=2,
                 ):
        """Initialize the moderator agent"""
        super().__init__(initial_dialog_history=initial_dialog_history,
                         agent_type=agent_type,
                         engine=engine,
                         api_key=api_key
                         )

        self.trace_n_history = trace_n_history
        print("Initializing moderator with engine %s" % self.engine)
        return

    def moderate(self,
                 dialog_history, who_was_last="buyer",
                 retry=True):
        """Moderate the conversation between the buyer and the seller"""
        history_len = len(dialog_history)
        if who_was_last == "buyer":
            prompt = "Alice: %s\n" % dialog_history[history_len - 1]["content"]
            offset = 1
        else:
            prompt = "Bob: %s\n" % dialog_history[history_len - 1]["content"]
            offset = 0

        for i in range(self.trace_n_history - 1):
            idx = history_len - i - 2
            content = dialog_history[idx]["content"]
            if i % 2 == offset:
                prompt = "Alice: %s\n" % content + prompt
            else:
                prompt = "Bob: %s\n" % content + prompt

        prompt += "question: have Alice and Bob achieved a deal? Yes or No\nanswer:"
        self.last_prompt = prompt

        messages = deepcopy(self.dialog_history)
        messages[-1]['content'] += "\n\n" + prompt

        response = self.call_engine(messages)
        return response["content"]

    def parse_proposal(self,
                       dialog_history, who_was_last="buyer",
                       retry=True):
        """Moderate the conversation between the buyer and the seller"""
        history_len = len(dialog_history)
        if who_was_last == "buyer":
            prompt = "Alice: %s\n" % dialog_history[history_len - 1]["content"]
            offset = 1
        else:
            prompt = "Bob: %s\n" % dialog_history[history_len - 1]["content"]
            offset = 0

        for i in range(self.trace_n_history - 1):
            idx = history_len - i - 2
            content = dialog_history[idx]["content"]
            if i % 2 == offset:
                prompt = "Alice: %s\n" % content + prompt
            else:
                prompt = "Bob: %s\n" % content + prompt

        prompt += "question: what's the proposal two party both agree with? \nanswer:"
        self.last_prompt = prompt

        messages = deepcopy(self.dialog_history)
        messages[-1]['content'] += "\n\n" + prompt

        response = self.call_engine(messages)

        return extract_jsons_from_text(response["content"])[-1]
