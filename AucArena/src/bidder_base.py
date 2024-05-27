from typing import List
from langchain.base_language import BaseLanguageModel
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.chat_models import (
    ChatAnthropic,
    ChatOpenAI,
    AzureChatOpenAI,
)
import openai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.input import get_colored_text
from langchain.callbacks import get_openai_callback
from collections import defaultdict
from copy import deepcopy
from pydantic import BaseModel
import queue
import threading
import os
import random
import time
import dill
import ujson as json
import matplotlib.pyplot as plt
from .item_base import Item, item_list_equal
from src.prompt_base import (
    AUCTION_HISTORY,
    # INSTRUCT_OBSERVE_TEMPLATE,
    _LEARNING_STATEMENT,
    INSTRUCT_PLAN_TEMPLATE,
    INSTRUCT_BID_TEMPLATE,
    INSTRUCT_SUMMARIZE_TEMPLATE,
    INSTRUCT_LEARNING_TEMPLATE,
    INSTRUCT_REPLAN_TEMPLATE,
    SYSTEM_MESSAGE,
    INSTRUCT_INIT_PLAN_SCENE_TEMPLATE,
    INSTRUCT_REPLAN_SCENE_TEMPLATE,
    INSTRUCT_REFLEXION_TEMPLATE,
    PARSE_GOALS_SYSTEM_MESSAGE,
)
from typing import ClassVar
import sys
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

from .retrieve_base import NodeRetrieve

sys.path.append('..')
from utils import LoadJsonL, extract_jsons_from_text, extract_numbered_list, trace_back, assign_labels_to_last_layer

# DESIRE_DESC = {
#     'default': "Your goal is to fully utilize your budget while actively participating in the auction",
#     'maximize_profit': "Your goal is to maximize your overall profit, and fully utilize your budget while actively participating in the auction. This involves strategic bidding to win items for less than their true value, thereby ensuring the difference between the price paid and the item's value is as large as possible",
#     'maximize_items': "Your goal is to win as many items as possible, and fully utilize your budget while actively participating in the auction. While keeping your budget in mind, you should aim to participate broadly across different items, striving to be the highest bidder more often than not",
# }   # remove period at the end of each description


DESIRE_DESC = {
    'maximize_profit': "Your primary objective is to secure the highest profit at the end of this auction, compared to all other bidders",
    'maximize_items': "Your primary objective is to win the highest number of items at the end of this auction, compared to everyone else",
}


class Bidder(BaseModel):
    name: str
    model_name: str
    budget: int
    desire: str
    plan_strategy: str
    temperature: float = 0.0
    overestimate_percent: int = 10
    correct_belief: bool
    enable_refine: bool = False
    enable_prune: bool = False
    enable_tool: bool = False

    base_url: str
    model_path: str

    learning_strategy: str = ''
    retrieve_strategy: str = ''
    max_depth: float = 0.8
    retrieve_tool: dict = {}
    beam_width: int = 5
    arms: dict = {}

    custom_api_base: str = ''

    llm: BaseLanguageModel = None
    openai_cost: int = 0
    llm_token_count: int = 0

    verbose: bool = False
    auction_hash: str = ''

    system_message: str = ''
    initial_system_message: str = ''
    original_budget: int = 0
    guidance: str = ''
    prune_list: list = []

    # working memory
    profit: int = 0
    cur_item_id: int = 0
    items: list = []
    dialogue_history: list = []  # for gradio UI display
    llm_prompt_history: list = []  # for tracking llm calling
    items_won: list = []
    bid_history: list = []  # history of the bidding of a single item
    plan_instruct: str = ''  # instruction for planning
    cur_plan: str = ''  # current plan
    status_quo: dict = {}  # belief of budget and profit, self and others
    past_auction_log: str = ''
    withdraw: bool = False  # state of withdraw
    learnings: str = ''  # learnings from previous biddings. If given, then use it to guide the rest of the auction.
    max_bid_cnt: int = 4  # Rule Bidder: maximum number of bids on one item (K = 1 starting bid + K-1 increase bid)
    rule_bid_cnt: int = 0  # Rule Bidder: count of bids on one item

    # belief tracking
    failed_bid_cnt: int = 0  # count of failed bids (overspending)
    total_bid_cnt: int = 0  # count of total bids
    self_belief_error_cnt: int = 0
    total_self_belief_cnt: int = 0
    other_belief_error_cnt: int = 0
    total_other_belief_cnt: int = 0

    engagement_count: int = 0
    guidance_history: list = []
    budget_history: list = []
    profit_history: list = []
    budget_error_history: list = []
    profit_error_history: list = []
    win_bid_error_history: list = []
    engagement_history: dict = defaultdict(int)
    all_bidders_status: dict = {}  # track others' profit
    changes_of_plan: list = []

    # not used
    input_box: str = None
    need_input: bool = False
    semaphore: int = 0

    class Config:
        arbitrary_types_allowed = True

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name

    @classmethod
    def create(cls, **data):
        instance = cls(**data)
        instance._post_init()
        return instance

    def _post_init(self):
        self.original_budget = self.budget
        self.system_message = SYSTEM_MESSAGE.format(
            name=self.name,
            desire_desc=DESIRE_DESC[self.desire],
        )
        self.initial_system_message = deepcopy(self.system_message)
        self._parse_llm()
        self.dialogue_history += [
            SystemMessage(content=self.system_message),
            AIMessage(content='')
        ]
        self.budget_history.append(self.budget)
        self.profit_history.append(self.profit)

        # with open('objective-tree', 'rb') as f:
        # objective_tree = dill.load(f)  # loaded from file
        # self.arms = assign_labels_to_last_layer(objective_tree, max_depth=3)

        if self.retrieve_strategy != 'none':
            self.retrieve_tool = NodeRetrieve(search_model=self.retrieve_strategy,
                                              model_path=self.model_path,
                                              base_url=self.base_url,
                                              max_depth=self.max_depth,
                                              enable_prune=self.enable_prune,
                                              )
            self.enable_tool = True

    #
    def _parse_llm(self):
        if 'azure' in self.model_name:
            # azure_gpt-35-turbo-16k, azure_gpt-4-32k
            model_name = self.model_name.replace('azure_', '')
            self.llm = AzureChatOpenAI(model=model_name, temperature=self.temperature,
                                       openai_api_base=self.custom_api_base,
                                       openai_api_version="2023-03-15-preview",
                                       deployment_name="gpt_openapi",
                                       openai_api_key=os.environ['AZURE_OPENAI_API_KEY'],
                                       openai_api_type="azure")
        elif 'gpt-' in self.model_name:
            self.llm = ChatOpenAI(model=self.model_name, temperature=self.temperature, max_retries=30,
                                  request_timeout=1200)
        elif 'claude' in self.model_name:
            self.llm = ChatAnthropic(model=self.model_name, temperature=self.temperature, default_request_timeout=1200)
        elif 'gemini' in self.model_name:

            self.llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=self.temperature,
                max_output_tokens=2048,
            )
        elif 'rule' in self.model_name or 'human' in self.model_name:
            self.llm = None
        elif self.model_name in ['open-mixtral-8x7b', 'mistral-small',
                                 'mistral-small-latest', 'mistral-medium-latest',
                                 'mistral-large-latest']:
            self.llm = ChatMistralAI(model=self.model_name,
                                     temperature=self.temperature,
                                     max_retries=30,
                                     request_timeout=1200)
        elif 'mistral' in self.model_name or 'mixtral' in self.model_name:
            self.llm = ChatOpenAI(
                model=self.model_path,
                temperature=0,
                api_key="EMPTY",
                base_url=self.base_url,
            )
        elif 'qwen-' in self.model_name:
            self.llm = ChatOpenAI(
                model=self.model_path,
                temperature=0,
                api_key="EMPTY",
                base_url=self.base_url,
            )

        elif 'llama-' in self.model_name:
            print(self.model_path)
            print(self.base_url)
            print(self.name)
            self.llm = ChatOpenAI(
                model=self.model_path,
                temperature=0,
                api_key="EMPTY",
                base_url=self.base_url,
            )

        else:
            # deploying local llms with `fastchat` for an openai-like API service, and use the url as the api_base.
            assert self.custom_api_base != '', f"Custom API base url for Llama-ish models can't be '{self.custom_api_base}'."
            model_name = self.model_name
            self.llm = ChatOpenAI(model=model_name,
                                  temperature=self.temperature,
                                  api_key="EMPTY",
                                  max_retries=30,
                                  request_timeout=1200,
                                  openai_api_base=self.custom_api_base
                                  )

    def _rotate_openai_org(self):
        # use two organizations to avoid rate limit
        if os.environ.get('OPENAI_ORGANIZATION_1') and os.environ.get('OPENAI_ORGANIZATION_2'):
            return random.choice([os.environ.get('OPENAI_ORGANIZATION_1'), os.environ.get('OPENAI_ORGANIZATION_2')])
        else:
            return None

    def _run_llm_standalone(self, messages: list):

        # with get_openai_callback() as cb:
        # for i in range(6):
        # try:
        # gemini & mixtral doesn't support system message.
        _msgs = [HumanMessage(content=messages[0].content), AIMessage(content='Got it!')] + messages[1:]
        # try:
        # input_token_num = self.llm.get_num_tokens_from_messages(_msgs)
        # except:
        # print(f'WARNING: {self.model_name} has not implemented `get_num_tokens_from_messages`')

        if 'claude' in self.model_name:  # anthropic's claude
            result = self.llm(_msgs, max_tokens_to_sample=2048)
        elif 'gemini' in self.model_name or 'mixtral' in self.model_name.lower():
            result = self.llm(_msgs)
        elif 'gpt' in self.model_name:  # openai
            if 'gpt-4-0613' in self.model_name:
                max_tokens = min(2048, max(8000 - input_token_num, 40))
            elif 'gpt-3.5-turbo-16k-0613' in self.model_name or 'gpt-3.5-turbo-0613' in self.model_name:
                max_tokens = min(2048, max(16385 - input_token_num, 192))
            else:
                max_tokens = 2048
                result = self.llm(_msgs, max_tokens=max_tokens)
        elif 'mistral' in self.model_name or 'mixtral' in self.model_name:
            result = self.llm(_msgs)

        elif 'qwen' in self.model_name or 'qwen-' in self.model_name:
            result = self.llm(_msgs)
        else:
            result = self.llm(_msgs)

        return result.content.strip()

    def _get_estimated_value(self, item):
        value = item.true_value * (1 + self.overestimate_percent / 100)
        return int(value)

    def _get_cur_item(self, key=None):
        if self.cur_item_id < len(self.items):
            if key is not None:
                return self.items[self.cur_item_id].__dict__[key]
            else:
                return self.items[self.cur_item_id]
        else:
            return 'no item left'

    def _get_next_item(self, key=None):
        if self.cur_item_id + 1 < len(self.items):
            if key is not None:
                return self.items[self.cur_item_id + 1].__dict__[key]
            else:
                return self.items[self.cur_item_id + 1]
        else:
            return 'no item left'

    def _get_remaining_items(self, as_str=False):
        remain_items = self.items[self.cur_item_id + 1:]
        if as_str:
            return ', '.join([item.name for item in remain_items])
        else:
            return remain_items

    def _get_items_value_str(self, items: List[Item]):
        if not isinstance(items, list):
            items = [items]
        items_info = ''
        for i, item in enumerate(items):
            estimated_value = self._get_estimated_value(item)
            _info = f"{i + 1}. {item}, starting price is ${item.price}. Your estimated value for this item is ${estimated_value}.\n"
            items_info += _info
        return items_info.strip()

    # ********** Main Instructions and Functions ********** #

    def learn_from_prev_auction(self, past_auction_log):
        if 'rule' in self.model_name or 'human' in self.model_name:
            return ''

        if self.learning_strategy == 'reflexion':
            instruct_learn = INSTRUCT_REFLEXION_TEMPLATE.format(
                past_auction_log=past_auction_log)
            result = self._run_llm_standalone([HumanMessage(content=instruct_learn)])
            self.dialogue_history += [
                HumanMessage(content=instruct_learn),
                AIMessage(content=result),
            ]
            self.llm_prompt_history.append({
                'messages': [{x.type: x.content} for x in [HumanMessage(content=instruct_learn)]],
                'result': result,
                'tag': 'learn_0'
            })
            self.learnings += 'Reflections:\n-' + result + '\n'

        elif self.learning_strategy == 'clin':
            instruct_learn = INSTRUCT_LEARNING_TEMPLATE.format(
                past_auction_log=past_auction_log,
                past_learnings=self.learnings,
            )

            result = self._run_llm_standalone([HumanMessage(content=instruct_learn)])
            self.dialogue_history += [
                HumanMessage(content=instruct_learn),
                AIMessage(content=result),
            ]
            self.llm_prompt_history.append({
                'messages': [{x.type: x.content} for x in [HumanMessage(content=instruct_learn)]],
                'result': result,
                'tag': 'learn_0'
            })

            self.learnings += 'Learnings:\n-' + result + '\n'

        if self.learnings != '':
            self.system_message = self.initial_system_message + f"\n\nHere are your key learning points and practical tips from a previous " \
                                                                f"auction. You can use them to guide this auction:\n```\n{self.learnings}\n```"

        if self.verbose:
            print(f"Learn from previous auction: {self.name} ({self.model_name}).")

        return self.learnings

    def retrieve_from_objective_tree(self, system_msg, instruct, detail=True):

        self.retrieve_tool.expansion(instruct)
        if self.retrieve_tool.search_model == 'random':
            labels = self.retrieve_tool.random_search(num_paths=5)
        elif self.retrieve_tool.search_model == 'similarity':
            labels = self.retrieve_tool.embed_search(scene=system_msg + instruct,
                                                         beam_width=self.beam_width)
        else:
            labels = self.retrieve_tool.search(scene=system_msg + instruct,
                                                   beam_width=self.beam_width)
        path = [f"{node.goal}: {node.detail}" for node in self.retrieve_tool.cur_nodes if node.chosen]

        guidance_list = [f"{self.cur_item_id + 1} ({self._get_cur_item('name')})"]
        structured_text = f"\nHere are some possible subgoals and guidance derived from your primary objective: \n"

        temp_str = ""
        for label, guidance in zip(labels, path):
            temp_str += f" - {guidance}\n"  # details must be less than 30 words
            guidance_list.append(f"{label}: {guidance}")

        structured_text += temp_str
        structured_text += "\nIn this round, You may target some of these subgoals and guidance to improve your " \
                           "bidding strategy and action, in order to achieve your primary objective.\n"

        self.guidance = structured_text
        self.guidance_history.append(guidance_list)
        self.openai_cost += self.retrieve_tool.openai_cost

    def _choose_items(self, budget, items: List[Item]):
        '''
        Choose items within budget for rule bidders.
        Cheap ones first if maximize_items, expensive ones first if maximize_profit.
        '''
        sorted_items = sorted(items, key=lambda x: self._get_estimated_value(x),
                              reverse=self.desire == 'maximize_profit')

        chosen_items = []
        i = 0
        while budget >= 0 and i < len(sorted_items):
            item = sorted_items[i]
            if item.price <= budget:
                chosen_items.append(item)
                budget -= item.price
            i += 1

        return chosen_items

    def get_plan_instruct(self, items: List[Item]):
        self.items = items
        plan_instruct = INSTRUCT_PLAN_TEMPLATE.format(
            bidder_name=self.name,
            budget=self.budget,
            item_num=len(items),
            items_info=self._get_items_value_str(items),
            desire_desc=DESIRE_DESC[self.desire],
            learning_statement='' if self.learning_strategy == 'none' else _LEARNING_STATEMENT,
            guidance='' if not self.enable_tool else "Consider the given sub-goals and detailed "
                                                     "advices, You can focus on some of them to help "
                                                     "you plan."
        )
        return plan_instruct

    def init_plan(self, plan_instruct: str):
        '''
        Plan for bidding with auctioneer's instruction and items information for customize estimated value.
        plan = plan(system_message, instruct_plan)
        '''
        print(self.name, self.guidance)
        if self.enable_tool:
            instruct = INSTRUCT_INIT_PLAN_SCENE_TEMPLATE.format(
                bidder_name=self.name,
                budget=self.budget,
                item_num=len(self.items),
                items_info=self._get_items_value_str(self.items),
            )
            self.retrieve_from_objective_tree(self.system_message, instruct)
            print(self.name, self.guidance)

        if 'rule' in self.model_name:
            # self.cur_plan = ', '.join([x.name for x in self._choose_items(self.budget, self.items)])
            # self.dialogue_history += [
            #     HumanMessage(content=plan_instruct),
            #     AIMessage(content=self.cur_plan),
            # ]
            # return self.cur_plan
            return ''

        self.status_quo = {
            'remaining_budget': self.budget,
            'total_profits': {bidder: 0 for bidder in self.all_bidders_status.keys()},
            'winning_bids': {bidder: {} for bidder in self.all_bidders_status.keys()},
        }

        if self.plan_strategy == 'none':
            self.plan_instruct = ''
            self.cur_plan = ''
            return None

        system_msg = SystemMessage(content=self.system_message + self.guidance)
        plan_msg = HumanMessage(content=plan_instruct)
        messages = [system_msg, plan_msg]
        result = self._run_llm_standalone(messages)

        if self.verbose:
            print(get_colored_text(plan_msg.content, 'red'))
            print(get_colored_text(result, 'green'))

        self.dialogue_history += [
            plan_msg,
            AIMessage(content=result),
        ]
        self.llm_prompt_history.append({
            'messages': [{x.type: x.content} for x in messages],
            'result': result,
            'tag': 'plan_0'
        })
        self.cur_plan = result
        self.plan_instruct = plan_instruct

        self.changes_of_plan.append([
            f"{self.cur_item_id} (Initial)",
            False,
            json.dumps(extract_jsons_from_text(result)[-1]),
        ])

        if self.verbose:
            print(f"Plan: {self.name} ({self.model_name}) for {self._get_cur_item()}.")
        return result

    def get_rebid_instruct(self, auctioneer_msg: str):
        self.dialogue_history += [
            HumanMessage(content=auctioneer_msg),
            AIMessage(content='')
        ]
        return auctioneer_msg

    def get_bid_instruct(self, auctioneer_msg: str, bid_round: int):
        auctioneer_msg = auctioneer_msg.replace(self.name, f'You ({self.name})')

        bid_instruct = INSTRUCT_BID_TEMPLATE.format(
            auctioneer_msg=auctioneer_msg,
            bidder_name=self.name,
            cur_item=self._get_cur_item(),
            estimated_value=self._get_estimated_value(self._get_cur_item()),
            desire_desc=DESIRE_DESC[self.desire],
            learning_statement='' if self.learning_strategy == 'none' else _LEARNING_STATEMENT,
            guidance="3. Consider the given sub-goals and detailed advices, you may choose to focus on some of them "
                     "to help you decide on your action, or ignore them." if self.enable_tool else ''
        )
        if bid_round == 0:
            if self.plan_strategy in ['static', 'none']:
                # if static planner, then no replanning is needed. status quo is updated in replanning. thus need to
                # add status quo in bid instruct.
                bid_instruct = f"""The status quo of this auction so far is:\n"{json.dumps(self.status_quo, indent=4)}"\n\n{bid_instruct}\n---\n"""
        else:
            bid_instruct = f'Now, the auctioneer says: "{auctioneer_msg}"'

        self.dialogue_history += [
            HumanMessage(content=bid_instruct),
            AIMessage(content='')
        ]
        return bid_instruct

    def bid_rule(self, cur_bid: int, min_markup_pct: float = 0.1):
        '''
        :param cur_bid: current highest bid
        :param min_markup_pct: minimum percentage for bid increase
        :param max_bid_cnt: maximum number of bids on one item (K = 1 starting bid + K-1 increase bid)
        '''
        # dialogue history already got bid_instruction.
        cur_item = self._get_cur_item()

        if cur_bid <= 0:
            next_bid = cur_item.price
        else:
            next_bid = cur_bid + min_markup_pct * cur_item.price

        if self.budget - next_bid >= 0 and self.rule_bid_cnt < self.max_bid_cnt:
            msg = int(next_bid)
            self.rule_bid_cnt += 1
        else:
            msg = -1

        content = f'The current highest bid for {cur_item.name} is ${cur_bid}. '
        content += "I'm out!" if msg < 0 else f"I bid ${msg}! (Rule generated)"
        self.dialogue_history += [
            HumanMessage(content=''),
            AIMessage(content=content)
        ]

        return msg

    def bid(self, bid_instruct):
        '''
        Bid for an item with auctioneer's instruction and bidding history.
        bid_history = bid(system_message, instruct_plan, plan, bid_history)
        '''
        if self.model_name == 'rule':
            return ''

        bid_msg = HumanMessage(content=bid_instruct)

        if self.plan_strategy == 'none':
            messages = [SystemMessage(content=self.system_message + self.guidance)]
        else:
            messages = [SystemMessage(content=self.system_message + self.guidance),
                        HumanMessage(content=self.plan_instruct),
                        AIMessage(content=self.cur_plan)]

        self.bid_history += [bid_msg]
        messages += self.bid_history
        result = self._run_llm_standalone(messages)

        self.bid_history += [AIMessage(content=result)]

        self.dialogue_history += [
            HumanMessage(content=''),
            AIMessage(content=result)
        ]

        self.llm_prompt_history.append({
            'messages': [{x.type: x.content} for x in messages],
            'result': result,
            'tag': f'bid_{self.cur_item_id}'
        })

        if self.verbose:
            print(get_colored_text(bid_instruct, 'yellow'))
            print(get_colored_text(result, 'green'))

            print(f"Bid: {self.name} ({self.model_name}) for {self._get_cur_item()}.")
        self.total_bid_cnt += 1

        return result

    def get_summarize_instruct(self, bidding_history: str, hammer_msg: str, win_lose_msg: str):
        instruct = INSTRUCT_SUMMARIZE_TEMPLATE.format(
            cur_item=self._get_cur_item(),
            bidding_history=bidding_history,
            hammer_msg=hammer_msg.strip(),
            win_lose_msg=win_lose_msg.strip(),
            bidder_name=self.name,
            prev_status=self._status_json_to_text(self.status_quo),
        )
        return instruct

    def summarize(self, instruct_summarize: str):
        '''
        Update belief/status quo
        status_quo = summarize(system_message, bid_history, prev_status + instruct_summarize)
        '''
        self.budget_history.append(self.budget)
        self.profit_history.append(self.profit)

        if self.model_name == 'rule':
            self.rule_bid_cnt = 0  # reset bid count for rule bidder
            return ''

        messages = [SystemMessage(content=self.system_message + self.guidance)]
        # messages += self.bid_history
        summ_msg = HumanMessage(content=instruct_summarize)
        messages.append(summ_msg)

        status_quo_text = self._run_llm_standalone(messages)

        self.dialogue_history += [summ_msg, AIMessage(content=status_quo_text)]
        self.bid_history += [summ_msg, AIMessage(content=status_quo_text)]

        self.llm_prompt_history.append({
            'messages': [{x.type: x.content} for x in messages],
            'result': status_quo_text,
            'tag': f'summarize_{self.cur_item_id}'
        })

        cnt = 0
        while cnt <= 3:
            sanity_msg = self._sanity_check_status_json(extract_jsons_from_text(status_quo_text)[-1])
            if sanity_msg == '':
                # pass sanity check then track beliefs
                consistency_msg = self._belief_tracking(status_quo_text)
            else:
                sanity_msg = f'- {sanity_msg}'
                consistency_msg = ''

            if sanity_msg != '' or (consistency_msg != '' and self.correct_belief):
                err_msg = f"As {self.name}, here are some error(s) of your summary of the status JSON:\n{sanity_msg.strip()}\n{consistency_msg.strip()}\n\nPlease revise the status JSON based on the errors. Don't apologize. Just give me the revised status JSON.".strip()

                # print(f"{self.name}: revising status quo for the {cnt} time:")
                # print(get_colored_text(err_msg, 'green'))
                # print(get_colored_text(status_quo_text, 'red'))

                messages += [AIMessage(content=status_quo_text),
                             HumanMessage(content=err_msg)]
                status_quo_text = self._run_llm_standalone(messages)
                self.dialogue_history += [
                    HumanMessage(content=err_msg),
                    AIMessage(content=status_quo_text),
                ]
                cnt += 1
            else:
                break

        self.status_quo = extract_jsons_from_text(status_quo_text)[-1]

        if self.verbose:
            print(get_colored_text(instruct_summarize, 'blue'))
            print(get_colored_text(status_quo_text, 'green'))

            print(f"Summarize: {self.name} ({self.model_name}) for {self._get_cur_item()}.")

        # if self.enable_learning:
        #     self.learn(instruct_summarize, status_quo_text)

        return status_quo_text

    def get_replan_instruct(self, past_auction_log):
        self.past_auction_log = past_auction_log
        instruct = INSTRUCT_REPLAN_TEMPLATE.format(
            status_quo=self._status_json_to_text(self.status_quo),
            remaining_items_info=self._get_items_value_str(self._get_remaining_items()),
            bidder_name=self.name,
            desire_desc=DESIRE_DESC[self.desire],
            learning_statement='' if self.learning_strategy == 'none' else _LEARNING_STATEMENT,
            guidance='' if not self.enable_tool else "Consider the given sub-goals and detailed advices, You can "
                                                     "focus on some of them to help you plan.",
        )
        return instruct

    def replan(self, instruct_replan: str):
        '''
        plan = replan(system_message, instruct_plan, prev_plan, status_quo + (learning) + instruct_replan)
        '''

        if self.enable_tool:
            instruct = INSTRUCT_REPLAN_SCENE_TEMPLATE.format(
                bidder_name=self.name,
                status_quo=self._status_json_to_text(self.status_quo),
                remaining_items_info=self._get_items_value_str(self._get_remaining_items()),
                past_auction_log=self.past_auction_log,
            )
            self.retrieve_from_objective_tree(self.system_message, instruct)
            print(self.name, self.guidance)

        if self.model_name == 'rule':
            self.withdraw = False
            self.cur_item_id += 1
            return ''

        if self.plan_strategy in ['none', 'static']:
            self.bid_history = []  # clear bid history
            self.cur_item_id += 1
            self.withdraw = False

            return 'Skip replanning for bidders with static or no plan.'

        replan_msg = HumanMessage(content=instruct_replan)

        messages = [SystemMessage(content=self.system_message + self.guidance),
                    HumanMessage(content=self.plan_instruct),
                    AIMessage(content=self.cur_plan)]
        messages.append(replan_msg)

        result = self._run_llm_standalone(messages)

        new_plan_dict = extract_jsons_from_text(result)[-1]
        cnt = 0
        while len(new_plan_dict) == 0 and cnt < 2:
            err_msg = 'Your response does not contain a JSON-format priority list for items. Please revise your plan.'
            messages += [
                AIMessage(content=result),
                HumanMessage(content=err_msg),
            ]
            result = self._run_llm_standalone(messages)
            new_plan_dict = extract_jsons_from_text(result)[-1]

            self.dialogue_history += [
                HumanMessage(content=err_msg),
                AIMessage(content=result),
            ]
            cnt += 1

        old_plan_dict = extract_jsons_from_text(self.cur_plan)[-1]
        self.changes_of_plan.append([
            f"{self.cur_item_id + 1} ({self._get_cur_item('name')})",
            self._change_of_plan(old_plan_dict, new_plan_dict),
            json.dumps(new_plan_dict)
        ])

        self.plan_instruct = instruct_replan
        self.cur_plan = result
        self.withdraw = False
        self.bid_history = []  # clear bid history
        self.cur_item_id += 1

        self.dialogue_history += [
            replan_msg,
            AIMessage(content=result),
        ]
        self.llm_prompt_history.append({
            'messages': [{x.type: x.content} for x in messages],
            'result': result,
            'tag': f'plan_{self.cur_item_id}'
        })

        if self.verbose:
            print(get_colored_text(instruct_replan, 'blue'))
            print(get_colored_text(result, 'green'))

            print(f"Replan: {self.name} ({self.model_name}).")
        return result

    def _change_of_plan(self, old_plan: dict, new_plan: dict):
        for k in new_plan:
            if new_plan[k] != old_plan.get(k, None):
                return True
        return False

    # *********** Belief Tracking and Sanity Check *********** #

    def bid_sanity_check(self, bid_price, prev_round_max_bid, min_markup_pct):
        # can't bid more than budget or less than previous highest bid
        if bid_price < 0:
            msg = None
        else:
            min_bid_increase = int(min_markup_pct * self._get_cur_item('price'))
            if bid_price > self.budget:
                msg = f"you don't have insufficient budget (${self.budget} left)"
            elif bid_price < self._get_cur_item('price'):
                msg = f"your bid is lower than the starting bid (${self._get_cur_item('price')})"
            elif bid_price < prev_round_max_bid + min_bid_increase:
                msg = f"you must advance previous highest bid (${prev_round_max_bid}) by at least ${min_bid_increase} ({int(100 * min_markup_pct)}%)."
            elif self.budget < self._get_cur_item('price'):
                msg = f"your don't have insufficient budget(${self.budget} left) to attend the bidding war of this item, you must out!"
            else:
                msg = None
        return msg

    def rebid_for_failure(self, fail_instruct: str):
        result = self.bid(fail_instruct)
        self.failed_bid_cnt += 1
        return result

    def _sanity_check_status_json(self, data: dict):
        if data == {}:
            return "Error: No parsible JSON in your response. Possibly due to missing a closing curly bracket '}', or unpasible values (e.g., 'profit': 1000 + 400, instead of 'profit': 1400)."

        # Check if all expected top-level keys are present
        expected_keys = ["remaining_budget", "total_profits", "winning_bids"]
        for key in expected_keys:
            if key not in data:
                return f"Error: Missing '{key}' field in the status JSON."

        # Check if "remaining_budget" is a number
        if not isinstance(data["remaining_budget"], (int, float)):
            return "Error: 'remaining_budget' should be a number, and only about your remaining budget."

        # Check if "total_profits" is a dictionary with numbers as values
        if not isinstance(data["total_profits"], dict):
            return "Error: 'total_profits' should be a dictionary of every bidder."
        for bidder, profit in data["total_profits"].items():
            if not isinstance(profit, (int, float)):
                return f"Error: Profit for {bidder} should be a number."

        # Check if "winning_bids" is a dictionary and that each bidder's entry is a dictionary with numbers
        if not isinstance(data["winning_bids"], dict):
            return "Error: 'winning_bids' should be a dictionary."
        for bidder, bids in data["winning_bids"].items():
            if not isinstance(bids, dict):
                return f"Error: Bids for {bidder} should be a dictionary."
            for item, amount in bids.items():
                if not isinstance(amount, (int, float)):
                    return f"Error: Amount for {item} under {bidder} should be a number."

        # If everything is fine
        return ""

    def _status_json_to_text(self, data: dict):
        if 'rule' in self.model_name: return ''

        # Extract and format remaining budget
        structured_text = f"* Remaining Budget: ${data.get('remaining_budget', 'unknown')}\n\n"

        # Extract and format total profits for each bidder
        structured_text += "* Total Profits:\n"
        if data.get('total_profits'):
            for bidder, profit in data['total_profits'].items():
                structured_text += f"  * {bidder}: ${profit}\n"

        # Extract and list the winning bids for each item by each bidder
        structured_text += "\n* Winning Bids:\n"
        if data.get('winning_bids'):
            for bidder, bids in data['winning_bids'].items():
                structured_text += f"  * {bidder}:\n"
                if bids:
                    for item, amount in bids.items():
                        structured_text += f"    * {item}: ${amount}\n"
                else:
                    structured_text += f"    * No winning bids\n"

        return structured_text.strip()

    def _belief_tracking(self, status_text: str):
        '''
        Parse status quo and check if the belief is correct.
        '''
        belief_json = extract_jsons_from_text(status_text)[-1]
        # {"remaining_budget": 8000, "total_profits": {"Bidder 1": 1300, "Bidder 2": 1800, "Bidder 3": 0}, "winning_bids": {"Bidder 1": {"Item 2": 1200, "Item 3": 1000}, "Bidder 2": {"Item 1": 2000}, "Bidder 3": {}}}
        budget_belief = belief_json['remaining_budget']
        profits_belief = belief_json['total_profits']
        winning_bids = belief_json['winning_bids']

        msg = ''
        # track belief of budget
        self.total_self_belief_cnt += 1
        if budget_belief != self.budget:
            msg += f'- Your belief of budget is wrong: you have ${self.budget} left, but you think you have ${budget_belief} left.\n'
            self.self_belief_error_cnt += 1
            self.budget_error_history.append([
                self._get_cur_item('name'),
                budget_belief,
                self.budget,
            ])

        # track belief of profits
        for bidder_name, profit in profits_belief.items():
            if self.all_bidders_status.get(bidder_name) is None:
                # due to a potentially unreasonable parsing
                continue

            if self.name in bidder_name:
                bidder_name = self.name
                self.total_self_belief_cnt += 1
            else:
                self.total_other_belief_cnt += 1

            real_profit = self.all_bidders_status[bidder_name]['profit']

            if profit != real_profit:
                if self.name == bidder_name:
                    self.self_belief_error_cnt += 1
                else:
                    self.other_belief_error_cnt += 1

                msg += f'- Your belief of total profit of {bidder_name} is wrong: {bidder_name} has earned ${real_profit} so far, but you think {bidder_name} has earned ${profit}.\n'

                # add to history
                self.profit_error_history.append([
                    f"{bidder_name} ({self._get_cur_item('name')})",
                    profit,
                    real_profit
                ])

        # track belief of winning bids
        for bidder_name, items_won_dict in winning_bids.items():
            if self.all_bidders_status.get(bidder_name) is None:
                # due to a potentially unreasonable parsing
                continue

            real_items_won = self.all_bidders_status[bidder_name]['items_won']
            # items_won = [(item, bid_price), ...)]

            items_won_list = list(items_won_dict.keys())
            real_items_won_list = [str(x) for x, _ in real_items_won]

            if self.name in bidder_name:
                self.total_self_belief_cnt += 1
            else:
                self.total_other_belief_cnt += 1

            if not item_list_equal(items_won_list, real_items_won_list):
                if bidder_name == self.name:
                    self.self_belief_error_cnt += 1
                    _bidder_name = f'you'
                else:
                    self.other_belief_error_cnt += 1
                    _bidder_name = bidder_name

                msg += f"- Your belief of winning items of {bidder_name} is wrong: {bidder_name} won {real_items_won}, but you think {bidder_name} won {items_won_dict}.\n"

                self.win_bid_error_history.append([
                    f"{_bidder_name} ({self._get_cur_item('name')})",
                    ', '.join(items_won_list),
                    ', '.join(real_items_won_list)
                ])

        return msg

    def win_bid(self, item: Item, bid: int):
        self.budget -= bid
        self.profit += item.true_value - bid
        self.items_won += [[item, bid]]
        msg = f"Congratuations! You won {item} at ${bid}."  # Now you have ${self.budget} left. Your total profit so far is ${self.profit}."
        return msg

    def lose_bid(self, item: Item):
        return f"You lost {item}."  # Now, you have ${self.budget} left. Your total profit so far is ${self.profit}."

    # set the profit information of other bidders
    def set_all_bidders_status(self, all_bidders_status: dict):
        self.all_bidders_status = all_bidders_status.copy()

    def set_withdraw(self, bid: int):
        if bid < 0:  # withdraw
            self.withdraw = True
        elif bid == 0:  # enable discount and bid again
            self.withdraw = False
        else:  # normal bid
            self.withdraw = False
            self.engagement_count += 1
            self.engagement_history[self._get_cur_item('name')] += 1


    def profit_report(self):
        '''
        Personal profit report at the end of an auction.
        '''
        msg = f"* {self.name}, starting with ${self.original_budget}, has won {len(self.items_won)} items in this auction, with a total profit of ${self.profit}.:\n"
        profit = 0
        for item, bid in self.items_won:
            profit += item.true_value - bid
            msg += f"  * Won {item} at ${bid} over ${item.price}, with a true value of ${item.true_value}.\n"
        return msg.strip()

    def to_monitors(self, as_json=False):
        # budget, profit, items_won, tokens
        if len(self.items_won) == 0 and not as_json:
            items_won = [['', 0, 0]]
        else:
            items_won = []
            for item, bid in self.items_won:
                items_won.append([str(item), bid, item.true_value])

        profit_error_history = self.profit_error_history if self.profit_error_history != [] or as_json else [
            ['', '', '']]
        win_bid_error_history = self.win_bid_error_history if self.win_bid_error_history != [] or as_json else [
            ['', '', '']]
        budget_error_history = self.budget_error_history if self.budget_error_history != [] or as_json else [['', '']]
        changes_of_plan = self.changes_of_plan if self.changes_of_plan != [] or as_json else [['', '', '']]
        cnt = self.retrieve_tool.cnt if self.enable_tool else []
        self.prune_list = [f"{node.goal}: {node.detail}" for node in self.retrieve_tool.prune_list] if self.enable_prune else []

        if as_json:
            return {
                'auction_hash': self.auction_hash,
                'bidder_name': self.name,
                'model_name': self.model_name,
                'desire': self.desire,
                'plan_strategy': self.plan_strategy,
                'overestimate_percent': self.overestimate_percent,
                'temperature': self.temperature,
                'correct_belief': self.correct_belief,
                'enable_tool': self.enable_tool,
                'enable_refine': self.enable_refine,
                'enable_prune': self.enable_prune,
                'learning_strategy': self.learning_strategy,
                'retrieve_strategy': self.retrieve_strategy,
                'beam_width': self.beam_width,
                'budget': self.original_budget,
                'money_left': self.budget,
                'profit': self.profit,
                'items_won': items_won,
                'tokens_used': self.llm_token_count,
                'openai_cost': round(self.openai_cost, 2),
                'failed_bid_cnt': self.failed_bid_cnt,
                'self_belief_error_cnt': self.self_belief_error_cnt,
                'other_belief_error_cnt': self.other_belief_error_cnt,
                'failed_bid_rate': round(self.failed_bid_cnt / (self.total_bid_cnt + 1e-8), 2),
                'self_error_rate': round(self.self_belief_error_cnt / (self.total_self_belief_cnt + 1e-8), 2),
                'other_error_rate': round(self.other_belief_error_cnt / (self.total_other_belief_cnt + 1e-8), 2),
                'engagement_count': self.engagement_count,
                'engagement_history': self.engagement_history,
                'changes_of_plan': changes_of_plan,
                'budget_error_history': budget_error_history,
                'profit_error_history': profit_error_history,
                'win_bid_error_history': win_bid_error_history,
                'learnings': self.learnings,
                'prune_list': self.prune_list,
                'changes_of_cnt': cnt,
                'arms': self.arms,
                'guidance': self.guidance,
                'guidance_history': self.guidance_history,
                'history': self.llm_prompt_history
            }
        else:
            return [
                self.budget,
                self.profit,
                items_won,
                self.llm_token_count,
                round(self.openai_cost, 2),
                round(self.failed_bid_cnt / (self.total_bid_cnt + 1e-8), 2),
                round(self.self_belief_error_cnt / (self.total_self_belief_cnt + 1e-8), 2),
                round(self.other_belief_error_cnt / (self.total_other_belief_cnt + 1e-8), 2),
                self.engagement_count,
                draw_plot(f"{self.name} ({self.model_name})", self.budget_history, self.profit_history),
                changes_of_plan,
                budget_error_history,
                profit_error_history,
                win_bid_error_history
            ]

    def dialogue_to_chatbot(self):
        # chatbot: [[Human, AI], [], ...]
        # only dialogue will be sent to LLMs. chatbot is just for display.
        assert len(self.dialogue_history) % 2 == 0
        chatbot = []
        for i in range(0, len(self.dialogue_history), 2):
            # if exceeds the length of dialogue, append the last message
            human_msg = self.dialogue_history[i].content
            ai_msg = self.dialogue_history[i + 1].content
            if ai_msg == '': ai_msg = None
            if human_msg == '': human_msg = None
            chatbot.append([human_msg, ai_msg])
        return chatbot


def draw_plot(title, hedge_list, profit_list):
    x1 = [str(i) for i in range(len(hedge_list))]
    x2 = [str(i) for i in range(len(profit_list))]
    y1 = hedge_list
    y2 = profit_list

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Bidding Round')
    ax1.set_ylabel('Budget Left ($)', color=color)
    ax1.plot(x1, y1, color=color, marker='o')
    ax1.tick_params(axis='y', labelcolor=color)

    for i, j in zip(x1, y1):
        ax1.text(i, j, str(j), color=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Total Profit ($)', color=color)
    ax2.plot(x2, y2, color=color, marker='^')
    ax2.tick_params(axis='y', labelcolor=color)

    for i, j in zip(x2, y2):
        ax2.text(i, j, str(j), color=color)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc=0)

    # fig.tight_layout()
    plt.title(title)

    return fig


def bidding_multithread(bidder_list: List[Bidder],
                        instruction_list,
                        func_type,
                        thread_num=5,
                        retry=1):
    '''
    auctioneer_msg: either a uniform message (str) or customed (list)
    '''
    assert func_type in ['plan', 'bid', 'summarize', 'replan']

    result_queue = queue.Queue()
    threads = []
    semaphore = threading.Semaphore(thread_num)

    def run_once(i: int, bidder: Bidder, auctioneer_msg: str):
        try:
            semaphore.acquire()
            if func_type == 'bid':
                result = bidder.bid(auctioneer_msg)
            elif func_type == 'summarize':
                result = bidder.summarize(auctioneer_msg)
            elif func_type == 'plan':
                result = bidder.init_plan(auctioneer_msg)
            elif func_type == 'replan':
                result = bidder.replan(auctioneer_msg)
            else:
                raise NotImplementedError(f'func_type {func_type} not implemented')
            result_queue.put((True, i, result))
        finally:
            semaphore.release()

    if isinstance(instruction_list, str):
        instruction_list = [instruction_list] * len(bidder_list)

    for i, (bidder, msg) in enumerate(zip(bidder_list, instruction_list)):
        thread = threading.Thread(target=run_once, args=(i, bidder, msg))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join(timeout=600)

    results = [result_queue.get() for _ in range(len(bidder_list))]

    errors = []
    for success, id, result in results:
        if not success:
            errors.append((id, result))

    if errors:
        raise Exception(f"Error(s) in {func_type}:\n" + '\n'.join([f'{i}: {e}' for i, e in errors]))

    valid_results = [x[1:] for x in results if x[0]]
    valid_results.sort()

    return [x for _, x in valid_results]


def bidders_to_chatbots(bidder_list: List[Bidder], profit_report=False):
    if profit_report:  # usually at the end of an auction
        return [x.dialogue_to_chatbot() + [[x.profit_report(), None]] for x in bidder_list]
    else:
        return [x.dialogue_to_chatbot() for x in bidder_list]


def create_bidders(bidder_info_jsl, auction_hash, base_url, model_path):
    bidder_info_jsl = LoadJsonL(bidder_info_jsl)
    bidder_list = []
    for info in bidder_info_jsl:
        info['auction_hash'] = auction_hash
        info['base_url'] = base_url
        info['model_path'] = model_path
        bidder_list.append(Bidder.create(**info))

    return bidder_list
