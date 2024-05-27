from prompt_base import INSTRUCT_REFINE_TEMPLATE, INSTRUCT_REFLEXION_TEMPLATE, INSTRUCT_LEARNING_TEMPLATE, \
    INSTRUCT_DECISION_SCENE_TEMPLATE
from retrieve_base import NodeRetrieve
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

from utils import extract_numbered_list, wprint
import openai
import time
from tqdm import tqdm
import os
import json
import copy
import random
# from utils import *
import google.generativeai as genai
import matplotlib.pyplot as plt
import numpy as np
import dill
from openai import OpenAI
from typing import List
from collections import defaultdict
import queue
import threading
from utils import assign_labels_to_last_layer
from langchain.chat_models import (
    ChatAnthropic,
    ChatOpenAI,
    AzureChatOpenAI,
)

from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

DESIRE_DESC = {
    "guessing_game": "Your primary objective is to choose a number that you believe will be closest to 2/3 of the average of all numbers chosen by players, including your selection.",
    "public_goods": "Your primary objective is to maximize your own tokens at the end of game",
    "battle_royale": "Your primary objective is to eliminate other players to survive until the end and win the game.",
}


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def chat(
        model,  # gpt-4, gpt-4-0314, gpt-4-32k, gpt-4-32k-0314, gpt-3.5-turbo, gpt-3.5-turbo-0301
        messages,  # [{"role": "system"/"user"/"assistant", "content": "Hello!", "name": "example"}]
        temperature=0,  # [0, 2]: Lower values -> more focused and deterministic; Higher values -> more random.
        n=1,  # Chat completion choices to generate for each input message.
        max_tokens=1024,  # The maximum number of tokens to generate in the chat completion.
        delay=0.1  # Seconds to sleep after each request.
):
    time.sleep(delay)
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        n=n,
        max_tokens=max_tokens
    )
    if n == 1:
        return response.choices[0].message.content
    else:
        return [i.message.content for i in response.choice]


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion(
        model,  # text-davinci-003, text-davinci-002, text-curie-001, text-babbage-001, text-ada-001
        prompt,
        # The prompt(s) to generate completions for, encoded as a string, array of strings, array of tokens, or array of token arrays.
        temperature=0,  # [0, 2]: Lower values -> more focused and deterministic; Higher values -> more random.
        n=1,  # Completions to generate for each prompt.
        max_tokens=1024,  # The maximum number of tokens to generate in the chat completion.
        delay=0.1  # Seconds to sleep after each request.
):
    time.sleep(delay)

    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        temperature=temperature,
        n=n,
        max_tokens=max_tokens
    )

    if n == 1:
        return response['choices'][0]['text']
    else:
        response = response['choices']
        response.sort(key=lambda x: x['index'])
        return [i['text'] for i in response['choices']]


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def gemini_chat(
        model,
        # gemini-1.0-pro, gemini-1.0-pro-001, gemini-1.0-pro-latest, gemini-1.0-pro-vision-latest, gemini-pro, gemini-pro-vision
        messages,
        # [{'role': 'user', 'parts': "In one sentence, explain how a computer works to a young child."}, {'role': "model', 'parts': "A computer is like a very smart machine that can understand and follow our instructions, help us with our work, and even play games with us!"}
        temperature=0,  # [0, 2]: Lower values -> more focused and deterministic; Higher values -> more random.
        n=1,  # Chat response choices to generate for each input message.
        max_tokens=1024,  # The maximum number of tokens to generate in the chat completion.
        delay=0.1  # Seconds to sleep after each request.
):
    time.sleep(delay)
    model = genai.GenerativeModel(model)
    response = model.generate_content(
        messages,
        generation_config=genai.types.GenerationConfig(
            # Only one candidate for now.
            candidate_count=n,
            # stop_sequences=['x'],
            max_output_tokens=max_tokens,
            temperature=temperature)
    )

    if n == 1:
        return response.text


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def vllm_chat(
        base_url,
        msgs,
):
    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key="EMPTY",
        base_url=base_url,
    )
    models = client.models.list()
    model = models.data[0].id
    response = client.chat.completions.create(
        temperature=0,
        model=model,
        messages=msgs,
        top_logprobs=1,
        logprobs=True,
    )
    response = response.choices[0].message.content
    return response


def get_prompt(filename, inputs):
    with open(filename, 'r') as file:
        generated_prompt = file.read().split("<commentblockmarker>###</commentblockmarker>")[1].strip()
    for index, item in enumerate(inputs):
        key = f"!<INPUT {index}>!"
        generated_prompt = generated_prompt.replace(key, str(item))
    if "<part>///</part>" in generated_prompt:
        generated_prompt = [p.strip() for p in generated_prompt.split("<part>///</part>")]
    return generated_prompt


def get_rephrase_prompt(filename, inputs):
    with open(filename, 'r') as file:
        generated_prompt = file.read().split("<commentblockmarker>###</commentblockmarker>")[1].strip()
    for index, item in enumerate(inputs):
        key = f"!<REPHRASE_INPUT {index}>!"
        generated_prompt = generated_prompt.replace(key, str(item))
    return generated_prompt


def get_cot_prompt(cot):
    if cot:
        return " " + get_prompt(f"prompt_template/cot_prompts/cot{cot}.txt", [])
    else:
        return ""


def get_role_msg(role):
    if role:
        return " " + get_prompt(f"prompt_template/role_prompts/role{role}.txt", [])
    else:
        return ""


def select_players(player_list, attr_name, metric_list):
    if metric_list == 'ALL':
        return player_list
    else:
        return [player for player in player_list if getattr(player, attr_name, None) in metric_list]


class Player:
    def __init__(self, model,
                 model_path,
                 base_url,
                 id, prompt, learning_strategy,
                 enable_refine, retrieve_strategy,
                 records=None, utility=None, tokens=None, valuation=None):
        self.model = model
        self.model_path = model_path
        self.base_url = base_url
        self.id = id
        self.prompt = prompt
        self.initial_prompt = ''
        self.enable_refine = enable_refine
        self.learning_strategy = learning_strategy
        self.retrieve_strategy = retrieve_strategy
        self.records = records if records else []
        self.utility = utility if utility else []
        self.tokens = tokens if tokens else []
        self.valuation = valuation if valuation else []
        self.desire = 'public_goods'
        self.arms = ''
        self.learnings = ''
        self.guidance = ''
        self.guidance_history = []

        if self.retrieve_strategy != 'none':
            self.retrieve_tool = NodeRetrieve(search_model=self.retrieve_strategy,
                                              model_path=self.model_path,
                                              base_url=self.base_url,
                                              objective=DESIRE_DESC[self.desire])

    def request(self, round_id, inputs, request_key="option"):
        if self.model == "user":
            return self.user_request(inputs, request_key)

        elif self.model.startswith("specified"):
            return self.specified_request(round_id, request_key)

        else:
            return self.gpt_request(inputs)

    def user_request(self, outputs, request_key):
        output_str = '\n'.join([prompt["content"] for prompt in outputs])
        response = input(f"{output_str}\nPlease input the answer for {request_key}:")
        response = f'{{"{request_key}": "{response}"}}'
        return response

    def specified_request(self, round_id, request_key):
        options = self.model.split("=")[1].split('/')
        option_num = len(options)
        response = options[(round_id - 1) % option_num]
        response = f'{{"{request_key}": "{response}"}}'
        return response

    def gpt_request(self, inputs):
        start_time = time.time()
        while time.time() - start_time < 10:
            if self.model == 'text-davinci-003':
                response = completion(self.model, inputs).strip()
                self.print_prompt(inputs, response)
                return response
            elif self.model.startswith(('gpt-3.5-turbo', 'gpt-4')):
                response = chat(self.model, inputs).strip()
                self.print_prompt(self.id, inputs, response)
                return response
            elif self.model.startswith('gemini'):
                response = gemini_chat(self.model, inputs).strip()
                self.print_prompt(self.id, inputs, response)
                return response
            elif self.model.startswith('mistral') or self.model.startswith('mixtral'):
                msgs = []
                for i, msg in enumerate(inputs):
                    if msg['role'] == 'system':
                        msgs.extend([
                            {'role': 'user', 'content': msg['content']},
                        ])
                        if i + 1 < len(inputs) and inputs[i + 1]['role'] == 'user':
                            msgs.append({'role': 'assistant', 'content': 'Got it!'})
                    elif msg['role'] == 'user':
                        msgs.append(msg)
                        if i + 1 < len(inputs) and inputs[i + 1]['role'] == 'user':
                            msgs.append({'role': 'assistant', 'content': 'Got it!'})
                    else:
                        msgs.append(msg)
                if msgs[-1]['role'] == 'assistant':
                    msgs.pop()
                print(msgs)

                response = vllm_chat(
                    base_url=self.base_url,
                    msgs=msgs,
                )
                return response
            elif self.model.startswith('qwen'):
                response = vllm_chat(
                    base_url=self.base_url,
                    msgs=inputs,
                )
                return response
            elif self.model.startswith('llama'):
                response = vllm_chat(
                    base_url=self.base_url,
                    msgs=inputs,
                )
                return response

            else:
                raise ValueError("The model is not supported or does not exist.")

    def print_prompt(self, id, inputs, response):
        os.makedirs("records", exist_ok=True)
        with open(f"records/{id}.txt", 'a') as f:
            f.write(f"{inputs}\n----\n")
            f.write(f"{response}\n====\n")
        return

    def retrieve(self):
        dialog_history = copy.deepcopy(self.prompt)
        scene = f"{self.initial_prompt[0]['content']}\n\n"
        context = ""
        for h in dialog_history[1:]:
            context += '%s:  %s \n' % (
                h["role"].replace('user', 'Game Master').replace('assistant', f"{self.id}"), h["content"])

        instruct = INSTRUCT_DECISION_SCENE_TEMPLATE.format(
            player_name=self.id,
            history_dialogue=context,
        )
        self.retrieve_tool.expansion(instruct)
        if self.retrieve_tool.search_model == 'random':
            labels = self.retrieve_tool.random_search(num_paths=5)
        else:
            labels = self.retrieve_tool.search(scene=scene + instruct, beam_width=5)

        path = [f"{node.goal}: {node.detail}" for node in self.retrieve_tool.cur_nodes if node.chosen]

        guidance_list = []
        structured_text = f"\nHere are some sub-goals and guidance derived from your primary goal: \n"
        temp_str = ""
        for label, guidance in zip(labels, path):
            temp_str += f" - {guidance}\n"  # details must be less than 30 words
            guidance_list.append(f"{label}: {guidance}")

        structured_text += temp_str
        structured_text += "\nIn this round, You should try to improve your game strategy based on these " \
                           "guidance, in order to achieve your primary objective.\n "

        self.guidance = structured_text
        self.guidance_history.append(guidance_list)
        self.prompt[0]['content'] = self.initial_prompt[0]['content'] + '\n\n' + self.guidance
        wprint(self.guidance, color='magenta')

    def update_arms(self, learnings):
        length = len(self.arms)
        for idx, learning in enumerate(learnings):
            self.arms[length + idx] = learning.split('.')[1].strip()

    def learning(self, past_game_log):
        llm = ChatOpenAI(
            model=self.model_path,
            temperature=0,
            api_key="EMPTY",
            base_url=self.base_url,
        )

        if self.learning_strategy == 'reflexion':
            instruct_learn = INSTRUCT_REFLEXION_TEMPLATE.format(
                past_game_log=past_game_log)
            result = llm([HumanMessage(content=instruct_learn)])
            result = result.content
            self.learnings += 'Reflections:\n-' + result + '\n'

        elif self.learning_strategy == 'clin':
            instruct_learn = INSTRUCT_LEARNING_TEMPLATE.format(
                past_game_log=past_game_log,
                past_learnings=self.learnings)
            # prompt = {"role": "user", "content": instruct_learn}
            # result = self.gpt_request([prompt])
            result = llm([HumanMessage(content=instruct_learn)])
            result = result.content
            self.learnings += 'Learnings:\n-' + result + '\n'

        if self.learnings != '':
            self.prompt[0]['content'] = self.initial_prompt[0]['content'] + \
                                        f"\n\nHere are your key learning points and practical tips from " \
                                        f"previous game rounds. You can use them to guide this round:\n```\n" \
                                        f"{self.learnings}\n```"
        return self.learnings


class GameServer:
    def __init__(self, player_num,
                 round_id,
                 prompt_folder,
                 models,
                 model_path,
                 base_url,
                 version,
                 learning_strategy,
                 enable_refine,
                 retrieve_strategy):
        if models.startswith("gemini"):
            default_prompt = [
                {"role": "user", "parts": None}
            ]
        else:
            default_prompt = [
                {"role": "system", "content": ""}
            ]
        self.round_id = round_id
        self.player_num = player_num
        self.round_records = []
        self.prompt_folder = prompt_folder
        self.version = version
        self.learning_strategy = learning_strategy
        self.enable_refine = enable_refine
        self.retrieve_strategy = retrieve_strategy

        if isinstance(models, str):
            self.players = [Player(models,
                                   model_path,
                                   base_url,
                                   f"player_{i}",
                                   copy.deepcopy(default_prompt),
                                   self.learning_strategy,
                                   self.enable_refine,
                                   self.retrieve_strategy
                                   )
                            for i in range(player_num)
                            ]

        elif isinstance(models, list):
            self.players = [Player(models[i], f"player_{i}",
                                   copy.deepcopy(default_prompt),
                                   self.learning_strategy,
                                   self.enable_refine,
                                   self.retrieve_strategy,
                                   )
                            for i in range(player_num)
                            ]

    def cstm_color(self, x, min_x, max_x):
        # https://matplotlib.org/stable/gallery/color/colormap_reference.html
        # autumn(_r), viridis(_r), plasma, RdBu_r, Paired, coolwarm
        return plt.cm.plasma_r((np.clip(x, min_x, max_x) - min_x) / (max_x - min_x))

    def update_system_prompt(self, description_file, description_list):
        for player in self.players:
            description_prompt = get_prompt(description_file, description_list)
            if player.model.startswith("gemini"):
                for item in player.prompt:
                    if item.get("role") == "user":
                        item["parts"] = [description_prompt]
                        break
            else:
                for item in player.prompt:
                    if item.get("role") == "system":
                        item["content"] = description_prompt
                        break
            # player.prompt.append({'role': 'assistant',
            # 'content': "Yes, I'm ready to play the game!"})
            player.initial_prompt = copy.deepcopy(player.prompt)

    def save(self, savename, i, game_hash, game_info={}):
        save_data = {
            "meta": {
                "name_exp": self.name_exp,
                "player_num": self.player_num,
                **game_info,
                "round_id": self.round_id,
                "version": self.version,
            },
            "round_records": self.round_records,
            "player_data": [],
        }

        for player in self.players:
            if not player.model.startswith("gemini"):
                if self.round_id > 20:
                    player.prompt = player.prompt[:1] + player.prompt[2:]

            player_info = {
                "model": player.model,
                "id": player.id,
                "learning_strategy": player.learning_strategy,
                "retrieve_strategy": player.retrieve_strategy,
                "prompt": player.prompt,
                "records": player.records,
                "utility": player.utility,
                "learnings": player.learnings,
                "arms": assign_labels_to_last_layer(
                    player.retrieve_tool.root) if player.retrieve_strategy != "none" else {},
                "arms_cnt": player.retrieve_tool.cnt if player.retrieve_strategy != "none" else [],
            }
            save_data["player_data"].append(player_info)

            os.makedirs(f"save/{game_hash}", exist_ok=True)
            if player.retrieve_strategy != "none":
                with open(f"save/{game_hash}/{savename}-tree-{player.id}-{i}.json", 'wb') as f:
                    dill.dump(player.retrieve_tool.root, f)

        os.makedirs(f"save/{game_hash}", exist_ok=True)
        savepath = f'save/{game_hash}/{savename}-{i}.json'
        with open(savepath, 'w') as json_file:
            json.dump(save_data, json_file, indent=2)
        return savepath

    def load(self, round_records, players):
        self.round_records = round_records
        for index, loaded_player in enumerate(players):
            self.players[index] = Player(**loaded_player)
        return

    def get_dialog(self, player):
        dialog_history = ""
        dialog_history += f"{player.initial_prompt[0]['content']}\n\n"
        for h in player.prompt[1:]:
            dialog_history += '%s: %s \n' % (
                h["role"].replace('user', 'Game Master').replace('assistant', f"{player.id}"), h["content"])
        return dialog_history

    def run(self, repeat, rounds, description_file, description_list, game_hash):
        self.update_system_prompt(description_file, description_list)
        for round_count in range(self.round_id + 1, self.round_id + rounds + 1):
            for player in self.players:
                if player.retrieve_strategy != 'none':
                    player.retrieve()

            self.start(round_count)
            self.save(self.name_exp, repeat, game_hash)

            for player in self.players:
                if player.learning_strategy != 'none':
                    past_game_log = self.get_dialog(player)
                    player.learning(past_game_log)

            time.sleep(1)


def bidding_multithread(bidder_list: List[Player],
                        instruction_list,
                        round_id,
                        thread_num=10,
                        retry=1):
    '''
    auctioneer_msg: either a uniform message (str) or customed (list)
    '''

    result_queue = queue.Queue()
    threads = []
    semaphore = threading.Semaphore(thread_num)

    def run_once(i: int, player: Player, round_id: int, request_msg: str):
        try:
            result = player.request(round_id, request_msg)
            result_queue.put((True, i, result))
        finally:
            semaphore.release()

    if isinstance(instruction_list, str):
        instruction_list = [instruction_list] * len(bidder_list)

    for i, (bidder, msg) in enumerate(zip(bidder_list, instruction_list)):
        thread = threading.Thread(target=run_once, args=(i, bidder, round_id, msg))
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
        raise Exception(f"Error(s):\n" + '\n'.join([f'{i}: {e}' for i, e in errors]))

    valid_results = [x[1:] for x in results if x[0]]
    valid_results.sort()

    return [x for _, x in valid_results]
