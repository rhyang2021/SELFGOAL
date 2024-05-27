"""
Author: LAM Man Ho (mhlam@link.cuhk.edu.hk)
"""
from tqdm import tqdm
import matplotlib.pyplot as plt
from statistics import mean, stdev
import json
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
from prompt_base import PARSE_GOALS_SYSTEM_MESSAGE, PARSE_RESPONSE_SYSTEM_MESSAGE
from server import *
from utils import trace_back, extract_jsons_from_text


class GuessingGame(GameServer):
    def __init__(self, player_num,
                 min,
                 max,
                 ratio,
                 ratio_str,
                 version='v1',
                 name_exp='guessing_game',
                 round_id=0,
                 models='qwen-7b',
                 model_path='',
                 base_url='',
                 learning_strategy='reflexion',
                 retrieve_strategy='none',
                 enable_refine=False,
                 ):
        super().__init__(player_num,
                         round_id,
                         'guessing_game',
                         models,
                         model_path,
                         base_url,
                         version,
                         learning_strategy,
                         enable_refine,
                         retrieve_strategy
                         )
        self.min = min
        self.max = max
        self.ratio = ratio
        self.ratio_str = ratio_str
        self.name_exp = name_exp
        self.temp_score = 0

    def compute_result(self, responses):
        winner = min(responses, key=lambda x: abs(x - mean(responses) * self.ratio))
        record = {
            "responses": responses,
            "mean": mean(responses),
            "mean_ratio": mean(responses) * self.ratio,
            "winner": winner,
            "winner_num": responses.count(winner)
        }
        self.round_records.append(record)
        return record

    def report_result(self, round_record):
        result_list = round_record["responses"]
        random.shuffle(result_list)
        result_str = ', '.join(map(str, result_list))
        for player in self.players:
            player_choice = player.records[-1]
            won = player_choice == round_record["winner"]
            won_msg = "Congratulation you won" if won else "Unfortunately you lost"
            report_file = f'prompt_template/{self.prompt_folder}/report_{self.version}.txt'
            report_list = [self.round_id, round_record["mean"], self.ratio_str,
                           f'''{round_record["mean_ratio"]:.2f}''',
                           round_record["winner"], player_choice, won_msg]
            report_prompts = get_prompt(report_file, report_list)
            gemini_msg = []
            if player.model.startswith('gemini'):
                for i, msg in enumerate(report_prompts):
                    if i == 0:
                        player.prompt[-1]['parts'].append(msg)
                    elif i == 1:
                        player.prompt.append({'role': 'model', 'parts': [msg]})
                    else:
                        gemini_msg.append(msg)
                player.prompt.append({'role': 'user', 'parts': gemini_msg})
            else:
                report_prompts = [
                    {"role": f"{'assistant' if i == 1 else 'user'}", "content": msg}
                    for i, msg in enumerate(report_prompts)
                ]
                player.prompt = player.prompt + report_prompts
        return

    def graphical_analysis(self, i, game_hash, players_list):
        os.makedirs(f"figures/{game_hash}", exist_ok=True)
        round_numbers = [str(i) for i in range(1, self.round_id + 1)]
        player_color = [self.cstm_color(x, 1, self.player_num) for x in range(1, self.player_num + 1)]

        # Player
        for pid, player in enumerate(players_list):
            print(round_numbers)
            print(player.records)
            plt.plot(round_numbers, player.records, marker='.', color=player_color[pid], label=f"Player {pid + 1}")
        plt.legend(loc=1).set_zorder(1000)
        plt.title(f'Guessing Game')
        plt.xlabel('Round')
        plt.ylabel('Chosen Number')
        plt.xticks(ticks=range(1, 21, 2))
        plt.savefig(f'figures/{game_hash}/{self.name_exp}-{i}-player.svg', format="svg", dpi=300)
        plt.clf()

        # Average
        winning_list = [r["winner"] for r in self.round_records]
        plt.plot(round_numbers, winning_list, marker='.', label='Winner', color='r')
        responses_list = [r["responses"] for r in self.round_records]
        stdev_list = [stdev(r) for r in responses_list]
        mean_list = [mean(r) for r in responses_list]
        self.temp_score = mean(mean_list)

        plt.plot(round_numbers, mean_list, marker='.', label='Average', color='b')
        plt.fill_between(
            round_numbers,
            [y - s for y, s in zip(mean_list, stdev_list)],
            [y + s for y, s in zip(mean_list, stdev_list)],
            alpha=0.2, color='b'
        )
        plt.title(f'Guessing Game')
        plt.xlabel('Round')
        plt.ylabel('Chosen Number')
        plt.xticks(ticks=range(1, 21, 2))
        plt.legend(loc=1)
        plt.savefig(f"figures/{game_hash}/{self.name_exp}-{i}-average.svg", format="svg", dpi=300)
        plt.clf()

    def save(self, savename, i, game_hash):
        game_info = {
            "min": self.min,
            "max": self.max,
            "ratio": self.ratio,
            "ratio_str": self.ratio_str,
        }
        return super().save(savename, i, game_hash, game_info)

    def show(self, i, game_hash, attr_name=None, metric_list='ALL'):
        eligible_players = select_players(self.players, attr_name, metric_list)
        self.graphical_analysis(i, game_hash, eligible_players)

    def parse_response(self, text: str):

       llm = ChatOpenAI(model='gpt-3.5-turbo-1106', temperature=0.)
       result = llm.invoke([SystemMessage(content=PARSE_RESPONSE_SYSTEM_MESSAGE),
                            HumanMessage(
                            content=f"{text}\nDon't output anything else other than the JSON object.")])
       print(result.content)
       belief_json = json.loads(result.content)
       return belief_json


    def start(self, round):
        print(f"Round {round}: ")
        self.round_id = round

        request_file = f'prompt_template/{self.prompt_folder}/request_{self.version}.txt'

        cot_msg = get_cot_prompt(self.cot)
        if self.cot:
            output_format = f'{cot_msg}. Give your thinking process first, followed by your chosen number. Your chosen number must be an integer between {self.min} and {self.max}). \n\n Your thinking process and chosen number must be in the following JSON format: {{"explanation": "thinking_process", "chosen_number": "integer_between_{self.min}_and_{self.max}"}}'
        else:
            output_format = f'Please provide your chosen number in the following JSON format: {{"chosen_number": "integer_between_{self.min}_and_{self.max}"}}'

        request_list = [self.round_id, self.ratio_str, self.min, self.max, output_format]
        request_msg = get_prompt(request_file, request_list)
        request_prompt = [{"role": "user", "content": request_msg}]
        responses = []

        _player_list = []
        _instruct_list = []
        for player in tqdm(self.players):
            # player.prompt = player.prompt + request_prompt
            if player.model.startswith("gemini"):
                player.prompt[-1]['parts'].append(request_msg)
                _instruct_list.append(player.prompt)
                _player_list.append(player)
            else:
                _instruct_list.append(player.prompt + request_prompt)
                _player_list.append(player)

        _msgs = bidding_multithread(_player_list, _instruct_list, round_id=self.round_id, thread_num=1)
        for player, msgs in zip(_player_list, _msgs):
            parsered_responses = self.parse_response(msgs)
            parsered_responses = eval(str(parsered_responses["chosen_number"]))
            player.records.append(parsered_responses)
            responses.append(parsered_responses)
        round_record = self.compute_result(responses)
        self.report_result(round_record)

    def run(self, repeat, rounds, game_hash, cot=2, role=None):
        self.cot = cot
        # Update system prompt (number of round)
        description_file = f'prompt_template/{self.prompt_folder}/description_{self.version}.txt'
        role_msg = get_role_msg(role)
        description_list = [self.player_num, self.round_id + rounds, self.min, self.max, self.ratio_str, role_msg]
        super().run(repeat, rounds, description_file, description_list, game_hash)
        print(f"Score: {self.temp_score:.2f}")


def make_game_hash():
    return str(int(time.time()))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--model_name', '-m', type=str,
                        required=True)
    parser.add_argument('--model_path', '-p', type=str,
                        required=True)
    parser.add_argument('--base_url', '-u', type=str,
                        required=True)
    args = parser.parse_args()
    game_hash = make_game_hash()

    for i in tqdm(range(args.repeat), desc='Repeat'):
        print(f"Repeat {i}: ")
        game = GuessingGame(player_num=5, min=0, max=100,
                            models=args.model_name,
                            model_path=args.model_path,
                            base_url=args.base_url,
                            ratio=2 / 3, ratio_str='2/3',
                            version='v1',
                            learning_strategy='none',
                            enable_refine=False,
                            retrieve_strategy="none",
                            )
        game.run(repeat=i, rounds=20, game_hash=game_hash)


