from openai import OpenAI
import itertools
import numpy as np
import ujson as json
import random
from sklearn.metrics.pairwise import cosine_similarity
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.callbacks import get_openai_callback
from langchain.chat_models import (
    ChatOpenAI,
)

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

from src.prompt_base import (INSTRUCT_RANKING_TEMPLATE,
                             PARSE_ID_SYSTEM_MESSAGE,
                             INSTRUCT_INIT_GOAL_TEMPLATE,
                             INSTRUCT_GOAL_TEMPLATE,
                             PARSE_GOALS_SYSTEM_MESSAGE
                             )
from utils import assign_labels_to_last_layer


class Node:

    def __init__(self, id: str, goal: str, detail: str,
                 depth: int, msgs: list):
        self.id = id
        self.label = None
        self.goal = goal
        self.detail = detail
        self.depth = depth
        self.msgs = msgs
        self.chosen = False
        self.children = []
        self.chosen_cnt = []

    def __iter__(self):
        yield self.id, self.label, self.goal, self.detail
        for child in self.children:
            yield from child


class NodeRetrieve:

    def __init__(self,
                 search_model='random',
                 model_path='',
                 base_url='',
                 max_depth=None,
                 enable_prune=False
                 ):

        self.updates = []
        self.search_model = search_model
        self.counter = itertools.count()
        self.openai_cost = 0
        self.llm_token_count = 0
        self.root = Node(id='root',
                         goal='maximize profit: secure the highest profit at the end of this auction, compared to all '
                              'other bidders',
                         detail='',
                         depth=0,
                         msgs=[],
                         )
        self.root.chosen = True
        self.root.chosen_cnt.append(self.root.chosen)
        self.cur_nodes = [self.root]
        self.embeddings = [self.get_embedding(f"{node.goal}: {node.detail}") for node in self.cur_nodes]
        self.max_depth = max_depth
        self.cnt = [1]
        self.enable_prune = enable_prune
        self.prune_list = []
        self.base_url = base_url
        self.model_path = model_path

        if 'gpt-' in self.search_model:
            self.llm = ChatOpenAI(model=self.search_model,
                                  temperature=0.,
                                  max_retries=30,
                                  request_timeout=1200
                                  )
        elif 'random' in self.search_model or 'similarity' in self.search_model:
            self.llm = ChatOpenAI(model='gpt-3.5-turbo-1106',
                                  temperature=0.,
                                  max_retries=30,
                                  request_timeout=1200
                                  )
        elif 'gemini' in self.search_model:
            self.llm = ChatGoogleGenerativeAI(model=self.search_model, temperature=0.,
                                              max_output_tokens=2048,
                                              )
        elif 'mistral-' in search_model or 'mixtral-' in search_model:
            self.llm = ChatOpenAI(model=self.model_path,
                                  temperature=0,
                                  api_key="EMPTY",
                                  base_url=self.base_url,
                                  )

        elif 'qwen-' in search_model:

            self.llm = ChatOpenAI(
                model=self.model_path,
                temperature=0,
                api_key="EMPTY",
                base_url=self.base_url,
            )

    @retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(10))
    def chat_llm(self, messages):
        result = self.llm(messages)
        return result

    def get_instruct(self, scene, depth, goal):
        if depth == 0:
            instruct = INSTRUCT_INIT_GOAL_TEMPLATE.format(
                scene=scene,
                goal=goal,
            )
        else:
            instruct = INSTRUCT_GOAL_TEMPLATE.format(
                scene=scene,
                sub_goal=goal,
            )
        return instruct

    def parse_goals(self, text: str):

        with get_openai_callback() as cb:
            llm = ChatOpenAI(model='gpt-3.5-turbo-1106', temperature=0.)
            result = llm.invoke([SystemMessage(content=PARSE_GOALS_SYSTEM_MESSAGE),
                                 HumanMessage(content=f"{text}\nDon't output anything else other than the JSON object.")])
            belief_json = json.loads(result.content)
            self.openai_cost += cb.total_cost

        return belief_json

    def get_embedding(self, text, model="text-embedding-3-small"):
        text = text.replace("\n", " ")
        client = OpenAI()
        return client.embeddings.create(input=[text], model=model).data[0].embedding

    def check_sim(self, sentence1, cur_embedding):
        embedding1 = np.array(self.get_embedding(sentence1)).reshape(1, -1)
        for embedding in self.embeddings + [cur_embedding]:
            embedding2 = np.array(embedding).reshape(1, -1)
            similarity = cosine_similarity(embedding1, embedding2)
            if similarity[0, 0] >= self.max_depth:
                return False
        return True

    def check_converge(self):
        if len(self.cnt) > 3 and abs(np.mean(self.cnt[-3:]) - self.cnt[-1]) <= 2:
            return True
        else:
            return False

    def expansion(self, scene, max_children=10):
        if not self.check_converge():
            for _ in range(self.cnt[-1]):
                cur_embedding = self.embeddings.pop(0)
                cur_node = self.cur_nodes.pop(0)
                cur_node.chosen_cnt.append(cur_node.chosen)

                if not cur_node.chosen:
                    self.cur_nodes.append(cur_node)
                    self.embeddings.append(cur_embedding)
                else:
                    instruct = self.get_instruct(scene, cur_node.depth,
                                                 f"{cur_node.goal}: {cur_node.detail}")
                    msgs = cur_node.msgs[:2]
                    cur_node.msgs += [HumanMessage(content=instruct)]
                    msgs += [HumanMessage(content=instruct)]
                    with get_openai_callback() as cb:
                        result = self.chat_llm(msgs)
                        self.openai_cost += cb.total_cost
                    cur_node.msgs += [AIMessage(content=result.content)]
                    goal_dict = self.parse_goals(result.content)  # Generate the text when creating the node

                    cnt = 0
                    for i, (key, value) in enumerate(goal_dict.items()):
                        if i <= max_children:
                            print(f"{cur_node.id}-{i}: {key}")
                            children = Node(id=f"{cur_node.id}-{i}",
                                            goal=key,
                                            detail=value,
                                            msgs=cur_node.msgs.copy(),
                                            depth=cur_node.depth + 1,
                                            )
                            if self.check_sim(f"{children.goal}: {children.detail}", cur_embedding):
                                cnt += 1
                                cur_node.children.append(children)
                                self.cur_nodes.append(children)
                                self.embeddings.append(self.get_embedding(f"{children.goal}: {children.detail}"))
                    if cnt == 0:
                        self.cur_nodes.append(cur_node)
                        self.embeddings.append(cur_embedding)

            self.cnt.append(len(self.cur_nodes))
        else:
            if self.enable_prune:
                new_cur_nodes = []
                for node in self.cur_nodes:
                    if len(node.chosen_cnt) > 5 and sum(node.chosen_cnt[-5:]) == 0:
                        self.prune_list.append(node)
                    else:
                        new_cur_nodes.append(node)
                self.cur_nodes = new_cur_nodes

    def parse_ids(self, text: str):
        with get_openai_callback() as cb:
            llm = ChatOpenAI(model='gpt-3.5-turbo-1106', temperature=0.)

            attempt_count = 0
            max_attempts = 2
            while attempt_count < max_attempts:
                result = llm.invoke([SystemMessage(content=PARSE_ID_SYSTEM_MESSAGE),
                                     HumanMessage(
                                         content=f"{text}\nDon't output anything else other than the JSON object.")])
                self.openai_cost += cb.total_cost

                try:
                    belief_json = json.loads(str(result.content))
                    return belief_json["IDs"]
                except KeyError:
                    attempt_count += 1
                    print("Error: 'IDs' not found in JSON. Retrying...")

            raise KeyError("Failed to retrieve 'IDs' from response after multiple attempts.")

    def create_guidance_from_path(self):
        """
        Generate the formatted guidance for a given path with its ID.
        """
        guidance = ""
        for id, node in enumerate(self.cur_nodes):
            guidance += f"ID {id + 1}: {node.goal}: {node.detail} \n"
        return guidance

    def search(self, scene, beam_width):
        # Create instructions for each path
        guidance_text = self.create_guidance_from_path()
        instruct = INSTRUCT_RANKING_TEMPLATE.format(
            scene=scene,
            beam_width=beam_width,
            guidance=guidance_text,
            objective='maximize your profit: secure the highest profit at the end of this auction, compared to '
                      'all other bidders',
        )
        with get_openai_callback() as cb:
            messages = [HumanMessage(content=instruct)]
            result = self.chat_llm(messages)
            self.openai_cost += cb.total_cost

        ids = self.parse_ids(result.content)
        ids = [int(idx) - 1 for idx in ids if isinstance(idx, (int, float))]
        top_paths = [self.cur_nodes[idx] for idx in ids[:beam_width] if idx <= len(self.cur_nodes) - 1]
        for id, path in zip(ids, top_paths):
            self.cur_nodes[id].chosen = True
            # print(id, f"{path.goal}")

        return ids

    def embed_search(self, scene, beam_width):
        scene_embedding = np.array(self.get_embedding(scene)).reshape(1, -1)
        similarities = []
        for id, node in enumerate(self.cur_nodes):
            guidance = f"{node.goal}: {node.detail}"
            guidance_embedding = np.array(self.get_embedding(guidance)).reshape(1, -1)
            similarity = cosine_similarity(guidance_embedding, scene_embedding)
            similarities.append(similarity[0][0])

        sorted_indices = np.argsort(np.array(similarities))
        ids = sorted_indices[-beam_width:][::-1]
        top_paths = [self.cur_nodes[idx] for idx in ids[:beam_width] if idx <= len(self.cur_nodes) - 1]

        for id, path in zip(ids, top_paths):
            self.cur_nodes[id].chosen = True
            # print(id, f"{path.goal}")

        return ids

    def random_search(self, num_paths: int):
        ids = [k for k, _ in enumerate(self.cur_nodes)]
        random_ids = random.sample(ids, num_paths)
        for id in random_ids:
            self.cur_nodes[id].chosen = True
        return random_ids


if __name__ == '__main__':
    import dill
    import time

    with open('objective-tree-4', 'rb') as f:
        objective_tree = dill.load(f)  # this is your obj, loaded from file
    arms = assign_labels_to_last_layer(objective_tree)

    tool_retrieve = NodeRetrieve('gpt-4-1106-preview')
    time_start = time.time()
    result = tool_retrieve.search(
        'I am planning for bidding in the first round of this bidding war. I found myself have budge of $15000, '
        'but others might have high profit than me. There are 7 items, the value of them are 2000, 2000, 5000, 2000, '
        '2000, 2000, 5000',
        5)
    time_end = time.time()
    print(time_end - time_start)
