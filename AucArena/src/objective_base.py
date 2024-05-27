import ujson as json
import openai
from langchain_community.callbacks import get_openai_callback
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
import time
from prompt_base import (
    INSTRUCT_INIT_GOAL_TEMPLATE,
    INSTRUCT_GOAL_TEMPLATE,
    PARSE_GOALS_SYSTEM_MESSAGE,
)


class Node:

    def __init__(self, id: str, goal: str, detail: str, history_message: list,
                 depth: int, max_depth: int = 3, max_childern: int = 10):
        self.id = id
        self.label = None
        self.goal = goal
        self.detail = detail
        self.depth = depth
        self.openai_cost = 0
        self.msgs = []

        self.children = []
        if depth < max_depth:

            instruct = self.get_instruct()
            self.msgs += history_message
            self.msgs += [HumanMessage(content=instruct)]

            llm = ChatOpenAI(
                model="/data1/jgc/models/Mixtral-8x7B-Instruct-v0.1",
                temperature=0,
                api_key="EMPTY",
                base_url="http://10.176.40.140:8034/v1"
            )
            with get_openai_callback() as cb:
                for i in range(6):
                    try:
                        result = llm.invoke(self.msgs)
                        break
                    except Exception as e:
                        print(f'ERROR: {str(e)}')
                        print(f'Retrying for ({i + 1}/6), wait for {2 ** (i + 1)} sec...')
                        time.sleep(2 ** (i + 1))

                # llm = ChatGoogleGenerativeAI(model='gemini-pro', temperature=0.,
                # max_output_tokens=2048,
                # model_kwargs={"stop": ['\n']})

                # llm = ChatOpenAI(model="/data2/huangwenhao/hf_model/Mistral-7B-Instruct-v0.2",
                # temperature=0,
                # api_key="EMPTY",
                # base_url="http://10.176.40.140:8090/v1",
                # model_kwargs={"stop": ['<|im_end|>', '\n', ":", '(']})

            print(result.content)
            self.msgs += [AIMessage(content=result.content), ]

            goal_dict = self.parse_goals(result.content)  # Generate the text when creating the node

            for i, (key, value) in enumerate(goal_dict.items()):
                if i <= max_childern:
                    print(f"{id}-{i}: {key}")
                    self.children.append(Node(id=f"{id}-{i}",
                                              goal=key,
                                              detail=value,
                                              history_message=self.msgs.copy(),
                                              depth=depth + 1))

    @property
    def is_leaf(self):
        return len(self.children) == 0

    def get_instruct(self):
        if self.depth == 0:
            instruct = INSTRUCT_INIT_GOAL_TEMPLATE.format(
                goal=self.goal,
            )
        else:
            instruct = INSTRUCT_GOAL_TEMPLATE.format(
                sub_goal=self.goal,
            )
        return instruct

    def parse_goals(self, text: str):

        with get_openai_callback() as cb:
            llm = ChatOpenAI(model='gpt-3.5-turbo-1106', temperature=0.)
            result = llm.invoke([SystemMessage(content=PARSE_GOALS_SYSTEM_MESSAGE),
                                 HumanMessage(
                                     content=f"{text}\nDon't output anything else other than the JSON object.")])
            belief_json = json.loads(result.content)
            self.openai_cost += cb.total_cost

        return belief_json

    def __iter__(self):
        yield self.id, self.label, self.goal, self.detail
        for child in self.children:
            yield from child


def assign_labels_to_last_layer(root, max_depth):
    last_layer_info = {}

    def traverse(node, current_depth=0):
        if current_depth == max_depth - 1:
            for i, child in enumerate(node.children):
                child.label = assign_labels_to_last_layer.counter
                goal_detail_combination = f"{child.goal}: {child.detail}"
                last_layer_info[child.label] = goal_detail_combination
                assign_labels_to_last_layer.counter += 1
        else:
            for child in node.children:
                traverse(child, current_depth + 1)

    assign_labels_to_last_layer.counter = 0
    traverse(root)
    return last_layer_info


if __name__ == '__main__':

    import dill

    root = Node(id='root',
                goal='maximize profit: secure the highest profit at the end of this auction, compared to all other bidders',
                detail='', history_message=[], depth=0)

    root_iterator = iter(root)
    last_layer_info = assign_labels_to_last_layer(root, max_depth=3)

    for node_data in root_iterator:
        print(node_data)

    with open('../objective-tree-mistral-8x7b-v0.1', 'wb') as f:
        dill.dump(root, f)
