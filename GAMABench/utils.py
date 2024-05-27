import sys
import time
import ujson as json
import re
import traceback


def trace_back(error_msg):
    exc = traceback.format_exc()
    msg = f'[Error]: {error_msg}.\n[Traceback]: {exc}'
    return msg


def wprint(s, fd=None, verbose=True, color=None):
    # ANSI颜色代码
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m"
    }
    END_COLOR = "\033[0m"

    # 如果指定了颜色，则添加颜色代码
    if color in colors:
        s_colored = colors[color] + s + END_COLOR
    else:
        s_colored = s

    # 写入文件（如果fd不是None）
    if fd is not None:
        fd.write(s + '\n')

    # 控制台打印
    if verbose:
        print(s_colored)

    return


class Logger(object):
    def __init__(self, log_file, verbose=True):
        self.terminal = sys.stdout
        self.log = open(log_file, "w")
        self.verbose = verbose

        self.write("All outputs written to %s" % log_file)
        return

    def write(self, message):
        self.log.write(message + '\n')
        if (self.verbose): self.terminal.write(message + '\n')

    def flush(self):
        pass


# def reverse_identity(agent_type):
#     assert agent_type in ["buyer", "seller", "moderator", "critic"]
#     if(agent_type == "buyer"): return "seller"
#     elif(agent_type == "seller"): return "buyer"
#     else: return agent_type


def check_price_range(price, p_min=8, p_max=20):
    """check if one price is in legal range
    """
    if (price > p_min and price < p_max):
        return True
    else:
        return False


def check_k_price_range(prices, p_min=8, p_max=20):
    """check if all prices are in legal range
    """
    all_in_range = True
    for p in prices:
        if (not check_price_range(p, p_min, p_max)):
            all_in_range = False
            break
    return all_in_range


def parse_outputs(filepath, price_per_case=4):
    prices = []
    lines = open(filepath).readlines()
    case_price = []
    for l in lines:
        if (l.startswith("==== CASE")):
            if (len(case_price) > 0):
                assert (len(case_price) == price_per_case)
                prices.append(case_price)
            case_price = []
        elif (l.startswith("PRICE: ")):
            price = float(l.split('PRICE: ')[1].strip())
            case_price.append(price)

    if (len(case_price) > 0):
        assert (len(case_price) == price_per_case)
        prices.append(case_price)
    return prices


def parse_outputs_v2(filepath, price_per_case=4):
    prices = []
    lines = open(filepath).readlines()
    case_price = []
    for l in lines:
        if (l.startswith("==== ver")):
            if (len(case_price) > 0):
                # assert(len(case_price) == price_per_case)
                prices.append(case_price)
            case_price = []
        elif (l.startswith("PRICE: ")):
            price = float(l.split('PRICE: ')[1].strip())
            case_price.append(price)

    if (len(case_price) > 0):
        # assert(len(case_price) == price_per_case)
        prices.append(case_price)
    return prices


def extract_jsons_from_text(text):
    json_dicts = []
    stack = []
    start_index = None

    for i, char in enumerate(text):
        if char == '{':
            stack.append(char)
            if start_index is None:
                start_index = i
        elif char == '}':
            if stack:
                stack.pop()
            if not stack and start_index is not None:
                json_candidate = text[start_index:i + 1]
                try:
                    parsed_json = json.loads(json_candidate)
                    json_dicts.append(parsed_json)
                    start_index = None
                except json.JSONDecodeError:
                    pass
                finally:
                    start_index = None

    if len(json_dicts) == 0: json_dicts = [{}]
    return json_dicts


def LoadJsonL(filename):
    if isinstance(filename, str):
        jsl = []
        with open(filename) as f:
            for line in f:
                jsl.append(json.loads(line))
        return jsl
    else:
        return filename


def compute_time(start_time):
    return (time.time() - start_time) / 60.0


def assign_labels_to_last_layer(root):
    last_layer_info = {}

    def traverse(node):
        if not node.children:
            node.label = assign_labels_to_last_layer.counter
            goal_detail_combination = f"{node.goal}: {node.detail}"
            last_layer_info[node.label] = goal_detail_combination
            assign_labels_to_last_layer.counter += 1
        else:
            for child in node.children:
                traverse(child)

    assign_labels_to_last_layer.counter = 0
    traverse(root)
    return last_layer_info


def extract_numbered_list(paragraph):
    # Updated regular expression to match numbered list
    # It looks for:
    # - start of line
    # - one or more digits
    # - a period or parenthesis
    # - optional whitespace
    # - any character (captured in a group) until the end of line or a new number
    pattern = r"^\s*(\d+[.)]\s?.*?)(?=\s*\d+[.)]|$)"

    matches = re.findall(pattern, paragraph, re.DOTALL | re.MULTILINE)
    return [match.strip() for match in matches]
