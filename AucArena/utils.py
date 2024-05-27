import ujson as json
import re
import traceback


def trace_back(error_msg):
    exc = traceback.format_exc()
    msg = f'[Error]: {error_msg}.\n[Traceback]: {exc}'
    return msg


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


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i: i + n]


def reset_state_list(*states):
    empty = [None for _ in states[1:]]
    return [[]] + empty


def LoadJsonL(filename):
    if isinstance(filename, str):
        jsl = []
        with open(filename) as f:
            for line in f:
                if line.strip() != '':
                    jsl.append(json.loads(line))
        return jsl
    else:
        return filename


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

# Extract the partial order from a given text
# def extract_partial_order(text, feasible_items):
#     match = re.search(r'([A-Za-z\s]+[\s]*[><=][\s]*)+[A-Za-z\s]+(?=[\n\.\,])', text + '\n')
#     # or use r'([A-Za-z\s]+[\s]*[><=][\s]*)+[A-Za-z\s]+'
#     if match:
#         order = match.group().strip()
#     else:
#         order = ''

#     items_order = _parse_plan(order, feasible_items)
#     return items_order


# def _parse_plan(plan, feasible_items):
#     """
#     Parse the plan to extract the partial order.
#     Refine the parsed items using a list of feasible items with enhanced logic for first and last items.
#     """
#     # Splitting by '>' to get groups
#     groups = plan.split('>')
#     parsed_items = []
#     for group in groups:
#         # Splitting by '=' to get individual items
#         items = group.strip().split('=')
#         parsed_items.append([item.strip() for item in items])

#     refined_items = []

#     for group in parsed_items:
#         refined_group = []
#         for item in group:
#             matched_feasible_item = None
#             for feasible_item in feasible_items:
#                 if item.startswith(feasible_item):
#                     matched_feasible_item = feasible_item
#                     break
#                 elif item.endswith(feasible_item):
#                     matched_feasible_item = feasible_item
#                     break

#             if matched_feasible_item:
#                 refined_group.append(matched_feasible_item)

#         if refined_group:  # Only add groups with feasible items
#             refined_items.append(refined_group)

#     return refined_items


# def restore_plan_order(refined_items):
#     """
#     Restore the parsed plan order from the refined items.
#     """
#     # Join the groups using '='
#     groups = [' = '.join(group) for group in refined_items]
#     # Join the entire list using '>'
#     restored_order = ' > '.join(groups)
#     return restored_order


# # Check the consistency of a new plan against an old plan
# def _is_consistent_plans(old_items_order, new_items_order):
#     # Filtering the old plan to remove items not in the new plan
#     filtered_old_items_order = []
#     for group in old_items_order:
#         new_group = [item for item in group if any(item in new_group for new_group in new_items_order)]
#         if new_group:
#             filtered_old_items_order.append(new_group)

#     # Check if the order is consistent
#     if len(filtered_old_items_order) != len(new_items_order):
#         return False

#     for old_group, new_group in zip(filtered_old_items_order, new_items_order):
#         if len(old_group) != len(new_group) or any(o_item != n_item for o_item, n_item in zip(old_group, new_group)):
#             return False

#     return True


# def parse_price_from_text(text: str):
#     s = re.findall(r"-?\$\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?(?=\D|$)|\$-?\d+|-?\$\d+", text)
#     return [float(x.replace(',', '').replace('$', '')) for x in s]
