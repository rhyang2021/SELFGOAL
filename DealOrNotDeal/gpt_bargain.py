import time
import threading
import queue
import copy

from tqdm import tqdm
from src.agent import load_initial_instructions, involve_moderator, BuyerAgent, SellerAgent, ModeratorAgent
from src.items_base import create_items
from utils import wprint, assign_labels_to_last_layer, trace_back


def check_profits(proposal):
    items = ['book', 'hat', 'ball']
    bob_value = [bob.book_value, bob.hat_value, bob.ball_value]
    alice_value = [alice_variants[0].book_value, alice_variants[0].hat_value, alice_variants[0].ball_value]

    profit = {}
    profit['alice'] = 0
    for item, value in zip(items, alice_value):
        profit['alice'] += proposal['alice_items'].get(item, 0) * value

    profit['bob'] = 0
    for item, value in zip(items, bob_value):
        profit['bob'] += proposal['bob_items'].get(item, 0) * value

    return profit


def check_proposal(data):
    msgs = ""
    if data == {}:
        return "no parsible JSON"

    # Check if all expected top-level keys are present
    expected_keys = ["alice_items", "bob_items"]
    for key in expected_keys:
        if key not in data:
            return f"Error: Missing '{key}' field in the JSON."

    # Check if "Alice" is a dictionary with numbers as values
    if not isinstance(data["alice_items"], dict):
        return "Error: 'alice_items' should be a dictionary of every items."

    if not isinstance(data["bob_items"], dict):
        return "Error: 'bob_items' should be a dictionary of every items."

    for item, cnt in data["alice_items"].items():
        if not isinstance(cnt, (int)):
            return f"Alice's number of {item} is not an integer."

    for item, cnt in data["bob_items"].items():
        if not isinstance(cnt, (int)):
            return f"Bob's number of {item} is not an integer."

    expected_items = ["book", "hat", "ball"]
    for item in data["alice_items"].keys():
        if item not in expected_items:
            return f"The {item} is not included in this negotiation. Players can only receive balls, hats, or books."

    for item in data["bob_items"].keys():
        if item not in expected_items:
            return f"The {item} is not included in this negotiation. Players can only receive balls, hats, or books."

    expected_values = [bob.book_cnt, bob.hat_cnt, bob.ball_cnt]
    for item, value in zip(expected_items, expected_values):
        if not data["alice_items"].get(item, 0) + data["bob_items"].get(item, 0) <= value:
            return f"The number of {item} in the plan exceeds the available quantity: There are only {value} {item} in total."

    # If everything is fine
    return msgs


def get_dialog(alice):
    dialog_history = ""
    dialog_history += f"{alice.initial_dialog_history[0]['content']}\n\n{alice.initial_dialog_history[1]['content']}\n\n"
    for h in alice.dialog_history[2:]:
        dialog_history += '%s:  %s \n' % (
            h["role"].replace('user', 'Bob').replace('assistant', 'Alice'), h["content"])
    return dialog_history


def run(bob, alice, moderator, parser, i, game_hash, n=10, fd=None, verbose=True):
    wprint('Alice: %s' % alice.last_response, fd, verbose=verbose)
    wprint('Bob: %s' % bob.last_response, fd, verbose=verbose)

    wprint('---- start negotiation ----', fd, verbose=verbose, color='green')
    bob_run = bob.last_response
    start_involve_moderator = False
    deal_at = "none"

    for game_round in range(n):

        if game_round > 0:
            if alice.learning_strategy != 'none':
                wprint('---- start alice learning ----', fd, verbose=verbose, color='green')
                past_auction_log = get_dialog(alice)
                alice.learning(past_auction_log)

        if alice.retrieve_strategy != 'none':
            wprint('---- start alice retrieval ----', fd, verbose=verbose, color='green')
            alice.retrieve(bob_run)
        if alice.enable_react:
            alice_run = alice.react(bob_run)
        elif alice.enable_adapt:
            alice_run = alice.adapt(bob_run)
        else:
            alice_run = alice.call(bob_run)

        wprint('Alice: %s' % alice.last_response, fd, verbose=verbose)

        if (start_involve_moderator is False and involve_moderator(bob_run, alice_run, game_round)):
            start_involve_moderator = True
            wprint('---- start moderating ----', fd, verbose=verbose, color='blue')

        if (start_involve_moderator):
            moderate = moderator.moderate(alice.dialog_history, who_was_last="buyer", retry=False)
            wprint('MODERATE have Alice and Bob achieved a deal? Yes or No: %s' % moderate, fd, verbose=verbose,
                   color='red')
            if ("yes" in moderate.lower()):
                deal_at = "seller"
                break
            else:
                pass

        bob_run = bob.call(alice_run)
        wprint('Bob: %s' % bob.last_response, fd, verbose=verbose)

        if (start_involve_moderator is False and involve_moderator(bob_run, alice_run, game_round)):
            start_involve_moderator = True
            wprint('---- start moderating ----', fd, verbose=verbose, color='blue')

        if (start_involve_moderator):
            moderate = moderator.moderate(bob.dialog_history, who_was_last="seller", retry=False)
            wprint('MODERATE have Alice and Bob achieved a deal? Yes or No: %s' % moderate, fd, verbose=verbose,
                   color='red')

            if ("yes" in moderate.lower()):
                deal_at = "buyer"
                break
            else:
                pass

    dialog_history = get_dialog(alice)
    if (deal_at != "none"):
        if (deal_at == "seller"):
            final_price = parser.parse_proposal(alice.dialog_history, who_was_last="buyer", retry=False)
        else:
            final_price = parser.parse_proposal(bob.dialog_history, who_was_last="seller", retry=False)
        msgs = check_proposal(final_price)
        if msgs != "":
            print(final_price)

            result = {"alice_model": alice.engine,
                      "name": alice.name,
                      "bob_model": bob.engine,
                      "alice_desire": alice.desire,
                      "bob_desire": bob.desire,
                      "learning_strategy": alice.learning_strategy,
                      "retrieve_strategy": alice.retrieve_strategy,
                      'enable_react': alice.enable_react,
                      'enable_adapt': alice.enable_adapt,
                      'enable_prune': alice.enable_prune,
                      'enable_tool': alice.enable_tool,
                      "final_proposal": final_price,
                      "profits": msgs,
                      "api_cost": alice.openai_cost + bob.openai_cost + moderator.openai_cost + parser.openai_cost,
                      'prune_list': [f"{node.goal}: {node.detail}" for node in
                                     alice.retrieve_tool.prune_list] if alice.enable_tool else [],
                      "guidance_history": alice.guidance_history,
                      'learnings': alice.learnings,
                      'arms': assign_labels_to_last_layer(alice.retrieve_tool.root) if alice.enable_tool else {},
                      'arms_cnt': alice.retrieve_tool.cnt if alice.enable_tool else [],
                      "dialog_history": dialog_history
                      }
        else:
            profits = check_profits(final_price)
            result = {"alice_model": alice.engine,
                      "name": alice.name,
                      "bob_model": bob.engine,
                      "alice_desire": alice.desire,
                      "bob_desire": bob.desire,
                      "learning_strategy": alice.learning_strategy,
                      "retrieve_strategy": alice.retrieve_strategy,
                      'enable_react': alice.enable_react,
                      'enable_adapt': alice.enable_adapt,
                      'enable_prune': alice.enable_prune,
                      'enable_tool': alice.enable_tool,
                      "final_proposal": final_price,
                      "profits": profits,
                      "api_cost": alice.openai_cost + bob.openai_cost + moderator.openai_cost + parser.openai_cost,
                      'prune_list': [f"{node.goal}: {node.detail}" for node in
                                     alice.retrieve_tool.prune_list] if alice.enable_prune else [],
                      "guidance_history": alice.guidance_history,
                      'learnings': alice.learnings,
                      'arms': assign_labels_to_last_layer(alice.retrieve_tool.root) if alice.enable_tool else {},
                      'arms_cnt': alice.retrieve_tool.cnt if alice.enable_tool else [],
                      "dialog_history": dialog_history
                      }
    else:
        result = {"alice_model": alice.engine,
                  "name": alice.name,
                  "bob_model": bob.engine,
                  "alice_desire": alice.desire,
                  "bob_desire": bob.desire,
                  "learning_strategy": alice.learning_strategy,
                  "retrieve_strategy": alice.retrieve_strategy,
                  'enable_react': alice.enable_react,
                  'enable_adapt': alice.enable_adapt,
                  'enable_prune': alice.enable_prune,
                  'enable_tool': alice.enable_tool,
                  "final_proposal": {},
                  "profits": {"alice": 0, "bob": 0},
                  "api_cost": alice.openai_cost + bob.openai_cost + moderator.openai_cost + parser.openai_cost,
                  'prune_list': [f"{node.goal}: {node.detail}" for node in
                                 alice.retrieve_tool.prune_list] if alice.enable_tool else [],
                  "guidance_history": alice.guidance_history,
                  'learnings': alice.learnings,
                  'arms': assign_labels_to_last_layer(alice.retrieve_tool.root) if alice.enable_tool else {},
                  'arms_cnt': alice.retrieve_tool.cnt if alice.enable_tool else [],
                  "dialog_history": dialog_history
                  }

    if alice.enable_tool:
        log_file = f"outputs/{game_hash}/tree-{i}"
        if not os.path.exists(log_file):
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, 'wb') as f:
            dill.dump(alice.retrieve_tool.root, f)

    return result


def run_with_feedback(bob, alice, moderator, parser, i, game_hash, n=10, fd=None, verbose=True):
    result = run(bob, alice, moderator, parser, i, game_hash, n)

    cnt = 0
    while cnt < 3:
        if isinstance(result['profits'], str):
            bob.feedback(result)
            alice.feedback(result)
            moderator.reset()
            parser.reset()
            result = run(bob, alice, moderator, parser, i, game_hash, n)
            cnt += 1
        else:
            break
    return result


def parallel_run_round(bob, alice_variants, moderator, parser, i, game_hash, n):
    result_queue = queue.Queue()
    threads = []

    def run_and_store(alice_variant, result_queue):
        result = run_with_feedback(copy.copy(bob), copy.copy(alice_variant),
                                   copy.copy(moderator), copy.copy(parser), i, game_hash, n)
        result_queue.put(result)

    for alice_variant in alice_variants:
        thread = threading.Thread(target=run_and_store, args=(alice_variant, result_queue))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    results = [result_queue.get() for _ in range(len(threads))]
    return results


def initialize_models(item,
                      engine: str = "gpt-3.5-turbo-1106",
                      model_path: str = "",
                      base_url: str = "",
                      ):
    count, value = item.info()
    moderator_initial_dialog_history = load_initial_instructions("lib_prompt/moderator_0509.txt")
    moderator = ModeratorAgent(initial_dialog_history=moderator_initial_dialog_history,
                               agent_type="moderator",
                               engine="gpt-4-1106-preview"
                               )
    parser_initial_dialog_history = load_initial_instructions("lib_prompt/moderator_parse.txt")
    parser = ModeratorAgent(initial_dialog_history=parser_initial_dialog_history,
                            agent_type="moderator",
                            engine="gpt-4-1106-preview",
                            trace_n_history=6
                            )

    alice_initial_dialog_history = load_initial_instructions('lib_prompt/seller.txt')

    alice_adapt = SellerAgent(initial_dialog_history=alice_initial_dialog_history,
                              name='0',
                              agent_type="seller",
                              engine=engine,
                              model_path=model_path,
                              base_url=base_url,
                              enable_react=True,
                              retrieve_strategy='none',
                              book_value=value["Alice"].get('books_value'), book_cnt=count.get('books_cnt'),
                              ball_value=value["Alice"].get('balls_value'), ball_cnt=count.get('balls_cnt'),
                              hat_value=value["Alice"].get('hats_value'), hat_cnt=count.get('hats_cnt'),
                              desire="fair",
                              )

    alice_search = SellerAgent(initial_dialog_history=alice_initial_dialog_history,
                               agent_type="seller",
                               name='1',
                               engine=engine,
                               model_path=model_path,
                               base_url=base_url,
                               enable_react=False,
                               enable_adapt=True,
                               enable_prune=False,
                               retrieve_strategy="none",
                               book_value=value["Alice"].get('books_value'), book_cnt=count.get('books_cnt'),
                               ball_value=value["Alice"].get('balls_value'), ball_cnt=count.get('balls_cnt'),
                               hat_value=value["Alice"].get('hats_value'), hat_cnt=count.get('hats_cnt'),
                               desire="fair",
                               )

    bob_initial_dialog_history = load_initial_instructions('lib_prompt/buyer.txt')
    bob = BuyerAgent(initial_dialog_history=bob_initial_dialog_history, agent_type="bob", engine="gpt-3.5-turbo-1106",
                     book_value=value["Bob"].get('books_value'), book_cnt=count.get('books_cnt'),
                     ball_value=value["Bob"].get('balls_value'), ball_cnt=count.get('balls_cnt'),
                     hat_value=value["Bob"].get('hats_value'), hat_cnt=count.get('hats_cnt'),
                     desire="fair",
                     )
    alice_variants = [alice_adapt, alice_search]

    return alice_variants, bob, moderator, parser


def make_game_hash():
    return str(int(time.time()))


if __name__ == '__main__':

    import argparse
    import os
    import dill
    import ujson as json

    verbose = True
    total_money_spent = 0
    parser = argparse.ArgumentParser()

    items = create_items('data/items_demo.jsonl')
    parser.add_argument('--model_path', '-p', type=str,
                        required=True)
    parser.add_argument('--base_url', '-u', type=str,
                        required=True)
    parser.add_argument('--engine', '-e', type=str,
                        required=True)

    game_hash = make_game_hash()
    args = parser.parse_args()

    for i in tqdm(range(len(items)), desc='items progress'):
        cur_item = items.pop(0)
        alice_variants, bob, moderator, parser_ = initialize_models(cur_item,
                                                                    engine=args.engine,
                                                                    model_path=args.model_path,
                                                                    base_url=args.base_url
                                                                    )

        print("==== ROUND %d ====" % i)
        cnt = 3
        while cnt > 0:
            try:
                results = parallel_run_round(bob, alice_variants, moderator, parser_, i, game_hash, n=10)
                for k, result in enumerate(results):
                    print(f"FINAL RESULT: {result}\n")
                    print("\n\n")
                    log_file = f"outputs/{game_hash}/alice{result['name']}-{i}.jsonl"
                    if not os.path.exists(log_file):
                        os.makedirs(os.path.dirname(log_file), exist_ok=True)
                    with open(log_file, 'a') as f:
                        f.write(json.dumps(result) + '\n')

                memo = {
                    'dialog_history': {f"{result['name']}": result["dialog_history"] for result in results},
                    'learnings': {f"{result['name']}": result["learnings"] for result in results},
                    'prune_list': {f"{result['name']}": result["prune_list"] for result in results},
                    'profit': {f"{result['name']}": result["profits"] for result in results},
                    'arms': {f"{result['name']}": result["arms"] for result in results},
                    'arms_cnt': {f"{result['name']}": result["arms_cnt"] for result in results}
                }

                with open(f"outputs/{game_hash}/memo-{i}.json", 'w') as f:
                    f.write(json.dumps(memo) + '\n')

                break

            except Exception as e:
                cnt -= 1
                print(f"Error in {i}th auction: {e}\n{trace_back(e)}")
                print(f"Retry {cnt} more times...")

        if cnt <= 0:
            raise ValueError
