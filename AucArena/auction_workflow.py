import os
import time
import gradio as gr
import ujson as json
import traceback
from typing import List
from tqdm import tqdm
from src.auctioneer_base import Auctioneer
from src.bidder_base import Bidder, bidders_to_chatbots, bidding_multithread
from utils import trace_back, assign_labels_to_last_layer

LOG_DIR = 'logs'
enable_gr = gr.update(interactive=True)
disable_gr = gr.update(interactive=False)


def monitor_all(bidder_list: List[Bidder]):
    return sum([bidder.to_monitors() for bidder in bidder_list], [])


def parse_bid_price(auctioneer: Auctioneer, bidder: Bidder, msg: str):
    # rebid if the message is not parsible into a bid price
    bid_price = auctioneer.parse_bid(msg)
    while bid_price is None:
        re_msg = bidder.bid(
            "You must be clear about your bidding decision, say either \"I'm out!\" or \"I bid $xxx!\". Please rebid.")
        bid_price = auctioneer.parse_bid(re_msg)
        print(f"{bidder.name} rebid: {re_msg}")
    return bid_price


def enable_human_box(bidder_list):
    signals = []
    for bidder in bidder_list:
        if 'human' in bidder.model_name and not bidder.withdraw:
            signals.append(gr.update(interactive=True, visible=True,
                                     placeholder="Please bid! Enter \"I'm out\" or \"I bid $xxx\"."))
        else:
            signals.append(disable_gr)
    return signals


def disable_all_box(bidder_list):
    signals = []
    for bidder in bidder_list:
        if 'human' in bidder.model_name:
            signals.append(gr.update(interactive=False, visible=True,
                                     placeholder="Wait a moment to engage in the auction."))
        else:
            signals.append(gr.update(interactive=False, visible=False))
    return signals


def run_auction(
        auction_hash: str,
        auctioneer: Auctioneer,
        bidder_list: List[Bidder],
        thread_num: int,
        yield_for_demo=True,
        log_dir=LOG_DIR,
        repeat_num=0,
        memo_file=None):
    # bidder_list[0].verbose=True

    if yield_for_demo:
        chatbot_list = bidders_to_chatbots(bidder_list)
        yield [bidder_list] + chatbot_list + monitor_all(bidder_list) + [auctioneer.log()] + [disable_gr,
                                                                                              disable_gr] + disable_all_box(
            bidder_list)

    # ***************** Plan Round *****************
    # init bidder profit
    bidder_profit_info = auctioneer.gather_all_status(bidder_list)
    for bidder in bidder_list:
        bidder.set_all_bidders_status(bidder_profit_info)

    plan_instructs = [bidder.get_plan_instruct(auctioneer.items) for bidder in bidder_list]

    bidding_multithread(bidder_list, plan_instructs, func_type='plan', thread_num=thread_num)

    if yield_for_demo:
        chatbot_list = bidders_to_chatbots(bidder_list)
        yield [bidder_list] + chatbot_list + monitor_all(bidder_list) + [auctioneer.log()] + [disable_gr,
                                                                                              disable_gr] + disable_all_box(
            bidder_list)

    bar = tqdm(total=len(auctioneer.items_queue), desc='Auction Progress')
    while not auctioneer.end_auction():
        cur_item = auctioneer.present_item()

        bid_round = 0
        while True:
            # ***************** Bid Round *****************
            auctioneer_msg = auctioneer.ask_for_bid(bid_round)
            _bidder_list = []
            _bid_instruct_list = []

            # remove highest bidder and withdrawn bidders
            for bidder in bidder_list:
                if bidder is auctioneer.highest_bidder or bidder.withdraw:
                    bidder.need_input = False
                    continue
                else:
                    bidder.need_input = True  # enable input from demo
                    instruct = bidder.get_bid_instruct(auctioneer_msg, bid_round)
                    _bidder_list.append(bidder)
                    _bid_instruct_list.append(instruct)

            if yield_for_demo:
                chatbot_list = bidders_to_chatbots(bidder_list)
                yield [bidder_list] + chatbot_list + monitor_all(bidder_list) + [auctioneer.log()] + [disable_gr,
                                                                                                      disable_gr] + enable_human_box(
                    bidder_list)

            _msgs = bidding_multithread(_bidder_list, _bid_instruct_list, func_type='bid', thread_num=thread_num)

            record = []
            for i, (msg, bidder) in enumerate(zip(_msgs, _bidder_list)):
                if bidder.model_name == 'rule':
                    bid_price = bidder.bid_rule(auctioneer.prev_round_max_bid, auctioneer.min_markup_pct)
                else:
                    bid_price = parse_bid_price(auctioneer, bidder, msg)

                # can't bid more than budget or less than previous highest bid
                while True:
                    fail_msg = bidder.bid_sanity_check(bid_price, auctioneer.prev_round_max_bid,
                                                       auctioneer.min_markup_pct)
                    if fail_msg is None:
                        break
                    else:
                        bidder.need_input = True  # enable input from demo
                        auctioneer_msg = auctioneer.ask_for_rebid(fail_msg=fail_msg, bid_price=bid_price)
                        rebid_instruct = bidder.get_rebid_instruct(auctioneer_msg)

                        if yield_for_demo:
                            chatbot_list = bidders_to_chatbots(bidder_list)
                            yield [bidder_list] + chatbot_list + monitor_all(bidder_list) + [auctioneer.log()] + [
                                disable_gr, disable_gr] + disable_all_box(bidder_list)

                        msg = bidder.rebid_for_failure(rebid_instruct)
                        bid_price = parse_bid_price(auctioneer, bidder, msg)

                    if yield_for_demo:
                        chatbot_list = bidders_to_chatbots(bidder_list)
                        yield [bidder_list] + chatbot_list + monitor_all(bidder_list) + [auctioneer.log()] + [
                            disable_gr, disable_gr] + disable_all_box(bidder_list)

                bidder.set_withdraw(bid_price)
                record.append({'bidder': bidder, 'bid': bid_price, 'raw_msg': msg})

            auctioneer.record_bid(record, bid_round)

            if yield_for_demo:
                chatbot_list = bidders_to_chatbots(bidder_list)
                yield [bidder_list] + chatbot_list + monitor_all(bidder_list) + [auctioneer.log()] + [disable_gr,
                                                                                                      disable_gr] + disable_all_box(
                    bidder_list)

            is_sold = auctioneer.check_hammer(bid_round)
            bid_round += 1
            if is_sold:
                break
            else:
                if auctioneer.fail_to_sell and auctioneer.enable_discount:
                    for bidder in bidder_list:
                        bidder.set_withdraw(0)  # back in the game

        # ***************** Summarize ***************** 
        summarize_instruct_list = []
        for bidder in bidder_list:
            if bidder is auctioneer.highest_bidder:
                win_lose_msg = bidder.win_bid(cur_item, auctioneer.highest_bid)
            else:
                win_lose_msg = bidder.lose_bid(cur_item)
            msg = bidder.get_summarize_instruct(
                bidding_history=auctioneer.all_bidding_history_to_string(),
                hammer_msg=auctioneer.get_hammer_msg(),
                win_lose_msg=win_lose_msg
            )
            summarize_instruct_list.append(msg)

        # record profit information of all bidders for each bidder
        # (not used in the auction, just for belief tracking evaluation)
        bidder_profit_info = auctioneer.gather_all_status(bidder_list)
        for bidder in bidder_list:
            bidder.set_all_bidders_status(bidder_profit_info)

        bidding_multithread(bidder_list, summarize_instruct_list, func_type='summarize', thread_num=thread_num)

        if yield_for_demo:
            chatbot_list = bidders_to_chatbots(bidder_list)
            yield [bidder_list] + chatbot_list + monitor_all(bidder_list) + [auctioneer.log()] + [disable_gr,
                                                                                                  disable_gr] + disable_all_box(
                bidder_list)

        # ***************** Replan *****************
        if len(auctioneer.items_queue) > 0:  # no need to replan if all items are sold
            # ***************** Learn Round ****************
            for bidder in bidder_list:
                if bidder.learning_strategy != "none" or bidder.enable_refine:
                    # if no prev memo file, then no need to learn.
                    bidder.learn_from_prev_auction(auctioneer.log(show_model_name=False))

            replan_instruct_list = [bidder.get_replan_instruct(
                past_auction_log=auctioneer.log(show_model_name=False,),
                # last_log=auctioneer.last_log(show_model_name=False,),
                # bidding_history=auctioneer.all_bidding_history_to_string(), 
                # hammer_msg=auctioneer.get_hammer_msg()
            ) for bidder in bidder_list]
            bidding_multithread(bidder_list, replan_instruct_list, func_type='replan', thread_num=thread_num)

            if yield_for_demo:
                chatbot_list = bidders_to_chatbots(bidder_list)
                yield [bidder_list] + chatbot_list + monitor_all(bidder_list) + [auctioneer.log()] + [disable_gr,
                                                                                                      disable_gr] + disable_all_box(
                    bidder_list)

        auctioneer.hammer_fall()
        bar.update(1)

    total_cost = sum([b.openai_cost for b in bidder_list]) + auctioneer.openai_cost
    bidder_reports = [bidder.profit_report() for bidder in bidder_list]

    if yield_for_demo:
        chatbot_list = bidders_to_chatbots(bidder_list, profit_report=True)
        yield [bidder_list] + chatbot_list + monitor_all(bidder_list) + [
            auctioneer.log(bidder_reports) + f'\n## Total Cost: ${total_cost}'] + [disable_gr,
                                                                                   enable_gr] + disable_all_box(
            bidder_list)

    memo = {'auction_log': auctioneer.log(show_model_name=False),
            'memo_text': bidder_reports,
            'profit': {bidder.name: bidder.profit for bidder in bidder_list},
            'total_cost': total_cost,
            'learnings': {bidder.name: bidder.learnings for bidder in bidder_list},
            'arms': {},
            'arms_cnt': {bidder.name: bidder.retrieve_tool.cnt if bidder.enable_tool else [] for bidder in bidder_list},
            'model_info': {bidder.name: bidder.model_name for bidder in bidder_list},
            'prune_list': {bidder.name: bidder.prune_list if bidder.enable_tool else [] for bidder in bidder_list},
            }

    tree_list = []
    for bidder in bidder_list:
        if bidder.enable_tool:
            tree = bidder.retrieve_tool.root
            tree_list.append(tree)
            memo['arms'][bidder.name] = assign_labels_to_last_layer(tree)
        else:
            memo['arms'][bidder.name] = {}
            tree_list.append({})

    log_bidders(log_dir, auction_hash, bidder_list, repeat_num, memo, tree_list)

    auctioneer.finish_auction()

    if not yield_for_demo:
        yield total_cost


def log_bidders(log_dir: str, auction_hash: str, bidder_list: List[Bidder], repeat_num: int, memo: dict, tree_list):
    for bidder in bidder_list:
        log_file = f"{log_dir}/{auction_hash}/{bidder.name.replace(' ', '')}-{repeat_num}.jsonl"
        if not os.path.exists(log_file):
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, 'a') as f:
            log_data = bidder.to_monitors(as_json=True)
            f.write(json.dumps(log_data) + '\n')

    with open(f"{log_dir}/{auction_hash}/memo-{repeat_num}.json", 'w') as f:
        f.write(json.dumps(memo) + '\n')
    for bidder, tree in zip(bidder_list, tree_list):
        if bidder.enable_tool:
            with open(f"{log_dir}/{auction_hash}/tree-{bidder.name.replace(' ', '')}-{repeat_num}", 'wb') as f:
                dill.dump(tree, f)


def make_auction_hash():
    return str(int(time.time()))


if __name__ == '__main__':
    import argparse
    import dill
    from src.item_base import create_items
    from src.bidder_base import create_bidders
    from transformers import GPT2TokenizerFast

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', '-i', type=str, default='data/')
    parser.add_argument('--model_path', '-p', type=str,
                        required=True)
    parser.add_argument('--base_url', '-u', type=str,
                        required=True)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--threads', '-t', type=int,
                        help='Number of threads. Max is number of bidders. Reduce it if rate limit is low (e.g., '
                             'GPT-4).',
                        required=True)
    args = parser.parse_args()

    auction_hash = make_auction_hash()
    total_money_spent = 0
    for i in tqdm(range(args.repeat), desc='Repeat'):
        cnt = 3
        while cnt > 0:
            try:
                item_file = os.path.join(args.input_dir, f'items_demo.jsonl')
                bidder_file = os.path.join(args.input_dir, f'bidders_demo.jsonl')
                items = create_items(item_file)
                bidders = create_bidders(bidder_file,
                                         auction_hash=auction_hash,
                                         base_url=args.base_url,
                                         model_path=args.model_path,
                                         )
                auctioneer = Auctioneer(enable_discount=False)
                auctioneer.init_items(items)
                if args.shuffle:
                    auctioneer.shuffle_items()
                money_spent = list(run_auction(
                    auction_hash,
                    auctioneer,
                    bidders,
                    thread_num=min(args.threads, len(bidders)),
                    yield_for_demo=False,
                    log_dir=args.input_dir,
                    repeat_num=i,
                ))
                total_money_spent += sum(money_spent)
                break
            except Exception as e:
                cnt -= 1
                print(f"Error in {i}th auction: {e}\n{trace_back(e)}")
                print(f"Retry {cnt} more times...")

        if cnt <= 0:
            raise ValueError

    print('Total money spent: $', total_money_spent)
