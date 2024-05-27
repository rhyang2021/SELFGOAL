from typing import List
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from .bidder_base import Bidder, draw_plot
from .item_base import Item
from langchain.input import get_colored_text
import time


class HumanBidder(Bidder):
    name: str
    human_name: str = "Adam"
    budget: int
    auction_hash: str
    
    cur_item_id: int = 0
    items: list = []
    withdraw: bool = False
    
    engagement_count: int = 0
    original_budget: int = 0
    profit: int = 0
    items_won: list = []
    
    all_bidders_status: dict = {}         # track others' profit
    
    # essential for demo
    need_input: bool = False
    semaphore: int = 0              # if needs input, then semaphore is set as 1, else waits.
    input_box: str = None           # global variable for accepting user input
    custom_api_base: str = ''
    
    # not used
    model_name: str = 'human'
    openai_cost: int = 0
    desire: str = ''
    plan_strategy: str = ''
    correct_belief: bool = True
    
    class Config:
        arbitrary_types_allowed = True
    
    def get_plan_instruct(self, items: List[Item]):
        self.items = items
        plan_instruct = "As {bidder_name}, you have a total budget of ${budget}. This auction has a total of {item_num} items to be sequentially presented, they are:\n{items_info}".format(
            bidder_name=self.name,       
            budget=self.budget, 
            item_num=len(items), 
            items_info=self._get_items_value_str(items)
        )
        return plan_instruct
    
    def init_plan(self, plan_instruct: str):
        # Human = auctioneer, AI = bidder
        self.dialogue_history += [
            HumanMessage(content=plan_instruct),
            AIMessage(content='(Getting ready...)')
        ]
        return ''
    
    def get_bid_instruct(self, auctioneer_msg, bid_round):
        self.dialogue_history += [
            HumanMessage(content=auctioneer_msg), 
            AIMessage(content='')
        ]
        return auctioneer_msg
    
    def bid(self, bid_instruct):
        # wait for the cue to handle user input
        while self.semaphore <= 0:
            time.sleep(1)
        
        self.dialogue_history += [
            HumanMessage(content=''),
            AIMessage(content=self.input_box)
        ]
        self.semaphore -= 1
        self.need_input = False
        return self.input_box
    
    def get_summarize_instruct(self, bidding_history: str, hammer_msg: str, win_lose_msg: str):
        instruct_summarize = f"{bidding_history}\n\n{hammer_msg}\n{win_lose_msg}"
        return instruct_summarize
    
    def summarize(self, instruct_summarize: str):
        self.dialogue_history += [
            HumanMessage(content=instruct_summarize),
            AIMessage(content='(Taking notes...)')
        ]
        self.budget_history.append(self.budget)
        self.profit_history.append(self.profit)
        return ''
    
    def get_replan_instruct(self):
        return ''

    def replan(self, instruct_replan):
        self.withdraw = False
        self.cur_item_id += 1
        return ''
    
    def to_monitors(self, as_json=False):
        items_won = []
        for item, bid in self.items_won:
            items_won.append([str(item), bid, item.true_value])
        if as_json:
            return {
                'auction_hash': self.auction_hash,
                'bidder_name': self.name,
                'human_name': self.human_name,
                'model_name': self.model_name,
                'budget': self.original_budget,
                'money_left': self.budget,
                'profit': self.profit,
                'items_won': items_won,
                'engagement_count': self.engagement_count,
            }
        else:
            return [
                self.budget, 
                self.profit, 
                items_won, 
                0, 
                0, 
                round(self.failed_bid_cnt / (self.total_bid_cnt+1e-8), 2), 
                0, 
                0, 
                self.engagement_count,
                draw_plot(f"{self.name} ({self.model_name})", self.budget_history, self.profit_history), 
                [],
                [],
                [], 
                []
            ]
    