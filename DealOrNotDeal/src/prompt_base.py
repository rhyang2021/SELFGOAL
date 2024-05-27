SYSTEM_MESSAGE = """
Now enter the role-playing mode. In the following conversation, you will play as {player_name}, taking part in a two-way negotiation over a fixed set of items with your partner. 

Here are rules:
- Your profit, upon reaching an agreement, is calculated by multiplying the number of items received by your valuation of those items. The maximum value each of you can gain from the items is limited to 10.
- If an agreement isn't made within twenty rounds, the negotiation will end with no profit for either party.
""".strip()

INSTRUCT_LEARNING_TEMPLATE = """
Review and reflect on the historical data provided from a past negotiation. 

{past_auction_log}

Here are your past learnings:

{past_learnings}

Based on the dialogue log, formulate or update your learning points that could be advantageous to your strategies in the future. Your learnings should be strategic, and of universal relevance and practical use for future negotiations. Consolidate your learnings into a concise numbered list of sentences.

Each numbered item in the list can ONLY be of the form:
X MAY BE NECCESSARY to Y.
X SHOULD BE NECCESSARY to Y.
X MAY BE CONTRIBUTE to Y.
X DOES NOT CONTRIBUTE to Y.
""".strip()

INSTRUCT_REFLEXION_TEMPLATE = """
You are an advanced reasoning agent that can improve based on self refection. 
Review and reflect on the historical data provided from a past negotiation. 

{past_auction_log}

Based on the dialogue log, in a few sentences, diagnose a possible reason for failure or phrasing discrepancy and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences.
""".strip()

INSTRUCT_CLIN_TEMPLATE = """
1. Clear communication of value priorities SHOULD BE NECCESSARY to set negotiation direction and expectations.
2. Flexibility in compromise MAY BE CONTRIBUTE to building a cooperative atmosphere and reaching agreements.
3. Asserting high-value items SHOULD BE NECCESSARY to protect personal interests and maintain leverage.
4. Proposing balanced initial offers SHOULD BE NECCESSARY to streamline the negotiation process and avoid deadlock.
5. Time-bound negotiations SHOULD BE NECCESSARY to ensure efficiency and prevent prolongation.
6. Concise responses SHOULD BE NECCESSARY to maintain focus and prevent unnecessary delays.
7. A clear valuation framework SHOULD BE NECCESSARY for strategic decision-making and allocation.
8. Being prepared to make decisions SHOULD BE NECCESSARY to maximize negotiation outcomes.
9. Demonstrating fairness MAY BE CONTRIBUTE to fostering trust and cooperation.
10. Efficient use of negotiation time SHOULD BE NECCESSARY to reach a mutually beneficial agreement.
""".strip()

REACT_TEMPLATE = """
Here is the dialogue history:

{dialog_history}

As Alice, based on the dialogue history, plan your negotiation strategy step by step, then reply to your partner. Your answer should follow this format:

Think: ... 
Response: ...
""".strip()

ADAPT_TEMPLATE = """
Here is the dialogue history:

{dialog_history}

As Alice, learn from and incorporate information from previous dialogues, e.g. do not repeat previously successful or unsuccesful commands. 
Write an abstract plan to successfully complete the goal: {goal}, then reply to your partner. 
Your answer should follow this format:

Plan: ... 
Response: ...
""".strip()

INSTRUCT_RULE_TEMPLATE = """
Let's play a game. You are {player_name} and you want to negotiate with your partner, about the allocation of items: {books_cnt} books, {hats_cnt} hats, and {balls_cnt} balls, You value books at {books_value}, hats at {hats_value}, and balls at {balls_value}. {desire_desc} Your response to partner must be short and succinct.
""".strip()

FEEDBACK_TEMPLATE = """
Here's the proposal you and your partner agreed on last round: {text} It contains some errors: {errors}
Make sure not to repeat these mistakes in this round.
""".strip()


INSTRUCT_INIT_GOAL_TEMPLATE = """
Humans exhibit numerous behaviors and sub-goals, which can be traced back to the primary aim of survival. For instance,
1. Food Acquisition: To maintain physical and mental functionality, individuals seek nourishment. They target foods with high energy and nutritional values to augment their health, thus enhancing survival possibilities.
2. Shelter Construction: Safe and secure housing is a fundamental human need. It offers protection from potentially harmful natural elements and potential threats.

Image you are an agent involved in a two-way negotiation, discussing how to divide a limited set of items with a partner. Both parties place different values on the items and may have separate goals for this negotiation.
Here are some must-known rules for this negotiation:
- You must allocate each item (balls, hats, books) in whole numbers (e.g., half a ball/book/hat is not allowed).
- Your profit, upon reaching an agreement, is the sum of items you receive times their value to you. The most you can earn from the items is 10.
- If an agreement isn't made within twenty rounds, the negotiation will end with no profit for either party.

Taking analogy from human behaviors, suppose your fundamental goal is "{goal}". What sub-goals you might have?
""".strip()

INSTRUCT_GOAL_TEMPLATE = """
Here's the current scenario:

{scene}

---

As Alice, for the goal: "{sub_goal}", based on current state, can you further run some deduction for fine-grained goals or brief guidelines?
""".strip()

INSTRUCT_RANKING_TEMPLATE = """
Here's the current scenario:

{scene}

---

To better reach your main goal: {objective}, in this context, please do the following:
1.Evaluate how the sub-goals listed below can assist you in reaching your main goal given the present circumstances.

Sub-goals:

{guidance}

2. Select {beam_width} most useful sub-goals that will help you reach your main goal in the current situation, and note their IDs.

Start by explaining your step-by-step thought process. Then, list the {beam_width} IDs you've chosen, using the format of this example: {{"IDs": [1, 3, 10, 21, 7]}}.
""".strip()

INSTRUCT_INIT_PLAN_SCENE_TEMPLATE = """
As {player_name}, Now you are going to attend the initial round of this negotiation, you have to create an allocation plan and persuade your partner to accept it.
""".strip()

INSTRUCT_DECISION_SCENE_TEMPLATE = """
As {player_name}, you've had several rounds of discussion with your partner and are preparing for a new round to create and discuss an allocation plan.

Here is the previous conversation:

{history_dialogue}

""".strip()

PARSE_ID_SYSTEM_MESSAGE = """
Please provide a JSON object that accurately represents the given text. The JSON object should include the following information:
- The scale evaluate the helpfulness of the guideline.

For example, the given text is:

'''
Thought Process:
1. First, I need to understand the value of each item and my budget. The total value of all items is $19,000 and my budget is $15,000. This means I cannot buy all items and need to make a strategic decision on which items to bid for.
2. I need to consider the potential profit of each item. The items with a value of $5,000 will bring more profit than the ones valued at $2,000. However, I also need to consider the competition. If other bidders also target the high-value items, the bidding price might go up and reduce my profit.
3. I should also consider the number of items. If I bid for the two high-value items, I will spend $10,000 and have $5,000 left for the other items. This means I can only bid for one or two more items. If I bid for the low-value items, I can buy more items and potentially make more profit.
4. I need to consider the bidding strategy. If I bid high for the high-value items, I might scare off other bidders and secure these items. However, this also means I have less money for the other items. If I bid low, I might lose the high-value items but have more money for the other items.
5. Finally, I need to consider the risk. Bidding for the high-value items is riskier because I might end up with fewer items and less profit. Bidding for the low-value items is safer because I can buy more items and have a higher chance of making a profit.
Based on this thought process, I would choose the following IDs: {"IDs": [2, 4, 6, 8, 10]}. These guidelines suggest to understand the value of each item, consider the potential profit, consider the number of items, consider the bidding strategy, and consider the risk.
'''

Then, the JSON object should strictly follow this format and structure:

{"IDs": [2, 4, 6, 8, 10]}
""".strip()

PARSE_RESPONSE_SYSTEM_MESSAGE = """
Please provide a JSON object that accurately represents the given text. The JSON object should include the following information:
- The response in the given text.

For example, the given text is:

'''
Think: I should start by understanding Bob's preferences and priorities. I also need to assess the value he assigns to each item. 
Response: Sure, let's start by discussing your preferences and the value you assign to each item.
'''

Then, the JSON object should strictly follow this format and structure:

{"response": "Sure, let's start by discussing your preferences and the value you assign to each item."}
""".strip()


PARSE_GOALS_SYSTEM_MESSAGE = """
Please provide a JSON object that accurately represents the given text. The JSON object should include the following information:
- The deduced sub-goals and its detailed information, list as a key-value pair(e.g. {"Learn from past experiences": "Reflect on previous auctions and analyze situations where you dropped out or continued bidding. Identify patterns or indicators that proved to be accurate predictors of when to drop out. Incorporate these lessons into your strategy to make more informed decisions in future auctions."})

For example, the given text is:

'''
Certainly! Here are some fine-grained goals or guidelines for the sub-goal of "Knowing when to drop out":

1. Set a predetermined maximum price: Before participating in the auction, determine the absolute maximum price you are willing to pay for the item. This ensures that you have a clear threshold beyond which you will not continue bidding, helping to prevent overpaying and maximizing your potential profit.

2. Monitor the bidding activity: Pay close attention to the bidding activity during the auction. Observe how other bidders are responding and assess their willingness to keep increasing their bids. If it becomes clear that the bidding has escalated beyond a point that aligns with your maximum price, it may be prudent to drop out to avoid overpaying.

3. Consider the current market value: Continuously assess the current market value of the item being auctioned. If the bidding surpasses the market value and exceeds what you believe is a fair price, it may be a sign to drop out and avoid overpaying.
'''

Then, the JSON object should strictly follow this format and structure:

{"Set a predetermined maximum price": "Before participating in the auction, determine the absolute maximum price you are willing to pay for the item. This ensures that you have a clear threshold beyond which you will not continue bidding, helping to prevent overpaying and maximizing your potential profit.",
 "Monitor the bidding activity": "Pay close attention to the bidding activity during the auction. Observe how other bidders are responding and assess their willingness to keep increasing their bids. If it becomes clear that the bidding has escalated beyond a point that aligns with your maximum price, it may be prudent to drop out to avoid overpaying.",
 "Consider the current market value": "Continuously assess the current market value of the item being auctioned. If the bidding surpasses the market value and exceeds what you believe is a fair price, it may be a sign to drop out and avoid overpaying."
 }
""".strip()

CHECK_GOALS_TEMPLATE = """
Your task is to determine if the sub-goal provided is an executable action. Here are the rules:
- If the sub-goal is executable, output 1.
- If the sub-goal is not executable, output 0.
Here is the sub-goal:

{response}

Don't say anything else other than just a number: either 1 or 0.
""".strip()

INSTRUCT_REFINE_TEMPLATE = """
Review and reflect on the historical dialog provided from a past negotiation. 

{past_auction_log}

Here is the guidance list you have learned from past auctions:

{guidance}

---

Think about a list of concise key learning points that can supplement your existing guidance list. These new insights could help Alice better achieve her goal: {goal}. Your learnings should be strategic, and of universal relevance and practical use for future negotiations. Consolidate your learnings into a concise numbered list of sentences.
""".strip()

INSTRUCT_PRUNE_TEMPLATE = """
Here is the guidance list you have learned from past auctions:

{guidance}

---
Here is the dialog provided from a past negotiation:

{past_auction_log}

Based on the current scenario, identify a subset of guidelines(10 guidelines) from the list that are most not relevant.
Give your reason first. Then, list the IDs you've selected that are not relevant, using the following format as an example: {{"IDs": [1, 3, 10, 21, 7, 6, 18, 20, 33, 12]}}.
""".strip()

OBJECTIVE_TEMPLATE = """
Your primary objective is to secure the highest profit at the end of this auction, compared to all other bidders. Based on the current scenario, here is some guidance to help you attain your primary objective:

{guidance}

""".strip()
