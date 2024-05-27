SYSTEM_MESSAGE = """
Now enter the role-playing mode. In the following conversation, you will play as {player_name}, taking part in a two-way negotiation over a fixed set of items with your partner. 

Here are rules:
- Your profit, upon reaching an agreement, is calculated by multiplying the number of items received by your valuation of those items. The maximum value each of you can gain from the items is limited to 10.
- If an agreement isn't made within twenty rounds, the negotiation will end with no profit for either party.
""".strip()

INSTRUCT_LEARNING_TEMPLATE = """
Review and reflect on the historical data provided from past game rounds. 

{past_game_log}

Here are your past learnings:

{past_learnings}

Based on the game log, formulate or revise learning points that could benefit your future strategies. Ensure your learnings are strategic, universally relevant, and practically applicable in upcoming game rounds. Summarize your learnings in a concise numbered list of sentences.
Each numbered item in the list can ONLY be of the form:
X MAY BE NECCESSARY to Y.
X SHOULD BE NECCESSARY to Y.
X MAY BE CONTRIBUTE to Y.
X DOES NOT CONTRIBUTE to Y.
""".strip()

INSTRUCT_REFLEXION_TEMPLATE = """
You are an advanced reasoning agent that can improve based on self refection. 
Review and reflect on the historical data provided from past game rounds. 

{past_game_log}

Based on the game log, in a few sentences, diagnose a possible reason for failure or phrasing discrepancy and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences.
""".strip()

REACT_TEMPLATE = """
Here is the dialogue history:

{dialog_history}

As Alice, based on the dialogue history, plan your negotiation strategy step by step, then reply to your partner. Your answer should follow this format:

Think: ... 
Response: ...
""".strip()

INSTRUCT_RULE_TEMPLATE = """
Let's play a game. You are {player_name} and you want to negotiate with your partner, about the allocation of items: {books_cnt} books, {hats_cnt} hats, and {balls_cnt} balls, You value books at {books_value}, hats at {hats_value}, and balls at {balls_value}. {desire_desc} Your response to partner must be short and succinct.
""".strip()

FEEDBACK_TEMPLATE = """
Here's the proposal you and your partner agreed on last round: {text} It contains some errors: {errors}
Make sure not to repeat these mistakes in this round.
""".strip()

'''
INSTRUCT_INIT_GOAL_TEMPLATE = """
Humans exhibit numerous behaviors and sub-goals, which can be traced back to the primary aim of survival. For instance:
1. Food Acquisition: To maintain physical and mental functionality, individuals seek nourishment. They target foods with high energy and nutritional values to augment their health, thus enhancing survival possibilities.
2. Shelter Construction: Safe and secure housing is a fundamental human need. It offers protection from potentially harmful natural elements and potential threats.

Image you are an agent participating in a game played by 10 players repeat 20 rounds. Each player selects an integer number between 0 and 100, inclusive. After all selections are made, the average of all chosen numbers is calculated. The target number is 2/3 of this average. The winner is the player(s) who selected a number closest to the target number.
Taking analogy from human behaviors, suppose your fundamental goal is "{goal}". What sub-goals you might have?
""".strip()

'''

INSTRUCT_INIT_GOAL_TEMPLATE = """
Humans exhibit numerous behaviors and sub-goals, which can be traced back to the primary aim of survival. For instance:
1. Food Acquisition: To maintain physical and mental functionality, individuals seek nourishment. They target foods with high energy and nutritional values to augment their health, thus enhancing survival possibilities.
2. Shelter Construction: Safe and secure housing is a fundamental human need. It offers protection from potentially harmful natural elements and potential threats.

Image you are an agent participating in a game played by 10 players repeat 20 rounds. In each round, you, as a player, must decide how many of your private tokens you wish to contribute secretly to the public pot. The total tokens in this pot will be multiplied by the factor 2 to create the "public good" payoff. This payoff will then be evenly divided among all players, regardless of their individual contribution. Any tokens you do not contribute will be retained in your private collection.
Taking analogy from human behaviors, suppose your fundamental goal is "{goal}". What sub-goals you might have?
""".strip()


'''
INSTRUCT_INIT_GOAL_TEMPLATE = """
Humans exhibit numerous behaviors and sub-goals, which can be traced back to the primary aim of survival. For instance:
1. Food Acquisition: To maintain physical and mental functionality, individuals seek nourishment. They target foods with high energy and nutritional values to augment their health, thus enhancing survival possibilities.
2. Shelter Construction: Safe and secure housing is a fundamental human need. It offers protection from potentially harmful natural elements and potential threats.

Image you are an agent participating in a game played by 10 players repeat 20 rounds. You are in a survival game where only one can survive and win. Players take turns shooting at others in a predetermined order based on their hit rates, from the lowest to the highest (higher hit rate means higher probabilities of hitting the target). You have an unlimited number of bullets. You may choose to intentionally miss your shot on your turn. 

Taking analogy from human behaviors, suppose your fundamental goal is "{goal}". What sub-goals you might have?
""".strip()
'''

INSTRUCT_GOAL_TEMPLATE = """
Here's the current scenario:

{scene}

---

For the goal: "{sub_goal}", based on current state, can you further run some deduction for fine-grained goals or brief guidelines?
""".strip()

INSTRUCT_RANKING_TEMPLATE = """
Here's the current scenario:

{scene}

---

To better reach your main goal: "{objective}", in this context, please do the following:
1.Evaluate how the sub-goals listed below can assist you in reaching your main goal given the present circumstances.

Sub-goals:

{guidance}

2. Select {beam_width} most useful sub-goals that will help you reach your main goal in the current situation, and note their IDs.

Start by explaining your step-by-step thought process. Then, list the {beam_width} IDs you've chosen, using the format of this example: {{"IDs": [1, 3, 10, 21, 7]}}.
""".strip()


INSTRUCT_DECISION_SCENE_TEMPLATE = """
As {player_name}, you've had several rounds of game with other players and are preparing for a new round.

Here is the previous game log:

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
Your task is to create a JSON object that accurately represents the given text, while removing all backslashes during the extraction process and ensuring no newline characters are used. The JSON object should include the following information:

An "explanation" key with a value that describes the reasoning behind the chosen number (e.g., "explanation": "Given that the average has been decreasing, I assume that the other players will continue to choose lower numbers. Therefore, I will try to choose a number higher than the average to increase my chances of being closer to 2/3 of the new average. However, I don't want to choose a number too high that I will be far from the target number. Based on the trend, I estimate the average to be around 3-4. I will choose a number around 10 to have a good balance between being higher than the average and not being too far from the target number.")
A "chosen_number" key with a value that represents the selected number (e.g., "chosen_number": 10)
For example, the given text is:

'''
 {"explanation": "Based on the historical data, the average chosen number has been decreasing in recent rounds. To estimate the target number more accurately, I will calculate 2/3 of the average number chosen by all players. I will also choose a number closer to the estimated target number to increase my chances of winning.

Average Number Chosen: 4.3
Target Number (2/3 of Average): 2.87

To calculate a number closer to the target number, I will aim for a number that is approximately 30% greater than the target number. This will give me some room for error and still keep my selection within the range of 0 to 100.

Chosen Number: 3.65 (rounded to the nearest tenth)

Therefore, the JSON format would be:

{\"explanation\": \"Based on the historical data, I calculated the target number to be approximately 2.87. I chose a number that is 30% greater than the target number, resulting in a chosen number of 3.65.\", \"chosen\_number\": 3.65}
"}
'''

Then, the JSON object should strictly follow this format and structure:

{"explanation": "Based on the historical data, I calculated the target number to be approximately 2.87. I chose a number that is 30% greater than the target number, resulting in a chosen number of 3.65.", "chosen_number": 3.65}
""".strip()

PARSE_RESPONSE_SYSTEM_MESSAGEv2 = """
Your task is to create a JSON object that accurately represents the given text, while removing all backslashes during the extraction process and ensuring no newline characters are used. The JSON object should include the following information:

An "explanation" key with a value that describes the reasoning behind the contribution tokens (e.g., "explanation": "I've considered the possibility of contributing all tokens to maximize the public good, but also the need to maintain some tokens for future rounds. By contributing 8 tokens, I strike a balance between contributing enough to potentially benefit from the increased pot and not revealing my entire stash too early.")
A "tokens_contributed" key with a value that represents the contributed tokens to public pot (e.g., "tokens_contributed": 8)
For example, the given text is:

'''
{"explanation": "In this round, I will consider contributing a number of tokens that maximizes my total tokens at the end of the game. I will calculate the expected gain from contributing a certain number of tokens and compare it to the gain I would get by keeping all my tokens. I will then contribute the number of tokens that results in the higher expected total.

Let's calculate the expected gain from contributing x tokens:

Expected gain = (Total tokens after round x / Total number of players) * (Total tokens in public pot after round x * 2) / Total number of rounds

Let's assume the total number of rounds is 20 and the total number of players is 5.

First, let's calculate the total tokens after round x if I contribute x tokens:
Total tokens after round x = (My tokens after round x - 1) + (Public pot tokens after round x * 2 / Total number of players)

Let's calculate the public pot tokens after round x:
Public pot tokens after round x = Contributed tokens of all players in round x * 2

Now, let's calculate the expected gain:
Expected gain = ((My tokens after round x / 5) * (Public pot tokens after round x * 2 / 5) / 20)

Let's calculate the expected gain for each possible contribution from 0 to 20 tokens. I will then choose the contribution that results in the highest expected gain.

Here's the Python code to calculate the expected gain for each contribution:
```python
total_rounds = 20
total_players = 5

expected_gains = []

for contribution in range(0, 21):
 tokens_after_round = (20 - contribution + (public_pot_tokens * 2 / total_players) / total_players)
 expected_gain = ((tokens_after_round / total_players) * (public_pot_tokens * 2 / total_players) / total_rounds)
 expected_gains.append(expected_gain)

max_expected_gain = max(expected_gains)
max_expected_gain_index = expected_gains.index(max_expected_gain)

contribution = max_expected_gain_index

print(f"The contribution that maximizes my total tokens at the end of the game is {contribution} tokens.")

tokens_contributed = contribution
```

Based on the calculation, the contribution that maximizes my total tokens at the end of the game is 11 tokens.

{"explanation": "The contribution that maximizes my total tokens at the end of the game is 11 tokens.", "tokens_contributed": "11"}
"The contribution that maximizes my total tokens at the end of the game is 11 tokens."

Note: The actual calculation of the expected gains and the optimal contribution may vary depending on the contributions of the other players in each round. This is just a simplified analysis assuming all other players contribute randomly. In practice, it would be necessary to take into account the historical contributions of the other players and adjust the analysis accordingly.

I hope this explanation helps clarify my thought process. Let me know if you have any questions or if there's anything else I can help you with!", "tokens\_contributed": "11"}

'''

Then, the JSON object should strictly follow this format and structure:

{"explanation": "The contribution that maximizes my total tokens at the end of the game is 11 tokens.", "tokens_contributed": 11}
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

Think about a list of concise key learning points that could help Alice better achieve her goal: {goal}. Your learnings should be strategic, and of universal relevance and practical use for future negotiations. Consolidate your learnings into a concise numbered list of sentences.
""".strip()

OBJECTIVE_TEMPLATE = """
Your primary objective is to secure the highest profit at the end of this auction, compared to all other bidders. Based on the current scenario, here is some guidance to help you attain your primary objective:

{guidance}

""".strip()

