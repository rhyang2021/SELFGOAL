import trueskill
import ujson as json
import os
import re
import matplotlib.pyplot as plt
import numpy as np


def compute_trueskill_scores(matches):
    """
    Compute TrueSkill ratings and scores for a series of multi-agent matches.

    Parameters:
    - matches: A list of matches, where each match is represented as a tuple of:
      - teams: A list of teams, where each team is a list of agent names.
      - ranks: A list of ranks (starting from 0) corresponding to the teams. A lower rank is better.

    Returns:
    - scores: A dictionary where keys are agent names and values are their scores.
    """

    # Initialize TrueSkill environment
    env = trueskill.TrueSkill()
    default_rating = env.create_rating()

    # A dictionary to store the ratings of each agent
    ratings = {}
    profit_sums = {}
    profit_sums_squared = {}
    walkways_sums = {}
    match_counts = {}

    # Process each match
    for match in matches:
        teams, ranks, profits, walkways = match

        # Convert team names to their current ratings (or default ratings if not yet rated)
        team_ratings = [[ratings.get(agent, default_rating) for agent in team] for team in teams]

        # Update ratings based on match outcome
        updated_ratings = env.rate(team_ratings, ranks=ranks)

        # Store updated ratings
        for team, team_updated_ratings, profit, walkway in zip(teams, updated_ratings, profits, walkways):
            for agent, rating in zip(team, team_updated_ratings):
                ratings[agent] = rating
                walkways_sums[agent] = walkways_sums.get(agent, 0) + walkway
                if walkway:
                    profit_sums[agent] = profit_sums.get(agent, 0) + profit
                    profit_sums_squared[agent] = profit_sums_squared.get(agent, 0) + profit ** 2
                    match_counts[agent] = match_counts.get(agent, 0) + 1

    # Convert ratings to scores (using the mu value)
    scores = {agent: (rating.mu, rating.sigma) for agent, rating in ratings.items()}
    profits_stats = {}
    walkways_stats = {}
    for agent in ratings.keys():
        mean_profit = profit_sums[agent] / match_counts[agent]
        variance_profit = np.sqrt((profit_sums_squared[agent] / match_counts[agent]) - mean_profit ** 2)
        mean_walkaways = walkways_sums[agent]/40
        profits_stats[agent] = (mean_profit, variance_profit)
        walkways_stats[agent] = (mean_walkaways)
    print(match_counts)
    return scores, profits_stats, walkways_stats


def get_all_rounds(directory_path="exp"):
    """
    This function returns a dictionary where each key is a unique round and the corresponding value is
    a list of JSONL files for that round.

    Parameters:
    - directory_path: The directory where the bidder files are located. Default is "exp".

    Returns:
    - A dictionary with unique rounds as keys and corresponding bidder files as values.
    """

    all_files = os.listdir(directory_path)

    # Extract all unique rounds from the filenames using regex
    rounds = set(
        re.search(r'-([0-9]+)\.jsonl', file).group(1) for file in all_files if re.search(r'-([0-9]+)\.jsonl', file))

    # For each unique round, get the corresponding bidder files
    round_files = {int(r): [os.path.join(directory_path, f) for f in all_files if f.endswith(f"-{r}.jsonl")] for r in
                   rounds}

    return round_files


def load_json_files(files):
    data = []
    for file in files:
        with open(file, 'r') as f:
            for line in f:
                data.append(json.loads(line))
    return data


def json_list_to_rank(json_list):
    global title_var

    flag = True
    flag_list = []
    for js in json_list:
        temp_flag = True
        if not isinstance(js["profits"], dict):
            temp_flag = False
            flag = False
        else:
            if not isinstance(js["profits"]["alice"], (float, int)):
                temp_flag = False
                flag = False
            elif not isinstance(js["profits"]["bob"], (float, int)):
                temp_flag = False
                flag = False
            elif js["profits"]["bob"] == 0 and js["profits"]["alice"] == 0:
                temp_flag = False
                flag = False
        flag_list.append(temp_flag)

    # Create a list of tuples (bidder_name, profit)
    bidders = []
    for js, flag in zip(json_list, flag_list):
        if flag:
            bidders.append((f"{js['enable_adapt']}-{js['retrieve_strategy']}", abs(js["profits"]["alice"] - js["profits"]["bob"]),
                js["retrieve_strategy"] =='gpt-3.5', flag))
        else:
            bidders.append((f"{js['enable_adapt']}-{js['retrieve_strategy']}", 0,
                js["retrieve_strategy"] =='gpt-3.5', flag))

    # Sort the list of tuples in descending order of profit
    sorted_bidders = sorted(bidders, key=lambda x: (x[1], -x[2]))
    bidder_names = []
    ranks = []
    profits = []
    alice = []
    walkways = []

    for i, b in enumerate(sorted_bidders):
        bidder_names.append([b[0]])
        ranks.append(i)
        profits.append(b[1])
        alice.append(b[2])
        walkways.append(b[3])

    return bidder_names, ranks, profits, alice, walkways, flag


def test():
    # Example matches
    matches = [
        ([["A"], ["B"], ["C"], ["D"]], [0, 1, 2, 3]),
        ([["A"], ["B"], ["C"], ["D"]], [0, 1, 2, 3]),
        ([["A"], ["B"], ["C"], ["D"]], [0, 1, 2, 3]),
        ([["A"], ["B"], ["C"], ["D"]], [0, 3, 1, 2]),
    ]

    # Compute scores for the example matches
    print(compute_trueskill_scores(matches))


if __name__ == '__main__':
    import argparse
    from pprint import pprint
    import matplotlib.pyplot as plt
    import random
    random.seed(41)

    title_var = ''
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', '-d', type=str)
    args = parser.parse_args()

    all_rounds = get_all_rounds(args.dir)
    all_rounds = {k: all_rounds[k] for k in sorted(all_rounds.keys())}
    matches = []
    for k, v in all_rounds.items():
        json_obj_list = load_json_files(v)
        try:
            bidder_names, ranks, profits, alice, walkways, flag = json_list_to_rank(json_obj_list)
            print(bidder_names, ranks, profits, walkways)
            matches.append((bidder_names, ranks, profits, walkways))
        except Exception as e:
            print(f"Error processing {k}: {e}")
            continue

    final_scores, profit_scores, walkaways = compute_trueskill_scores(matches)
    print(final_scores)
    print(profit_scores)
    print(walkaways)

    agents = list(final_scores.keys())
    agents.sort()
    mu_values = [final_scores[agent][0] for agent in agents]
    sigma_values = [final_scores[agent][1] for agent in agents]

    # Bar chart visualization with enhanced appearance
    plt.figure(figsize=(10, 7))
    colors = ['#f6eff7', '#bdc9e1', '#67a9cf', '#1c9099', '#016c59']
    plt.bar(agents, mu_values, width=0.7, yerr=sigma_values, capsize=10, color=colors, edgecolor='black', alpha=0.7,
            error_kw=dict(lw=1.5, capthick=1.5, ecolor='black'))

    font_size = 24
    plt.xlabel("Bidders", fontsize=font_size)
    plt.xticks(fontsize=font_size - 1, rotation=15)
    plt.yticks(fontsize=font_size - 4)
    plt.ylabel("TrueSkill Score", fontsize=font_size)
    plt.title(f"Ranking for {title_var}-driven Players", fontsize=font_size + 1)
    plt.tight_layout()
    plt.savefig('trueskill.png')
