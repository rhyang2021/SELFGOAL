<p align="center">
    <img src="resource/icon.png" width="20%"> <br>
</p>
<h1 align="center">SELFGOAL: Your Language Agents
Already Know How to Achieve High-level Goals </h1>


<p align="center">
[<a href="https://selfgoal-agent.github.io/">Website</a>]
[<a href="http://arxiv.org/abs/2406.04784">Paper</a>] 

</p>



# SELFGOAL

SELFGOAL is a non-parametric learning algorithm for language agents, i.e., without parameter update. Concretely, SELFGOAL is featured with two key modules, Decomposition and Search which construct and utilize a subgoal tree respectively, namely GoalTree, to interact with the environment.

<p align="center">
    <img src="resource/main_new.jpg" width="100%"> <br>
</p>

More details are in the paper.

## Setup Environment

Create a conda environment and install dependency:
```bash
conda create -n selfgoal python=3.11
conda activate selfgoal
pip install -r requirements.txt
```
Here, we use ``AucArena`` as an example to illustrate the Decompose and Search Modules and how the agent utilizes guidance to make decisions.
## Decompose Module
To adapt SELFGOAL to your current scenario, you have to describe your environment in ``prompt_base.py``

**Main Goal Decomposition**
```bash
# main goal
Imagine you are an agent in a {scene}. 

Taking an analogy from human behaviors, if your fundamental objective in this scenario is "{goal}", what sub-goals you might have?
```

**Sub-Goal Decomposition**
```bash
# sub-goals
{scene}

---

For the goal: "{sub_goal}", based on the current state, can you further run some deduction for fine-grained goals or brief guidelines?
```
## Search Module
At each stage, you must also supply SELFGOAL with the current state to enable the Search Module to identify the most useful sub-goals.

```bash
# Sub-Goal
Here's the current scenario:

{scene}

------------------------------
For the goal: "{sub_goal}", can you further run some deduction for fine-grained goals or brief guidelines?
```
## An example of guidance
The generated sub-goal will be displayed as shown in the following example:

```bash
Based on the current auction scenario, here are some derived sub-goals and detailed guidance to help you better achieve your primary objective, i.e., maximize your total profit: 
* Sub-goal: Market Value Research: Understand the current market value of the item being auctioned. This will help you estimate the potential profit and decide if the bidding price is justified.
* Sub-goal: Profit Margin Calculation: Calculate the potential profit margin based on the current bidding price and the market value of the item. This will help you understand if the cost of bidding is worth the potential benefit.

Please consider these sub-goals and detailed advice in your next round of strategic planning and action execution in the auction to help achieve your primary objective.
```

## Act Module
To prompt the agent to use the selected sub-goals in its decision-making process, incorporate the guidance into the system message.

```bash
Here are some possible subgoals and guidance derived from your primary objective:

{sub-goals}

In this round, You may target some of these subgoals and guidance to improve your bidding strategy and action, in order to achieve your primary objective.
```
You should also prompt agents in each round of decision-making, for example, using ``Reflect on derived sub-goals``.

## Run
#### For Open Source Models
For open source models, deploy the model as an API using <a href="https://docs.vllm.ai/en/latest/">vLLM</a>

```bash
python -m vllm.entrypoints.openai.api_server --model facebook/opt-125m --port 8001
```
### AucArena
```bash
python auction_workflow.py --shuffle --repeat 10 -t 4
```

### DealOrNotDeal
```bash
python gpt_bargain.py
```

### GAMABench
```bash
python guessing_game.py
python public_goods.py
```


## Contact

If you have any problems, please contact [Ruihan Yang](mailto:rhyang17@fudan.edu.cn).


## Citation
```
@article{yang2024selfgoallanguageagentsknow,
      title={SelfGoal: Your Language Agents Already Know How to Achieve High-level Goals}, 
      author={Ruihan Yang and Jiangjie Chen and Yikai Zhang and Siyu Yuan and Aili Chen and Kyle Richardson and Yanghua Xiao and Deqing Yang},
      year={2024},
      eprint={2406.04784},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2406.04784}, 
}
```

