import numpy
import random
import time
import os
import json
import matplotlib.pyplot as plt
import pandas

from th_rl.environments import *
from th_rl.agents import *

def plot_learning_curve(rlog, alog, labels=['Agent 1', 'Agent 2'],smooth = 1000):
    df = pandas.DataFrame(rlog)
    df["Nash"] = 22.2
    df["Collusion"] = 25
    df.columns = labels + ['Nash','Collusion']
    df['Total'] = df[labels].sum(axis=1)
    f, ax = plt.subplots(2, 1)
    df.ewm(halflife=smooth).mean().plot(ax=ax[0])
    ax[0].set_ylim([10, 25])
    ax[0].grid()
    ax[0].set_title('Rewards')
    
    df = pandas.DataFrame(data=alog, columns=labels)
    df.ewm(halflife=smooth).mean().plot(ax=ax[1])
    ax[1].grid()
    ax[1].set_title('Actions')

def create_game(configpath):
    # Load config
    config = json.load(open(configpath))

    # Create agents
    agents = [eval(agent["name"])(**agent) for agent in config["agents"]]

    # Create environment
    assert (
        len(agents) == config["environment"]["nplayers"]
    ), "Bad config. Check number of agents."
    environment = eval(config["environment"]["name"])(**config["environment"])

    return config, agents, environment


def train_one(exp_path, configpath):
    # Handle home location
    if not os.path.exists(exp_path):
        os.mkdir(os.path.join(exp_path))

    config, agents, environment = create_game(configpath)

    # Init Logs
    epochs = config.get("training", {}).get("epochs", 0)
    pfreq = config.get("training", {}).get("print_freq", epochs // 20)
    plot = config.get("training", {}).get("plot", False)
    actions = numpy.zeros((len(agents),))
    rlog = numpy.zeros((epochs, len(agents)))
    alog = numpy.zeros((epochs, len(agents)))
    t = time.time()
    outs = [[],[]]
    state = numpy.random.rand()*10
    for e in range(epochs):
        # Sample quantities
        actions[0], outs[0] = agents[0].sample(state)
        actions[1], outs[1] = agents[1].sample(state)

        # Run env
        px, rewards = environment(actions)

        # Remember
        for i in range(2):
            if type(agents[i]).__name__=='Reinforce':
                agents[i].memory.append(rewards[i], outs[i]) 
            elif type(agents[i]).__name__=='Exp3':
                agents[i].memory.append(actions[i], rewards[i])
            else:
                agents[i].memory.append(state, actions[i], rewards[i], px)

        # Update Qs
        [agents[i].train() for i in range(2)]

        # Update state
        state = px

        # Log
        rlog[e] = rewards
        alog[e] = actions

        if pfreq!=0 and (not (e + 1) % pfreq):
            print(
                "Epoch:{} \t Time:{:0.2f}min \t R1:{:0.2f} \t R2:{:0.2f} \t A1:{:0.2f} \t A2:{:0.2f}".format(
                    e + 1,
                    (time.time() - t) / 60,
                    rlog[e - pfreq + 1 : e + 1, 0].mean(),
                    rlog[e - pfreq + 1 : e + 1, 1].mean(),
                    alog[e - pfreq + 1 : e + 1, 0].mean(),
                    alog[e - pfreq + 1 : e + 1, 1].mean(),
                )
            )

    # Store result
    for i, a in enumerate(agents):
        a.save(os.path.join(exp_path, str(i)))

    with open(os.path.join(exp_path, "config.json"), "w") as f:
        json.dump(config, f, indent=3)

    rpd = pandas.DataFrame(data=rlog, columns=numpy.arange(len(agents)))
    apd = pandas.DataFrame(data=alog, columns=numpy.arange(len(agents)))
    rpd = rpd.rolling(window=100).mean()[::100]
    apd = apd.rolling(window=100).mean()[::100]

    log = pandas.concat([rpd, apd], axis=1, keys=["rewards", "actions"])
    log.to_csv(os.path.join(exp_path, "log.csv"), index=None)

    # Plot learning curve
    if plot:
        plot_learning_curve(rlog, alog,[type(Q).__name__ for Q in agents], smooth=10)
