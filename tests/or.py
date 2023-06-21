from th_rl.agents import QTable
from th_rl.environments import Cournot
import sys
import numpy
import time
import matplotlib.pyplot as plt
import pandas

env = Cournot()


def qtable1():
    epochs = 1000000
    alpha = [0.1, 0.1]
    gamma = [0.98, 0.98]
    eps_end = [0.0, 0.0]
    epsilon = [1, 1]
    nactions = 100
    action_range = [[0, 5], [0, 0]]
    eps_step = 1 - 5 / epochs
    env = Cournot()
    QS = [
        QTable(
            alpha=alpha[i],
            gamma=gamma[i],
            eps_end=eps_end[i],
            epsilon=epsilon[i],
            actions=nactions,
            states=nactions,
            eps_step=eps_step,
            action_range=action_range[i],
            min_memory=10,
            capacity=10,
        )
        for i in range(2)
    ]

    pfreq = epochs // 20
    rewards = numpy.zeros((epochs, 2))
    actions = numpy.zeros((epochs, 2))
    t = time.time()

    state = [Q.sample(3) for Q in QS]
    for e in range(epochs):
        # Sample quantities
        newstate = [QS[i].sample(state[1 - i]) for i in range(2)]

        # Run env
        px, r = env(newstate)

        # Remember
        [
            QS[i].memory.append(state[1 - i], newstate[i], r[i], newstate[1 - i])
            for i in range(2)
        ]

        # Update Qs
        [QS[i].train() for i in range(2)]

        # Update state
        state = newstate

        # Log
        rewards[e] = r
        actions[e] = state
        if not (e + 1) % pfreq:
            print(
                "Epoch:{} \t Time:{:0.2f}min \t Eps1:{:0.3f} \t Eps2:{:0.3f} \t R1:{:0.2f} \t R2:{:0.2f} \t A1:{:0.2f} \t A2:{:0.2f}".format(
                    e + 1,
                    (time.time() - t) / 60,
                    QS[0].epsilon,
                    QS[0].epsilon,
                    rewards[e - pfreq + 1 : e + 1, 0].mean(),
                    rewards[e - pfreq + 1 : e + 1, 1].mean(),
                    actions[e - pfreq + 1 : e + 1, 0].mean(),
                    actions[e - pfreq + 1 : e + 1, 1].mean(),
                )
            )

    df = pandas.DataFrame(rewards)
    df.sum(axis=1).ewm(halflife=1000).mean().plot()
    plt.show()


def qtable2():
    epochs = 5000000
    alpha = [0.1, 0.1]
    gamma = [0.8, 0.8]
    eps_end = [0.001, 0.001]
    epsilon = [1, 1]
    nactions = 50
    action_range = [0, 5]
    eps_step = 1 - 5 / epochs
    env = Cournot()
    QS = [
        QTable(
            alpha=alpha[i],
            gamma=gamma[i],
            eps_end=eps_end[i],
            epsilon=epsilon[i],
            actions=nactions,
            states=nactions,
            eps_step=eps_step,
            action_range=action_range,
            min_memory=10,
            capacity=10,
        )
        for i in range(2)
    ]

    pfreq = epochs // 20
    rewards = numpy.zeros((epochs, 2))
    actions = numpy.zeros((epochs, 2))
    t = time.time()

    state = [Q.sample(3) for Q in QS]
    for e in range(epochs):
        # Sample quantities
        newstate = [QS[i].sample(state[1 - i]) for i in range(2)]

        # Run env
        px, r = env(newstate)

        # Remember
        [
            QS[i].memory.append(state[1 - i], newstate[i], r[i], newstate[1 - i])
            for i in range(2)
        ]

        # Update Qs
        [QS[i].train() for i in range(2)]

        # Update state
        state = newstate

        # Log
        rewards[e] = r
        actions[e] = state
        if not (e + 1) % pfreq:
            print(
                "Epoch:{} \t Time:{:0.2f}min \t Eps1:{:0.3f} \t Eps2:{:0.3f} \t R1:{:0.2f} \t R2:{:0.2f} \t A1:{:0.2f} \t A2:{:0.2f}".format(
                    e + 1,
                    (time.time() - t) / 60,
                    QS[0].epsilon,
                    QS[0].epsilon,
                    rewards[e - pfreq + 1 : e + 1, 0].mean(),
                    rewards[e - pfreq + 1 : e + 1, 1].mean(),
                    actions[e - pfreq + 1 : e + 1, 0].mean(),
                    actions[e - pfreq + 1 : e + 1, 1].mean(),
                )
            )

    df = pandas.DataFrame(rewards).sum(axis=1).to_frame()
    df["Nash"] = 22.2
    df["Collusion"] = 25
    df.columns = ["Reward", "Nash", "Collusion"]

    f, ax = plt.subplots(1, 1)
    df.ewm(halflife=10000).mean().plot(ax=ax)
    ax.set_ylim([18, 25])
    ax.grid()
    plt.show()


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) > 0:
        eval(args[0])()
