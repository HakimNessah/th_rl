from th_rl.agents import QTable, CAC, ActorCritic, DQN, PPO, Exp3, Reinforce
from th_rl.environments import Cournot
import sys
import numpy
import time
import matplotlib.pyplot as plt
import pandas

env = Cournot()

def plot(rlog, alog, labels=['Agent 1', 'Agent 2'],smooth = 1000):
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
    plt.show()

def qtable():
    epochs = 500000
    alpha = 0.1
    gamma = 0.95
    eps_end = 0.001
    epsilon = 1
    nstates = 40
    nactions = 20
    action_range = [[2, 4],[2,4]]
    state_range = [2, 6]
    eps_step = 1 - 10 / epochs
    env = Cournot()
    QS = [
        QTable(
            alpha=alpha,
            gamma=gamma,
            eps_end=eps_end,
            epsilon=epsilon,
            actions=nactions,
            states=nstates,
            eps_step=eps_step,
            action_range=action_range[i],
            state_range=state_range,
            min_memory=100,
            capacity=100,
        )
        for i in range(2)
    ]

    pfreq = epochs // 20
    rlog = numpy.zeros((epochs, 2))
    alog = numpy.zeros((epochs, 2))
    t = time.time()

    state = numpy.random.rand()*10
    for e in range(epochs):
        # Sample quantities
        actions = [QS[i].sample(state) for i in range(2)]

        # Run env
        px, rewards = env(actions)

        # Remember
        [
            QS[i].memory.append(state, actions[i], rewards[i], px)
            for i in range(2)
        ]

        # Update Qs
        [QS[i].train() for i in range(2)]

        # Update state
        state = px

        # Log
        rlog[e] = rewards
        alog[e] = actions
        if not (e + 1) % pfreq:
            print(
                "Epoch:{} \t Time:{:0.2f}min \t Eps1:{:0.3f} \t Eps2:{:0.3f} \t R1:{:0.2f} \t R2:{:0.2f} \t A1:{:0.2f} \t A2:{:0.2f}".format(
                    e + 1,
                    (time.time() - t) / 60,
                    QS[0].epsilon,
                    QS[0].epsilon,
                    rlog[e - pfreq + 1 : e + 1, 0].mean(),
                    rlog[e - pfreq + 1 : e + 1, 1].mean(),
                    alog[e - pfreq + 1 : e + 1, 0].mean(),
                    alog[e - pfreq + 1 : e + 1, 1].mean(),
                )
            )
    plot(rlog, alog,[type(Q).__name__ for Q in QS], smooth=1000)

def dqn():
    epochs = 500000
    alpha = 0.1
    gamma = 0.95
    eps_end = 0.001
    epsilon = 1
    nactions = 100
    nstates = 1
    action_range = [[0, 10],[0,10]]
    state_range = [0, 10]
    eps_step = 1 - 10 / epochs
    env = Cournot()
    QS = [
        DQN(
            alpha=alpha,
            gamma=gamma,
            eps_end=eps_end,
            epsilon=epsilon,
            actions=nactions,
            states=nstates,
            eps_step=eps_step,
            action_range=action_range[i],
            state_range=state_range,
            min_memory=1-00,
            capacity=100,
        )
        for i in range(2)
    ]

    pfreq = epochs // 20
    rlog = numpy.zeros((epochs, 2))
    alog = numpy.zeros((epochs, 2))
    t = time.time()

    state = numpy.random.rand()*10
    for e in range(epochs):
        # Sample quantities
        actions = [QS[i].sample(state) for i in range(2)]

        # Run env
        px, rewards = env(actions)

        # Remember
        [
            QS[i].memory.append(state, actions[i], rewards[i], px)
            for i in range(2)
        ]

        # Update Qs
        [QS[i].train() for i in range(2)]

        # Update state
        state = px

        # Log
        rlog[e] = rewards
        alog[e] = actions
        if not (e + 1) % pfreq:
            print(
                "Epoch:{} \t Time:{:0.2f}min \t Eps1:{:0.3f} \t Eps2:{:0.3f} \t R1:{:0.2f} \t R2:{:0.2f} \t A1:{:0.2f} \t A2:{:0.2f}".format(
                    e + 1,
                    (time.time() - t) / 60,
                    QS[0].epsilon,
                    QS[0].epsilon,
                    rlog[e - pfreq + 1 : e + 1, 0].mean(),
                    rlog[e - pfreq + 1 : e + 1, 1].mean(),
                    alog[e - pfreq + 1 : e + 1, 0].mean(),
                    alog[e - pfreq + 1 : e + 1, 1].mean(),
                )
            )
    plot(rlog, alog,[type(Q).__name__ for Q in QS], smooth=1000)

def reinforce_static():
    epochs = 200000
    gamma = 0.95
    nactions = 100
    nstates = 1
    action_range = [[0, 10],[0,0.5]]
    entropy = [0,0.]
    env = Cournot()
    QS = [
        Reinforce(
            gamma=gamma,
            actions=nactions,
            states=nstates,
            action_range=action_range[i],
            min_memory=100,
            capacity=100,
        )
        for i in range(2)
    ]

    pfreq = epochs // 20
    rlog = numpy.zeros((epochs, 2))
    alog = numpy.zeros((epochs, 2))
    t = time.time()

    state = numpy.random.rand()*10
    for e in range(epochs):
        # Sample quantities
        samples = [QS[i].sample(state) for i in range(2)]
        actions = [s[0] for s in samples]
        logprobs = [s[1] for s in samples]

        # Run env
        px, rewards = env(actions)

        # Remember
        [
            QS[i].memory.append(state, actions[i], rewards[i], logprobs[i])
            for i in range(2)
        ]

        # Update Qs
        [QS[i].train() for i in range(2)]

        # Update state
        state = px

        # Log
        rlog[e] = rewards
        alog[e] = actions
        if not (e + 1) % pfreq:
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
    
    plot(rlog, alog,[type(Q).__name__ for Q in QS], smooth=1000)

def actorcritic_static():
    epochs = 200000
    gamma = 0.95
    nactions = 100
    nstates = 1
    action_range = [[0, 10],[3.3,3.35]]
    env = Cournot()
    QS = [
        ActorCritic(
            gamma=gamma,
            actions=nactions,
            states=nstates,
            action_range=action_range[i],
            min_memory=50,
            capacity=50,
            name=i
        )
        for i in range(2)
    ]

    pfreq = epochs // 20
    rlog = numpy.zeros((epochs, 2))
    alog = numpy.zeros((epochs, 2))
    t = time.time()

    state = numpy.random.rand()*10
    for e in range(epochs):
        # Sample quantities
        samples = [QS[i].sample(state) for i in range(2)]
        actions = [s[0] for s in samples]
        logprobs = [s[1] for s in samples]

        # Run env
        px, rewards = env(actions)

        # Remember
        [
            QS[i].memory.append(state, logprobs[i], rewards[i], px)
            for i in range(2)
        ]

        # Update Qs
        [QS[i].train() for i in range(2)]

        # Update state
        state = px

        # Log
        rlog[e] = rewards
        alog[e] = actions
        if not (e + 1) % pfreq:
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
    
    plot(rlog, alog,[type(Q).__name__ for Q in QS], smooth=1000)

def cac():
    epochs = 1000000
    gamma = 0.95
    action_range = [[2,4],[2,4]]
    state_range = [2,6]
    entropy = [0.001,0.001]
    env = Cournot()
    QS = [
        CAC(
            gamma=gamma,
            action_range=action_range[i],
            state_range=state_range,
            min_memory=20,
            capacity=20,
            entropy=entropy[i]
        )
        for i in range(2)
    ]

    pfreq = epochs // 20
    rlog = numpy.zeros((epochs, 2))
    alog = numpy.zeros((epochs, 2))
    t = time.time()

    state = numpy.random.rand()*10
    for e in range(epochs):
        # Sample quantities
        actions = [QS[i].sample(state) for i in range(2)]

        # Run env
        px, rewards = env(actions)

        # Remember
        [
            QS[i].memory.append(state, actions[i], rewards[i], px)
            for i in range(2)
        ]

        # Update Qs
        [QS[i].train() for i in range(2)]

        # Update state
        state = px

        # Log
        rlog[e] = rewards
        alog[e] = actions
        if not (e + 1) % pfreq:
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
    
    plot(rlog, alog,[type(Q).__name__ for Q in QS], smooth=1000)

def ppo():
    epochs = 100000
    gamma = 0.95
    action_range = [[0, 10],[3.3,3.35]]
    state_range = [0, 10]
    entropy = [0,0.]
    env = Cournot()
    QS = [
        PPO(
            gamma=gamma,
            action_range=action_range[i],
            state_range=state_range,
            min_memory=10000,
            capacity=10000,
            batch=100,
            entropy=entropy[i]
        )
        for i in range(2)
    ]

    pfreq = epochs // 20
    rlog = numpy.zeros((epochs, 2))
    alog = numpy.zeros((epochs, 2))
    t = time.time()

    state = numpy.random.rand()*10
    for e in range(epochs):
        # Sample quantities
        samples = [QS[i].sample(state) for i in range(2)]
        actions = [a[0] for a in samples]
        probs = [a[1] for a in samples]

        # Run env

        px, rewards = env(actions)

        # Remember
        [
            QS[i].memory.append(state, actions[i], rewards[i], px, probs[i])
            for i in range(2)
        ]

        # Update Qs
        [QS[i].train() for i in range(2)]

        # Update state
        state = px

        # Log
        rlog[e] = rewards
        alog[e] = actions
        if not (e + 1) % pfreq:
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
    
    plot(rlog, alog,[type(Q).__name__ for Q in QS], smooth=1000)

def exp3():
    epochs = 2000000
    gamma = 0.99
    nactions = 20
    action_range = [[2, 4],[2,4]]
    state_range = [2, 6]
    env = Cournot()
    QS = [
        Exp3(
            gamma=gamma,
            eta= 0.0001,
            actions=nactions,
            action_range=action_range[0],
            state_range=state_range,
            min_memory=200,
            capacity=200,
        )
        for i in range(2)
    ]

    pfreq = epochs // 20
    rlog = numpy.zeros((epochs, 2))
    alog = numpy.zeros((epochs, 2))
    t = time.time()

    state = numpy.random.rand()*10
    for e in range(epochs):
        # Sample quantities
        actions = [QS[i].sample() for i in range(2)]

        # Run env
        px, rewards = env(actions)

        # Remember
        [
            QS[i].memory.append(actions[i], rewards[i])
            for i in range(2)
        ]

        # Update Qs
        [QS[i].train() for i in range(2)]

        # Update state
        state = px

        # Log
        rlog[e] = rewards
        alog[e] = actions
        if not (e + 1) % pfreq:
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
    plot(rlog, alog,[type(Q).__name__ for Q in QS], smooth=1000)

def qtable_exp3():
    epochs = 1000000
    nactions = 20
    alpha = 0.1
    action_range = [2,4]
    state_range = [2, 6]
    eps_end = 0.001
    epsilon = 1
    nstates = 40
    eps_step = 1 - 10 / epochs

    env = Cournot()
    QS = [       
        QTable(
            alpha=alpha,
            gamma=0.95,
            eps_end=eps_end,
            epsilon=epsilon,
            actions=nactions,
            states=nstates,
            eps_step=eps_step,
            action_range=action_range,
            state_range=state_range,
            min_memory=100,
            capacity=100,
        ),
        Exp3(
            gamma=0.999,
            eta= 0.0001,
            actions=nactions,
            action_range=action_range,
            state_range=state_range,
            min_memory=1000,
            capacity=1000,
        )
    ]

    pfreq = epochs // 20
    rlog = numpy.zeros((epochs, 2))
    alog = numpy.zeros((epochs, 2))
    t = time.time()

    state = numpy.random.rand()*10
    for e in range(epochs):
        # Sample quantities
        actions = [QS[0].sample(state), QS[1].sample()]

        # Run env
        px, rewards = env(actions)

        # Remember
        QS[0].memory.append(state, actions[0], rewards[0], px)
        QS[1].memory.append(actions[1], rewards[1])
        

        # Update Qs
        [QS[i].train() for i in range(2)]

        # Update state
        state = px

        # Log
        rlog[e] = rewards
        alog[e] = actions
        if not (e + 1) % pfreq:
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
    plot(rlog, alog,[type(Q).__name__ for Q in QS], smooth=1000)

def qtable_reinforce():
    epochs = 100000
    alpha = 0.1
    gamma = 0.95
    eps_end = 0.001
    epsilon = 1
    nactions = 100
    nstates = 40
    action_range = [2,4]
    state_range = [2, 6]
    eps_step = 1 - 10 / epochs
    env = Cournot()
    QS = [
        QTable(
            alpha=alpha,
            gamma=gamma,
            eps_end=eps_end,
            epsilon=epsilon,
            actions=nactions,
            states=nstates,
            eps_step=eps_step,
            action_range=action_range,
            state_range=state_range,
            min_memory=100,
            capacity=100,
        ),        
        Reinforce(
            gamma=gamma,
            actions=nactions,
            states=1,
            eps_step=eps_step,
            action_range=action_range,
            min_memory=100,
            capacity=100,
        )
    ]

    pfreq = epochs // 20
    rlog = numpy.zeros((epochs, 2))
    alog = numpy.zeros((epochs, 2))
    t = time.time()

    state = numpy.random.rand()*10
    for e in range(epochs):
        # Sample quantities
        actionq,_ = QS[0].sample(state)
        actionr, logprob = QS[1].sample(state)

        actions = [actionq, actionr]

        # Run env
        px, rewards = env(actions)

        # Remember
        QS[0].memory.append(state, actionq, rewards[0], px)
        QS[1].memory.append(rewards[1], logprob)        

        # Update Qs
        [QS[i].train() for i in range(2)]

        # Update state
        state = px

        # Log
        rlog[e] = rewards
        alog[e] = actions
        if not (e + 1) % pfreq:
            print(
                "Epoch:{} \t Time:{:0.2f}min \t Eps:{:0.3f} \t R1:{:0.2f} \t R2:{:0.2f} \t A1:{:0.2f} \t A2:{:0.2f}".format(
                    e + 1,
                    (time.time() - t) / 60,
                    QS[0].epsilon,
                    rlog[e - pfreq + 1 : e + 1, 0].mean(),
                    rlog[e - pfreq + 1 : e + 1, 1].mean(),
                    alog[e - pfreq + 1 : e + 1, 0].mean(),
                    alog[e - pfreq + 1 : e + 1, 1].mean(),
                )
            )
    plot(rlog, alog,[type(Q).__name__ for Q in QS], smooth=1000)

def qtable_dqn():
    epochs = 500000
    alpha = 0.1
    gamma = 0.95
    eps_end = 0.001
    epsilon = 1
    nactions = 100
    nstates = 1
    action_range = [2,4]
    state_range = [2, 6]
    eps_step = 1 - 10 / epochs
    env = Cournot()
    QS = [
        QTable(
            alpha=alpha,
            gamma=gamma,
            eps_end=eps_end,
            epsilon=epsilon,
            actions=nactions,
            states=nstates,
            eps_step=eps_step,
            action_range=action_range,
            state_range=state_range,
            min_memory=100,
            capacity=100,
        ),        
        DQN(
            alpha=alpha,
            gamma=gamma,
            eps_end=eps_end,
            epsilon=epsilon,
            actions=nactions,
            states=nstates,
            eps_step=eps_step,
            action_range=action_range,
            state_range=state_range,
            min_memory=100,
            capacity=100,
        )
    ]

    pfreq = epochs // 20
    rlog = numpy.zeros((epochs, 2))
    alog = numpy.zeros((epochs, 2))
    t = time.time()

    state = numpy.random.rand()*10
    for e in range(epochs):
        # Sample quantities
        actions = [QS[i].sample(state) for i in range(2)]

        # Run env
        px, rewards = env(actions)

        # Remember
        [
            QS[i].memory.append(state, actions[i], rewards[i], px)
            for i in range(2)
        ]

        # Update Qs
        [QS[i].train() for i in range(2)]

        # Update state
        state = px

        # Log
        rlog[e] = rewards
        alog[e] = actions
        if not (e + 1) % pfreq:
            print(
                "Epoch:{} \t Time:{:0.2f}min \t Eps1:{:0.3f} \t Eps2:{:0.3f} \t R1:{:0.2f} \t R2:{:0.2f} \t A1:{:0.2f} \t A2:{:0.2f}".format(
                    e + 1,
                    (time.time() - t) / 60,
                    QS[0].epsilon,
                    QS[0].epsilon,
                    rlog[e - pfreq + 1 : e + 1, 0].mean(),
                    rlog[e - pfreq + 1 : e + 1, 1].mean(),
                    alog[e - pfreq + 1 : e + 1, 0].mean(),
                    alog[e - pfreq + 1 : e + 1, 1].mean(),
                )
            )
    plot(rlog, alog,[type(Q).__name__ for Q in QS], smooth=1000)

def exp3_cac():
    epochs = 500000
    gamma = 0.99
    nactions = 20
    action_range = [[2, 4],[2,4]]
    state_range = [2, 6]

    env = Cournot()
    QS = [
        Exp3(
            gamma=gamma,
            eta= 0.0001,
            actions=nactions,
            action_range=action_range[0],
            state_range=state_range,
            min_memory=1000,
            capacity=1000,
        ),
        CAC(
            gamma=gamma,
            action_range=action_range[1],
            state_range=state_range,
            min_memory=100,
            capacity=100,
            entropy=0
        )
    ]

    pfreq = epochs // 20
    rlog = numpy.zeros((epochs, 2))
    alog = numpy.zeros((epochs, 2))
    t = time.time()

    state = numpy.random.rand()*10
    for e in range(epochs):
        # Sample quantities
        actions = [QS[0].sample(), QS[1].sample(state)]

        # Run env
        px, rewards = env(actions)

        # Remember
        QS[0].memory.append(actions[0], rewards[0])
        QS[1].memory.append(state, actions[1], rewards[1], px)

        # Update Qs
        [QS[i].train() for i in range(2)]

        # Update state
        state = px

        # Log
        rlog[e] = rewards
        alog[e] = actions
        if not (e + 1) % pfreq:
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

    plot(rlog, alog,[type(Q).__name__ for Q in QS], smooth=1000)

def qtable_cac():
    epochs = 1000000
    alpha = 0.1
    gamma = 0.99
    eps_end = 0.001
    epsilon = 1
    nactions = 20
    nstates = 40
    action_range = [2,4]
    state_range = [2,6]
    eps_step = 1 - 10 / epochs
    env = Cournot()
    QS = [
        QTable(
            alpha=alpha,
            gamma=gamma,
            eps_end=eps_end,
            epsilon=epsilon,
            actions=nactions,
            states=nstates,
            eps_step=eps_step,
            action_range=action_range,
            state_range=state_range,
            min_memory=100,
            capacity=100,
        ),
        CAC(
            gamma=gamma,
            action_range=action_range,
            state_range=state_range,
            min_memory=100,
            capacity=100,
            entropy=0
        )
    ]

    pfreq = epochs // 20
    rlog = numpy.zeros((epochs, 2))
    alog = numpy.zeros((epochs, 2))
    t = time.time()

    state = numpy.random.rand()*10
    for e in range(epochs):
        # Sample quantities
        actions = [QS[i].sample(state) for i in range(2)]

        # Run env
        px, rewards = env(actions)

        # Remember
        [
            QS[i].memory.append(state, actions[i], rewards[i], px)
            for i in range(2)
        ]

        # Update Qs
        [QS[i].train() for i in range(2)]

        # Update state
        state = px

        # Log
        rlog[e] = rewards
        alog[e] = actions
        if not (e + 1) % pfreq:
            print(
                "Epoch:{} \t Time:{:0.2f}min \t Eps1:{:0.3f} \t Eps2:{:0.3f} \t R1:{:0.2f} \t R2:{:0.2f} \t A1:{:0.2f} \t A2:{:0.2f}".format(
                    e + 1,
                    (time.time() - t) / 60,
                    QS[0].epsilon,
                    QS[0].epsilon,
                    rlog[e - pfreq + 1 : e + 1, 0].mean(),
                    rlog[e - pfreq + 1 : e + 1, 1].mean(),
                    alog[e - pfreq + 1 : e + 1, 0].mean(),
                    alog[e - pfreq + 1 : e + 1, 1].mean(),
                )
            )
    plot(rlog, alog,[type(Q).__name__ for Q in QS], smooth=1000)

if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) > 0:
        eval(args[0])()
