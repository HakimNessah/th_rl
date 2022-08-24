import numpy
import random
import time
import os
import json
import torch
import pandas

from th_rl.environments import *
from th_rl.agents import *
from collections import deque


def create_game(configpath):
    # Load config
    config = json.load(open(configpath))
    # Create environment
    environment = eval(config["environment"]["name"])(**config["environment"])

    # Create agents
    agents = [
        eval(agent["name"])(env=environment, **agent) for agent in config["agents"]
    ]

    return config, agents, environment


def stateless(agents, environment, **kwargs):
    epochs = kwargs.get("epochs", 10000)
    print_freq = kwargs.get("print_freq", 500)
    # Init Logs
    max_steps = environment.max_steps
    rewards_log = numpy.zeros((epochs, len(agents)))
    actions_log = numpy.zeros((epochs, len(agents)))

    # Train
    t = time.time()
    state = torch.tensor(1.0)
    for e in range(epochs):
        # Play trajectory
        done = False
        environment.episode = 0
        while not done:
            # choose actions
            acts = [agent.sample_action(state) for agent in agents]
            scaled_acts = [agent.scale(act) for agent, act in zip(agents, acts)]
            demand, reward, done = environment.step(torch.tensor(scaled_acts))

            # save transition to the replay memory
            for agent, r, action in zip(agents, reward, acts):
                agent.memory.append(state, action, r, not done, state)

            # Log
            rewards_log[e, :] += numpy.array(reward) / max_steps
            actions_log[e, :] += numpy.array(scaled_acts) / max_steps
        # Train
        [A.train_net() for A in agents]

        # Log progress
        if not (e + 1) % print_freq:
            rew = numpy.mean(rewards_log[e - print_freq + 1 : e + 1, :], axis=0)
            act = numpy.mean(actions_log[e - print_freq + 1 : e + 1, :], axis=0)
            if "epsilon" in dir(agents[0]):
                print(
                    "eps:{} | time:{:2.2f} | episode:{:3d} | reward:{} | actions:{}".format(
                        numpy.round(numpy.array([a.epsilon for a in agents]) * 1000)
                        / 1000,
                        time.time() - t,
                        e,
                        numpy.round(100 * rew) / 100,
                        numpy.round(100 * act) / 100,
                    )
                )
            else:
                print(
                    "time:{:2.2f} | episode:{:3d} | reward:{} | actions:{}".format(
                        time.time() - t,
                        e,
                        numpy.round(100 * rew) / 100,
                        numpy.round(100 * act) / 100,
                    )
                )
            t = time.time()

    return rewards_log, actions_log


def perfectmonitoring(agents, environment, **kwargs):
    epochs = kwargs.get("epochs", 10000)
    print_freq = kwargs.get("print_freq", 500)
    # Init Logs
    max_steps = environment.max_steps
    rewards_log = numpy.zeros((epochs, len(agents)))
    actions_log = numpy.zeros((epochs, len(agents)))

    # Train
    lag = kwargs.get("lag", 0)
    if lag == 0:
        buffer = deque(maxlen=1)
    else:
        buffer = deque(maxlen=lag * max_steps)
    t = time.time()
    for e in range(epochs):
        # Play trajectory
        done = False
        environment.episode = 0
        state = torch.Tensor([1.0, 1.0])
        while not done:
            # choose actions
            acts = [agent.sample_action(x) for agent, x in zip(agents, state)]
            scaled_acts = [agent.scale(act) for agent, act in zip(agents, acts)]
            demand, reward, done = environment.step(torch.tensor(scaled_acts))

            # save transition to the replay memory
            for s, agent, r, action, next_s in zip(
                state, agents, reward, acts, scaled_acts[::-1]
            ):
                agent.memory.append(s, action, r, not done, next_s)
            buffer.append(scaled_acts)
            if e >= lag:
                state = buffer[0][::-1]

            # Log
            rewards_log[e, :] += numpy.array(reward)
            actions_log[e, :] += numpy.array(scaled_acts)
        # Train
        [A.train_net(False) for A in agents]

        # Log progress
        if not (e + 1) % print_freq:
            rew = (
                numpy.mean(rewards_log[e - print_freq + 1 : e + 1, :], axis=0)
                / max_steps
            )
            act = (
                numpy.mean(actions_log[e - print_freq + 1 : e + 1, :], axis=0)
                / max_steps
            )
            if "epsilon" in dir(agents[0]):
                print(
                    "eps:{} | time:{:2.2f} | episode:{:3d} | reward:{} | actions:{}".format(
                        numpy.round(numpy.array([a.epsilon for a in agents]) * 1000)
                        / 1000,
                        time.time() - t,
                        e,
                        numpy.round(100 * rew) / 100,
                        numpy.round(100 * act) / 100,
                    )
                )
            else:
                print(
                    "time:{:2.2f} | episode:{:3d} | reward:{} | actions:{}".format(
                        time.time() - t,
                        e,
                        numpy.round(100 * rew) / 100,
                        numpy.round(100 * act) / 100,
                    )
                )
            t = time.time()

    return rewards_log / max_steps, actions_log / max_steps


def selfmonitoring(agents, environment, **kwargs):
    epochs = kwargs.get("epochs", 10000)
    print_freq = kwargs.get("print_freq", 500)
    # Init Logs
    max_steps = environment.max_steps
    rewards_log = numpy.zeros((epochs, len(agents)))
    actions_log = numpy.zeros((epochs, len(agents)))

    # Train
    t = time.time()
    lag = kwargs.get("lag", 0)
    if lag == 0:
        buffer = deque(maxlen=1)
    else:
        buffer = deque(maxlen=lag * max_steps)
    for e in range(epochs):
        # Play trajectory
        done = False
        environment.episode = 0
        state = torch.Tensor([1.0, 1.0])
        while not done:
            # choose actions
            acts = [agent.sample_action(x) for agent, x in zip(agents, state)]
            scaled_acts = [agent.scale(act) for agent, act in zip(agents, acts)]
            demand, reward, done = environment.step(torch.tensor(scaled_acts))

            # save transition to the replay memory
            for s, agent, r, action, next_s in zip(
                state, agents, reward, acts, scaled_acts
            ):
                agent.memory.append(s, action, r, not done, next_s)
            buffer.append(scaled_acts)
            if e >= lag:
                state = buffer[0]

            # Log
            rewards_log[e, :] += numpy.array(reward)
            actions_log[e, :] += numpy.array(scaled_acts)
        # Train
        [A.train_net() for A in agents]

        # Log progress
        if not (e + 1) % print_freq:
            rew = (
                numpy.mean(rewards_log[e - print_freq + 1 : e + 1, :], axis=0)
                / max_steps
            )
            act = (
                numpy.mean(actions_log[e - print_freq + 1 : e + 1, :], axis=0)
                / max_steps
            )
            if "epsilon" in dir(agents[0]):
                print(
                    "eps:{} | time:{:2.2f} | episode:{:3d} | reward:{} | actions:{}".format(
                        numpy.round(numpy.array([a.epsilon for a in agents]) * 1000)
                        / 1000,
                        time.time() - t,
                        e,
                        numpy.round(100 * rew) / 100,
                        numpy.round(100 * act) / 100,
                    )
                )
            else:
                print(
                    "time:{:2.2f} | episode:{:3d} | reward:{} | actions:{}".format(
                        time.time() - t,
                        e,
                        numpy.round(100 * rew) / 100,
                        numpy.round(100 * act) / 100,
                    )
                )
            t = time.time()

    return rewards_log / max_steps, actions_log / max_steps


def train_one(exp_path, configpath):
    # Handle home location
    if not os.path.exists(exp_path):
        os.mkdir(os.path.join(exp_path))

    config, agents, environment = create_game(configpath)
    trainfun = eval(config["training"]["name"])

    # Train
    rewards_log, actions_log = trainfun(agents, environment, **config["training"])

    # Store result
    for i, a in enumerate(agents):
        a.save(os.path.join(exp_path, str(i)))

    with open(os.path.join(exp_path, "config.json"), "w") as f:
        json.dump(config, f, indent=3)

    rpd = pandas.DataFrame(data=rewards_log, columns=numpy.arange(len(agents)))
    apd = pandas.DataFrame(data=actions_log, columns=numpy.arange(len(agents)))
    log = pandas.concat([rpd, apd], axis=1, keys=["rewards", "actions"])
    log.to_csv(os.path.join(exp_path, "log.csv"), index=None)
