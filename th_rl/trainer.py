import numpy
import random
import time
import os
import json
import torch
import pandas

from th_rl.environments import *
from th_rl.agents import *


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


def train_one(exp_path, configpath, loadonly=False, print_eps=False):
    # Handle home location
    if not os.path.exists(exp_path):
        os.mkdir(os.path.join(exp_path))

    config, agents, environment = create_game(configpath)

    # Init Logs
    epochs = config.get("training", {}).get("epochs", 0)
    max_steps = config.get("environment", {}).get("max_steps", 0)
    print_freq = config.get("training", {}).get("print_freq", 500)
    rewards_log = numpy.zeros((epochs, len(agents)))
    actions_log = numpy.zeros((epochs, len(agents)))

    # Train
    t = time.time()
    state = environment.reset()
    for e in range(epochs):
        # Play trajectory
        done = False
        environment.episode = 0
        while not done:
            # choose actions
            acts = [
                agent.sample_action(torch.from_numpy(state.astype("float32")))
                for agent in agents
            ]
            scaled_acts = [agent.scale(act) for agent, act in zip(agents, acts)]
            # Step through environment
            next_state, reward, done = environment.step(scaled_acts)

            # save transition to the replay memory
            for agent, r, action in zip(agents, reward, acts):
                agent.memory.append(state, action, r, not done, next_state)

            # Log
            rewards_log[e, :] += numpy.array(reward) / max_steps
            actions_log[e, :] += numpy.array(scaled_acts) / max_steps
            state = next_state

        # Train
        [A.train_net() for A in agents]

        # Log progress
        if not (e + 1) % print_freq:
            rew = numpy.mean(rewards_log[e - print_freq + 1 : e + 1, :], axis=0)
            act = numpy.mean(actions_log[e - print_freq + 1 : e + 1, :], axis=0)
            if print_eps:
                print(
                    "eps:{} | time:{:2.2f} | episode:{:3d} | reward:{} | agents:{} | actions:{}".format(
                        numpy.round(numpy.array([a.epsilon for a in agents]) * 1000)
                        / 1000,
                        time.time() - t,
                        e,
                        numpy.round(100 * rew) / 100,
                        ",".join([a["name"] for a in config["agents"]]),
                        numpy.round(100 * act) / 100,
                    )
                )
            else:
                print(
                    "time:{:2.2f} | episode:{:3d} | reward:{} | agents:{} | actions:{}".format(
                        time.time() - t,
                        e,
                        numpy.round(100 * rew) / 100,
                        ",".join([a["name"] for a in config["agents"]]),
                        numpy.round(100 * act) / 100,
                    )
                )
            t = time.time()

    # Store result
    for i, a in enumerate(agents):
        a.save(os.path.join(exp_path, str(i)))

    with open(os.path.join(exp_path, "config.json"), "w") as f:
        json.dump(config, f, indent=3)

    rpd = pandas.DataFrame(data=rewards_log, columns=numpy.arange(len(agents)))
    apd = pandas.DataFrame(data=actions_log, columns=numpy.arange(len(agents)))
    log = pandas.concat([rpd, apd], axis=1, keys=["rewards", "actions"])
    log.to_csv(os.path.join(exp_path, "log.csv"), index=None)
