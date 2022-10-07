from collections import namedtuple
import numpy
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal
from th_rl.buffers import *
from numba import jit


@jit
def q_train(
    gamma, alpha, table, counter, next_state, old_value, rewards, state, actions
):
    for ns, ov, re, st, ac in zip(next_state, old_value, rewards, state, actions):
        next_max = numpy.max(table[ns])
        new_value = (1 - alpha) * ov + alpha * (re + gamma * next_max)
        table[st, ac] = new_value
        counter[st, ac] += 1
    return table, counter


class QTable:
    def __init__(self, **kwargs):
        self.gamma = kwargs.get("gamma", 0.95)
        self.actions = kwargs.get("actions", 20)
        self.epsilon = kwargs.get("epsilon", 1.0)
        self.eps_step = kwargs.get("eps_step", 0.9995)
        self.eps_end = kwargs.get("eps_end", 0.01)
        self.states = kwargs.get("states", 1)
        self.statelen = kwargs.get("statelen", 1)
        self.alpha = kwargs.get("alpha", 0.1)
        self.action_range = kwargs.get("action_range", [0, 1])
        self.punish = kwargs.get("punish", 0)
        self.max_punish = kwargs.get("max_punish", 10)
        self.min_wait = kwargs.get("min_wait", self.max_punish + 1)
        self.wait = 0
        self.table = 10 / (1 - self.gamma) + numpy.random.randn(
            self.states, self.actions
        )
        self.action_space = numpy.arange(0, self.actions)
        self.min_memory = kwargs.get("min_memory", 100)
        self.experience = namedtuple(
            "Experience", field_names=["state", "action", "reward", "done", "new_state"]
        )
        self.memory = eval(kwargs.get("buffer", "ReplayBuffer"))(
            kwargs.get("capacity", 100), self.experience
        )
        self.counter = 0 * self.table

    def encode(self, state):
        state = numpy.reshape(state, [-1, self.statelen])
        out = numpy.zeros(
            state.shape[0],
        )
        for i in range(state.shape[1]):
            out += state[:, i] * self.actions**i
        return out.astype(int)

    def scale(self, action):
        if action == 0:
            return [1, 0]
        scaled = (action - 1) / (self.actions - 2) * (
            self.action_range[1] - self.action_range[0]
        ) + self.action_range[0]
        return [0, scaled]

    def train_net(self):
        if len(self.memory) >= self.min_memory:
            state, acts, rwrd, not_done, next_state = self.memory.replay()
            state = self.encode(numpy.array(state))
            next_state = self.encode(numpy.array(next_state))
            [actions, not_done, rewards] = [
                numpy.reshape(x, [-1]) for x in [acts, not_done, rwrd]
            ]
            old_value = self.table[state, actions]
            for ns, ov, re, st, ac in zip(
                next_state, old_value, rewards, state, actions
            ):
                next_max = numpy.max(self.table[ns])
                new_value = (1 - self.alpha) * ov + self.alpha * (
                    re + self.gamma * next_max
                )
                self.table[st, ac] = new_value
                self.counter[st, ac] += 1

            self.memory.empty()
        self.epsilon = self.eps_end + (self.epsilon - self.eps_end) * self.eps_step

    def sample_action(self, state):
        if self.punish > 1:
            self.punish -= 1
            return 0
        if self.punish == 1:
            self.punish -= 1
            return 1
        if random.uniform(0, 1) < self.epsilon:
            action = random.choice(self.action_space)
        else:
            if isinstance(state, torch.Tensor):
                st = state.numpy()
            else:
                st = state
            action = numpy.argmax(self.table[self.encode(st)])
        if action == 0:
            if self.wait < self.min_wait:
                action = 1
                self.wait += 1
            else:
                self.punish = self.max_punish
                self.wait = 0
        else:
            self.wait += 1
        return action

    def get_action(self, state):
        if self.punish:
            action = 0
            self.punish -= 1
        action = numpy.argmax(self.table[self.encode(state)])
        if action == 0:
            self.punish = self.max_punish
        return action

    def reset(self, eps_end):
        self.table = 10 / (1 - self.gamma) + numpy.random.randn(
            self.states, self.actions
        )
        self.epsilon = 1.0
        self.eps_end = eps_end

    def reset_value(self, eps_end):
        self.table = 10 / (1 - self.gamma) + numpy.random.randn(
            self.states, self.actions
        )  # numpy.zeros([states, actions])#

    def reset_pi(self, eps_end):
        self.epsilon = 1.0
        self.eps_end = eps_end

    def save(self, loc):
        numpy.save(loc, self.table)
        numpy.save(loc + "_counter", self.counter)

    def load(self, loc):
        self.table = numpy.load(loc + ".npy")
        self.counter = numpy.load(loc + "_counter.npy")


class ActorCritic(nn.Module):
    def __init__(
        self,
        states=4,
        actions=2,
        action_range=[0, 1],
        gamma=0.98,
        buffer="ReplayBuffer",
        capacity=50000,
        min_memory=1000,
        entropy=0,
        **kwargs
    ):
        super(ActorCritic, self).__init__()
        self.data = []
        self.gamma = gamma
        self.action_range = action_range
        self.actions = actions
        self.fc1 = nn.Linear(states, 256)
        self.fc_pi = nn.Linear(256, actions)
        self.fc_v = nn.Linear(256, 1)
        self.fc_v.bias.data.fill_(1000.0)
        self.optimizer = optim.Adam(self.parameters(), lr=2e-4)
        self.experience = namedtuple(
            "Experience", field_names=["state", "action", "reward", "done", "new_state"]
        )
        self.cast = [torch.float, torch.int64, torch.float, torch.float, torch.float]
        self.memory = eval(buffer)(capacity, self.experience)
        self.min_memory = min_memory
        self.entropy = entropy

    def pi(self, x, softmax_dim=0):  # Pi=policy-> Actor
        x = torch.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def v(self, x):  # v = Value -> Critic
        y = torch.relu(self.fc1(x))
        v = self.fc_v(y)
        return v

    def scale(self, action):
        return (
            action / self.actions * (self.action_range[1] - self.action_range[0])
            + self.action_range[0]
        )

    def sample_action(self, state):
        prob = self.pi(state)
        m = Categorical(prob)
        return m.sample().item()

    def get_action(self, state):
        pi = self.pi(torch.tensor(state.astype("float32")))
        m = torch.argmax(pi)
        return m.item()

    def train_net(self):
        if len(self.memory) >= self.min_memory:
            states, actions, rewards, done, s_prime = self.memory.replay(self.cast)
            [actions, done, rewards] = [
                torch.reshape(x, [-1]) for x in [actions, done, rewards]
            ]
            pi = self.pi(states, softmax_dim=1)
            v = self.v(states)
            v_prime = self.v(s_prime)

            advantage = (rewards + self.gamma * v_prime) - v
            critic_loss = advantage**2  # F.smooth_l1_loss(v, advantage.detach())

            dist = Categorical(probs=pi)

            actor_loss = (
                -dist.log_prob(actions) * advantage.detach()
            )  # delta[:,0].detach()
            entropy = -torch.mean(dist.entropy())

            loss = torch.mean(critic_loss + actor_loss) + self.entropy * entropy
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            self.optimizer.step()
            self.memory.empty()

    def reset(self, entropy):
        self.memory.empty()
        self.entropy = entropy
        for layer in self.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
        self.fc_v.bias.data.fill_(1000.0)

    def reset_value(self, entropy):
        self.memory.empty()
        self.entropy = entropy
        self.fc_v.reset_parameters()
        self.fc_v.bias.data.fill_(1000.0)

    def reset_pi(self, entropy):
        self.memory.empty()
        self.entropy = entropy
        self.fc_pi.reset_parameters()

    def save(self, loc):
        torch.save(self.state_dict(), loc)

    def load(self, loc):
        self.load_state_dict(torch.load(loc))


class Const:
    def __init__(
        self,
        states=4,
        action_range=[0, 1],
        gamma=0.98,
        buffer="ReplayBuffer",
        capacity=50000,
        min_memory=1000,
        **kwargs
    ):
        self.action = kwargs.get("action")
        self.data = []
        self.action_range = action_range
        self.experience = namedtuple(
            "Experience", field_names=["state", "action", "reward", "done", "new_state"]
        )
        self.cast = [torch.float, torch.float, torch.float, torch.float, torch.float]
        self.memory = eval(buffer)(capacity, self.experience)

    def scale(self, action):
        return (
            action * (self.action_range[1] - self.action_range[0])
            + self.action_range[0]
        )

    def sample_action(self, state):
        return self.action

    def get_action(self, state):
        return self.action

    def train_net(self, dobreak):
        return None


class CAC(nn.Module):
    def __init__(
        self,
        states=4,
        action_range=[0, 1],
        gamma=0.98,
        buffer="ReplayBuffer",
        capacity=50000,
        min_memory=1000,
        entropy=0,
        **kwargs
    ):
        super(CAC, self).__init__()
        self.data = []
        self.gamma = gamma
        self.action_range = action_range
        self.fc1 = nn.Linear(states, 256)
        self.fc_mu = nn.Linear(256, 1)
        self.fc_std = nn.Linear(256, 1)
        self.fc_v = nn.Linear(256, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=2e-4)
        self.experience = namedtuple(
            "Experience", field_names=["state", "action", "reward", "done", "new_state"]
        )
        self.cast = [torch.float, torch.float, torch.float, torch.float, torch.float]
        self.memory = eval(buffer)(capacity, self.experience)
        self.min_memory = min_memory
        self.entropy = entropy

    def pi(self, x):  # Pi=policy-> Actor
        x = torch.relu(self.fc1(x))
        mu = 4.0 * torch.tanh(self.fc_mu(x))
        std = F.softplus(self.fc_std(x))
        return mu, std

    def v(self, x):  # v = Value -> Critic
        y = torch.relu(self.fc1(x))
        v = self.fc_v(y)
        return v

    def scale(self, action):
        return (
            action * (self.action_range[1] - self.action_range[0])
            + self.action_range[0]
        )

    def sample_action(self, state):
        mu, std = self.pi(torch.tensor([state]))
        dist = Normal(mu, std)
        action = dist.sample()
        return torch.sigmoid(action).item()

    def get_action(self, state):
        mu, _ = self.pi(torch.from_numpy(state).float())
        dist = Normal(mu, 1e-5)
        action = dist.sample()
        return torch.sigmoid(action).item()

    def train_net(self, dobreak):
        if len(self.memory) >= self.min_memory:
            states, actions, rewards, done, s_prime = self.memory.replay(self.cast)
            [actions, done, rewards, states, s_prime] = [
                torch.reshape(x, [-1, 1])
                for x in [actions, done, rewards, states, s_prime]
            ]
            mu, std = self.pi(states)
            v = self.v(states)

            v_prime = self.v(s_prime)

            advantage = rewards + self.gamma * v_prime - v
            critic_loss = advantage**2

            dist = Normal(mu, std)

            _actions = 1e-6 + (1 - 1e-5) * actions
            logits = torch.log(_actions / (1 - _actions))
            actor_loss = -dist.log_prob(logits) * advantage.detach()
            entropy = -torch.mean(dist.entropy())

            if dobreak:
                breakpoint()
            loss = torch.mean(critic_loss + actor_loss) + self.entropy * entropy
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            self.optimizer.step()
            self.memory.empty()

    def reset(self, entropy):
        self.memory.empty()
        self.entropy = entropy
        for layer in self.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
        self.fc_v.bias.data.fill_(1000.0)

    def reset_value(self, entropy):
        self.memory.empty()
        self.entropy = entropy
        self.fc_v.reset_parameters()
        self.fc_v.bias.data.fill_(1000.0)

    def reset_pi(self, entropy):
        self.memory.empty()
        self.entropy = entropy
        self.fc_pi.reset_parameters()

    def save(self, loc):
        torch.save(self.state_dict(), loc)

    def load(self, loc):
        self.load_state_dict(torch.load(loc))


class ModelCAC(nn.Module):
    def __init__(
        self,
        env,
        states=4,
        action_range=[0, 1],
        gamma=0.98,
        buffer="ReplayBuffer",
        capacity=50000,
        min_memory=1000,
        entropy=0,
        replays=10,
        **kwargs
    ):
        super(ModelCAC, self).__init__()
        self.data = []
        self.gamma = gamma
        self.action_range = action_range
        self.fc1 = nn.Linear(states, 256)
        self.fc_mu = nn.Linear(256, 1)
        self.fc_std = nn.Linear(256, 1)
        self.fc_v = nn.Linear(256, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=2e-4)
        self.experience = namedtuple(
            "Experience", field_names=["state", "action", "reward", "done", "new_state"]
        )
        self.cast = [torch.float, torch.float, torch.float, torch.float, torch.float]
        self.memory = eval(buffer)(capacity, self.experience)
        self.experience = eval(buffer)(replays * capacity, self.experience)
        self.min_memory = min_memory
        self.replays = replays
        self.entropy = entropy
        self.env = env

    def pi(self, x):  # Pi=policy-> Actor
        x = torch.relu(self.fc1(x))
        mu = 4.0 * torch.tanh(self.fc_mu(x))
        std = F.softplus(self.fc_std(x))
        return mu, std

    def v(self, x):  # v = Value -> Critic
        y = torch.relu(self.fc1(x))
        v = self.fc_v(y)
        return v

    def scale(self, action):
        return (
            action * (self.action_range[1] - self.action_range[0])
            + self.action_range[0]
        )

    def sample_action(self, state):
        mu, std = self.pi(torch.tensor([state]))
        dist = Normal(mu, std)
        action = dist.sample()
        return torch.sigmoid(action).item()

    def get_action(self, state):
        mu, _ = self.pi(torch.from_numpy(state).float())
        dist = Normal(mu, 1e-5)
        action = dist.sample()
        return torch.sigmoid(action).item()

    def generate_experience(self):
        states, actions, rewards, dones, s_primes = self.memory.replay(self.cast)
        for state, action, reward, done, s_prime in zip(
            states, actions, rewards, dones, s_primes
        ):
            self.experience.append(state, action, reward, done, s_prime)
            for _ in range(self.replays):
                # choose actions
                # act = self.sample_action(state)
                act = numpy.random.rand()
                scaled_act = self.scale(act)
                _, R, _ = self.env.step(torch.tensor([scaled_act, state]))
                # save transition to the replay memory
                self.experience.append(state, act, R[0], done, scaled_act)

    def train_net(self):
        if len(self.memory) >= self.min_memory:
            self.generate_experience()
            states, actions, rewards, done, s_prime = self.experience.replay(self.cast)
            [actions, done, rewards, states, s_prime] = [
                torch.reshape(x, [-1, 1])
                for x in [actions, done, rewards, states, s_prime]
            ]
            mu, std = self.pi(states)
            v = self.v(states)

            v_prime = self.v(s_prime)

            advantage = rewards + self.gamma * v_prime - v
            critic_loss = advantage**2

            dist = Normal(mu, std)

            _actions = 1e-6 + (1 - 1e-5) * actions
            logits = torch.log(_actions / (1 - _actions))
            actor_loss = -dist.log_prob(logits) * advantage.detach()
            entropy = -torch.mean(dist.entropy())

            loss = torch.mean(critic_loss + actor_loss) + self.entropy * entropy
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            self.optimizer.step()
            self.memory.empty()
            self.experience.empty()

    def reset(self, entropy):
        self.memory.empty()
        self.entropy = entropy
        for layer in self.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
        self.fc_v.bias.data.fill_(1000.0)

    def reset_value(self, entropy):
        self.memory.empty()
        self.entropy = entropy
        self.fc_v.reset_parameters()
        self.fc_v.bias.data.fill_(1000.0)

    def reset_pi(self, entropy):
        self.memory.empty()
        self.entropy = entropy
        self.fc_pi.reset_parameters()

    def save(self, loc):
        torch.save(self.state_dict(), loc)

    def load(self, loc):
        self.load_state_dict(torch.load(loc))
