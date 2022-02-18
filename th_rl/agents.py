from collections import namedtuple
import numpy
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal
from th_rl.buffers import *


class QTable:
    def __init__(
        self,
        states=16,
        actions=4,
        action_range=[0, 1],
        gamma=0.99,
        buffer="ReplayBuffer",
        capacity=500,
        max_state=10,
        alpha=0.1,
        eps_end=2e-2,
        epsilon=0.5,
        eps_step=5e-4,
        min_memory=100,
        **kwargs
    ):
        self.table = 12.5 / (1 - gamma) + numpy.random.randn(states + 1, actions)
        self.gamma = gamma
        self.alpha = alpha
        self.action_space = numpy.arange(0, actions)
        self.action_range = action_range
        self.actions = actions
        self.epsilon = epsilon
        self.eps_step = eps_step
        self.eps_end = eps_end
        self.states = states
        self.max_state = max_state
        self.min_memory = min_memory
        self.experience = namedtuple(
            "Experience", field_names=["state", "action", "reward", "done", "new_state"]
        )
        self.memory = eval(buffer)(capacity, self.experience)
        self.counter = 0 * self.table

    def encode(self, state):
        astate = numpy.round(state / self.max_state * self.states).astype("int64")
        return astate

    def scale(self, actions):
        return (
            actions
            / (self.actions - 1.0)
            * (self.action_range[1] - self.action_range[0])
            + self.action_range[0]
        )

    def train_net(self):
        if len(self.memory) >= self.min_memory:
            price, acts, rwrd, not_done, next_state = self.memory.replay()
            state = self.encode(numpy.array(price))[:, 0]
            [actions, not_done, rewards] = [
                numpy.reshape(x, [-1]) for x in [acts, not_done, rwrd]
            ]
            next_state = self.encode(numpy.array(next_state))[:, 0]
            old_value = self.table[state, actions]
            for i, (ns, ov, re, st, ac) in enumerate(
                zip(next_state, old_value, rewards, state, actions)
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
        if random.uniform(0, 1) < self.epsilon:
            action = random.choice(self.action_space)
        else:
            if isinstance(state, torch.Tensor):
                st = state.numpy()
            else:
                st = state
            action = numpy.argmax(self.table[self.encode(st)])
        return action

    def get_action(self, state):
        return numpy.argmax(self.table[self.encode(state)])

    def reset(self, eps_end):
        self.table = 100 / (1 - self.gamma) + numpy.random.randn(
            self.states, self.actions
        )  # numpy.zeros([states, actions])#
        self.epsilon = 1.0
        self.eps_end = eps_end

    def reset_value(self, eps_end):
        self.table = 100 / (1 - self.gamma) + numpy.random.randn(
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


class Reinforce(nn.Module):
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
        super(Reinforce, self).__init__()
        self.data = []
        self.gamma = gamma
        self.action_range = action_range
        self.actions = actions
        self.fc1 = nn.Linear(states, 256)
        self.fc_pi = nn.Linear(256, actions)
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
                torch.reshape(x, [-1, 1]) for x in [actions, done, rewards]
            ]
            pi = self.pi(states, softmax_dim=1)

            discounted = rewards[:, 0].clone()
            for i, r in enumerate(torch.flip(rewards[:, 0], dims=[0])):
                if i > 0:
                    discounted[-i - 1] = r + self.gamma * discounted[-i]
            discounted = (discounted - torch.mean(discounted)) / torch.std(discounted)

            dist = Categorical(probs=pi)

            actor_loss = -torch.mean(dist.log_prob(actions[:, 0]) * discounted)
            entropy = -torch.mean(dist.entropy())

            loss = actor_loss + self.entropy * entropy
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
        mu, std = self.pi(state)
        dist = Normal(mu, std)
        action = dist.sample()
        return torch.sigmoid(action).item()

    def get_action(self, state):
        mu, _ = self.pi(torch.from_numpy(state).float())
        dist = Normal(mu, 0)
        action = dist.sample()
        return torch.sigmoid(action).item()

    def train_net(self):
        if len(self.memory) >= self.min_memory:
            states, actions, rewards, done, s_prime = self.memory.replay(self.cast)
            [actions, done, rewards] = [
                torch.reshape(x, [-1]) for x in [actions, done, rewards]
            ]

            mu, std = self.pi(states)
            v = self.v(states)
            v_prime = self.v(s_prime)

            advantage = rewards + self.gamma * v_prime - v
            critic_loss = advantage**2

            dist = Normal(mu, std)

            _actions = 5e-5 + (1 - 1e-4) * actions
            logits = torch.log(_actions / (1 - _actions))
            actor_loss = -dist.log_prob(logits) * advantage.detach()
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
