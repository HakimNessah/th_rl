from base64 import encode
from collections import namedtuple
from re import X
import numpy
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal
from th_rl.buffers import *
from numba import jit


class TitTat:
    def __init__(self, action_range=[0, 1], **kwargs):
        self.action_range = action_range
        self.statelen = kwargs.get("statelen", 1)
        self.action = kwargs.get("action", 0)
        self.actions = kwargs.get("actions", 20)
        self.experience = namedtuple(
            "Experience", field_names=["state", "action", "reward", "done", "new_state"]
        )
        self.cast = [torch.float, torch.float, torch.float, torch.float, torch.float]
        self.memory = eval(kwargs.get("buffer", "ReplayBuffer"))(
            kwargs.get("capacity", 100), self.experience
        )

    def scale(self, actions):
        return (
            actions
            / (self.actions - 1.0)
            * (self.action_range[1] - self.action_range[0])
            + self.action_range[0]
        )

    def sample_action(self, state):
        return state[self.action]

    def get_action(self, state):
        return state[self.action]

    def train_net(self):
        return None


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
        self.max_state = kwargs.get("max_state", 1)
        self.table = 12.5 / (1 - self.gamma) + numpy.random.randn(
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

    def scale(self, actions):
        return (
            actions
            / (self.actions - 1.0)
            * (self.action_range[1] - self.action_range[0])
            + self.action_range[0]
        )

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
        **kwargs,
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
        **kwargs,
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
        **kwargs,
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
        return [1, self.action]

    def get_action(self, state):
        return [1, self.action]

    def train_net(self):
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
        **kwargs,
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
        **kwargs,
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


class ConstIntention(nn.Module):
    def __init__(
        self,
        env,
        **kwargs,
    ):
        super(ConstIntention, self).__init__()
        self.env = env
        self.action = kwargs.get("action")
        self.experience = namedtuple(
            "Experience", field_names=["state", "action", "reward", "done", "new_state"]
        )
        self.memory = eval(kwargs.get("buffer", "ReplayBuffer"))(
            kwargs.get("capacity", 100), self.experience
        )

    def get_br_nash(self, qs):
        return (self.env.a - qs * self.env.b) / 2 / self.env.b

    @property
    def delta(self):
        return self.env.a / self.env.b / 8

    def train_net(self):
        return None

    def sample_action(
        self, state
    ):  # [myq(t-3),hisq(t-3),myq(t-2),hisq(t-2),myq(t-1),hisq(t-1)]
        nash = self.get_br_nash(state[0][-1])
        if self.action == 0:
            return [self.action, nash + self.delta]
        elif self.action == 1:
            return [self.action, nash]
        return [self.action, nash + self.delta]

    def get_action(self, state):
        return self.sample_action(state)


class IntentionAgent(nn.Module):
    def __init__(
        self,
        env,
        **kwargs,
    ):
        super(IntentionAgent, self).__init__()
        self.env = env
        self.gamma = kwargs.get("gamma", 0.95)
        self.alpha = kwargs.get("alpha", 0.15)
        self.epsilon = kwargs.get("epsilon", 1.0)
        self.eps_step = kwargs.get("eps_step", 0.9998)
        self.eps_end = kwargs.get("eps_end", 0.001)
        self.fc_intention = nn.Linear(4, 16)
        self.intention_mu = nn.Linear(16, 3)
        self.Q = numpy.random.rand(243, 3)
        self.min_memory = kwargs.get("min_memory", 100)
        self.optimizer = optim.Adam(self.parameters(), lr=2e-4)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.experience = namedtuple(
            "Experience", field_names=["state", "action", "reward", "done", "new_state"]
        )
        self.memory = eval(kwargs.get("buffer", "ReplayBuffer"))(
            kwargs.get("capacity", 100), self.experience
        )

    def get_br_nash(self, qs):
        return (self.env.a - qs * self.env.b) / 2 / self.env.b

    @property
    def delta(self):
        return self.env.a / self.env.b / 8

    def get_deltas(
        self, qs, tensors=True
    ):  # [myq(t-3),hisq(t-3),myq(t-2),hisq(t-2),myq(t-1),hisq(t-1)]
        nash = self.get_br_nash(qs)
        ix = [[2, 1], [3, 0], [4, 3], [5, 2]]
        ds = [qs[:, i[0]] - nash[:, i[1]] for i in ix]
        if tensors:
            return torch.stack(ds, axis=1)
        else:
            return numpy.stack(ds, axis=1)

    def intention(self, deltas):  # Pi=policy-> Actor
        x = torch.relu(self.fc_intention(deltas))
        mu = F.softmax(self.intention_mu(x), dim=-1)
        return mu

    def sample_action(
        self, state
    ):  # [myq(t-3),hisq(t-3),myq(t-2),hisq(t-2),myq(t-1),hisq(t-1)]
        nash = self.get_br_nash(state[0][-1])
        if random.uniform(0, 1) < self.epsilon:
            action = random.randint(0, 2)
        else:
            deltas = self.get_deltas(state, tensors=False)
            ins = torch.Tensor(deltas)
            intent = self.intention(ins)
            intent = intent.argmax(dim=-1).numpy()[0]
            ix = self.encode(intent, deltas[0])
            action = numpy.argmax(self.Q[ix])

        if action == 0:
            return [action, max(0, nash + self.delta)]
        elif action == 1:
            return [action, nash]
        return [action, min(self.env.a, nash - self.delta)]

    def get_action(self, state):
        nash = self.get_br_nash(state[0][-1])
        deltas = self.get_deltas(state, tensors=False)
        ins = torch.Tensor(deltas)
        intent = self.intention(ins)
        intent = intent.argmax(dim=-1).numpy()[0]
        ix = self.encode(intent, deltas[0])
        action = numpy.argmax(self.Q[ix])

        if action == 0:
            return [action, max(0, nash + self.delta)]
        elif action == 1:
            return [action, nash]
        return [action, min(self.env.a, nash - self.delta)]

    def encode(self, a, ds):
        qs = str(int(a))
        for d in ds:
            if d < -0.25:
                qs += "0"
            elif d > 0.25:
                qs += "2"
            else:
                qs += "1"
        return int(qs, 3)

    def train_net(self):
        if len(self.memory) >= self.min_memory:
            qs, acts, rwrd, not_done, next_qs = self.memory.replay()

            qs = torch.Tensor(
                numpy.stack(qs, axis=0)
            )  # [myq(t-3),hisq(t-3),myq(t-2),hisq(t-2),myq(t-1),hisq(t-1)]
            deltas = self.get_deltas(
                qs
            )  # [mydelta(t-2),hisdelta(t-2),mydelta(t-1),hisdelta(t-1)]

            his_intention = self.intention(deltas)  # [p0,p1,p2]
            obs = torch.concat([his_intention, deltas], axis=1)

            next_qs = torch.Tensor(next_qs)
            new_deltas = self.get_deltas(next_qs)
            new_intention = self.intention(new_deltas)
            new_obs = torch.concat([new_intention, new_deltas], axis=1)  # [1,0,1]

            # Intention loss
            y = new_deltas[:, -1].detach().numpy()
            y = 1 + numpy.where(numpy.abs(y) < 0.25, 0, -numpy.sign(y))
            y = torch.Tensor(y).type(torch.LongTensor)
            loss = self.loss_fn(his_intention, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # QTable update
            """
            Q[(0,1,1),action[0]] = alpha*Q[(0,1,1),action[0]] + (1-alpha)*(
                rewards[0] + gamma* (
                    max(p*Q([0,0,1], :)+(1-p)*Q([1,0,1], :))
                )            
            )
            """
            for a, r, row, newrow in zip(
                acts,
                rwrd,
                obs.detach().numpy(),
                new_obs.detach().numpy(),
            ):
                probs = row[:3]
                ds = row[3:]
                newprobs = newrow[:3]
                new_ds = newrow[3:]

                ix = self.encode(probs.argmax(), ds)
                nextQ = sum(
                    [newprobs[i] * self.Q[self.encode(i, new_ds)] for i in range(3)]
                )
                self.Q[ix, a] = self.alpha * self.Q[ix, a] + (1 - self.alpha) * (
                    r + self.gamma * max(nextQ)
                )

            self.memory.empty()
            self.epsilon = self.eps_end + (self.epsilon - self.eps_end) * self.eps_step


class Intention3(nn.Module):
    def __init__(
        self,
        env,
        **kwargs,
    ):
        super(Intention3, self).__init__()
        self.env = env
        self.lookback = kwargs.get("lookback", 3)
        self.gamma = kwargs.get("gamma", 0.9)
        self.alpha = kwargs.get("alpha", 0.1)
        self.epsilon = kwargs.get("epsilon", 1.0)
        self.eps_step = kwargs.get("eps_step", 0.9998)
        self.eps_end = kwargs.get("eps_end", 0.001)
        self.fc_intention = nn.Linear(2 * (self.lookback - 1), 16)
        self.intention_mu = nn.Linear(16, 3)
        self.Q = -numpy.random.rand(3, 3) + numpy.random.rand(3, 3) * 11.1111111 / (
            1 - self.gamma
        )
        self.Q[2, 2] = 12.5 / (1 - self.gamma)
        self.Q[1, 1] = 11.1111111 / (1 - self.gamma)
        self.Q[0, 0] = 3.81944444 / (1 - self.gamma)
        self.min_memory = kwargs.get("min_memory", 100)
        self.optimizer = optim.Adam(self.parameters(), lr=2e-4)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.experience = namedtuple(
            "Experience", field_names=["state", "action", "reward", "done", "new_state"]
        )
        self.memory = eval(kwargs.get("buffer", "ReplayBuffer"))(
            kwargs.get("capacity", 100), self.experience
        )

    def get_br_nash(self, qs):
        return (self.env.a - qs * self.env.b) / 2 / self.env.b

    @property
    def delta(self):
        return self.env.a / self.env.b / 8

    def get_deltas(
        self, qs, tensors=True
    ):  # [myq(t-3),hisq(t-3),myq(t-2),hisq(t-2),myq(t-1),hisq(t-1)]
        nash = self.get_br_nash(qs)

        mydeltas = [qs[:, i] - nash[:, i - 1] for i in range(2, 2 * self.lookback, 2)]
        hisdeltas = [qs[:, i] - nash[:, i - 3] for i in range(3, 2 * self.lookback, 2)]
        ds = mydeltas + hisdeltas
        if tensors:
            return torch.stack(ds, axis=1)
        else:
            return numpy.stack(ds, axis=1)

    def intention(self, deltas):  # Pi=policy-> Actor
        x = torch.tanh(self.fc_intention(deltas))
        mu = F.softmax(self.intention_mu(x), dim=-1)
        return mu

    def sample_action(self, state):
        nash = self.get_br_nash(state[0][-1])
        if random.uniform(0, 1) < self.epsilon:
            action = random.randint(0, 2)
        else:
            deltas = self.get_deltas(state, tensors=False)
            ins = torch.Tensor(deltas)
            intent = self.intention(ins)
            intent = intent.argmax(dim=-1).numpy()[0]
            ix = self.encode(intent)
            action = numpy.argmax(self.Q[ix])

        self.current_intention = action
        if action == 0:
            return [action, max(0, nash + self.delta)]
        elif action == 1:
            return [action, nash]
        return [action, min(self.env.a, nash - self.delta)]

    def get_action(self, state):
        nash = self.get_br_nash(state[0][-1])
        deltas = self.get_deltas(state, tensors=False)
        ins = torch.Tensor(deltas)
        intent = self.intention(ins)
        intent = intent.argmax(dim=-1).numpy()[0]
        ix = self.encode(intent)
        action = numpy.argmax(self.Q[ix])

        if action == 0:
            return [action, max(0, nash + self.delta)]
        elif action == 1:
            return [action, nash]
        return [action, min(self.env.a, nash - self.delta)]

    def encode(self, a):
        return int(a)

    def train_net(self):
        if len(self.memory) >= self.min_memory:
            qs, acts, rwrd, not_done, next_qs = self.memory.replay()

            qs = torch.Tensor(numpy.stack(qs, axis=0))
            deltas = self.get_deltas(qs)
            his_intention = self.intention(deltas)  # [p0,p1,p2]

            next_qs = torch.Tensor(numpy.stack(next_qs, axis=0))
            new_deltas = self.get_deltas(next_qs)
            new_intention = self.intention(new_deltas)

            # Intention loss
            y = new_deltas[:, -1].detach().numpy()
            y = 1 + numpy.where(numpy.abs(y) < 0.25, 0, -numpy.sign(y))
            y = torch.Tensor(y).type(torch.LongTensor)
            loss = self.loss_fn(his_intention, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # QTable update
            for a, prob, r, newprob in zip(
                acts,
                his_intention.detach().numpy(),
                rwrd,
                new_intention.detach().numpy(),
            ):
                ix = self.encode(prob.argmax())
                nextQ = sum([newprob[i] * self.Q[i] for i in range(3)])
                self.Q[ix, a] = (1 - self.alpha) * self.Q[ix, a] + self.alpha * (
                    r + self.gamma * max(nextQ)
                )
                # nextQ = sum([newprob[i] * max(self.Q[i]) for i in range(3)])
                # self.Q[ix, a] = self.alpha * self.Q[ix, a] + (1 - self.alpha) * (
                #    r + self.gamma * nextQ
                # )
            self.memory.empty()
            self.epsilon = self.eps_end + (self.epsilon - self.eps_end) * self.eps_step


class Intention_step(nn.Module):
    def __init__(
        self,
        env,
        **kwargs,
    ):
        super(Intention_step, self).__init__()
        self.env = env
        self.lookback = kwargs.get("lookback", 3)
        self.gamma = kwargs.get("gamma", 0.9)
        self.alpha = kwargs.get("alpha", 0.1)
        self.epsilon = kwargs.get("epsilon", 1.0)
        self.eps_step = kwargs.get("eps_step", 0.9998)
        self.eps_end = kwargs.get("eps_end", 0.001)
        self.step = kwargs.get("step", 0.1)
        self.fc_intention = nn.Linear(2 * self.lookback, 16)
        self.intention_mu = nn.Linear(16, 3)
        self.Q = -numpy.random.rand(3, 3) + numpy.random.rand(3, 3) * 11.1111111 / (
            1 - self.gamma
        )
        self.Q[2, 2] = 12.5 / (1 - self.gamma)
        self.Q[1, 1] = 11.1111111 / (1 - self.gamma)
        self.Q[0, 0] = 3.81944444 / (1 - self.gamma)
        self.min_memory = kwargs.get("min_memory", 100)
        self.optimizer = optim.Adam(self.parameters(), lr=2e-4)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.experience = namedtuple(
            "Experience", field_names=["state", "action", "reward", "done", "new_state"]
        )
        self.memory = eval(kwargs.get("buffer", "ReplayBuffer"))(
            kwargs.get("capacity", 100), self.experience
        )

    def get_br_nash(self):
        return [20 / 3 - 2.5, 10 / 3, 2.5]

    def intention(self, deltas):  # Pi=policy-> Actor
        x = torch.tanh(self.fc_intention(deltas))
        mu = F.softmax(self.intention_mu(x), dim=-1)
        return mu

    def sample_action(
        self, state
    ):  # [myq(t-3),hisq(t-3),myq(t-2),hisq(t-2),myq(t-1),hisq(t-1)]
        [compete, nash, collude] = self.get_br_nash()
        if random.uniform(0, 1) < self.epsilon:
            action = random.randint(0, 2)
        else:
            ins = torch.Tensor(state)
            intent = self.intention(ins)
            intent = intent.argmax(dim=-1).numpy()[0]
            ix = self.encode(intent)
            action = numpy.argmax(self.Q[ix])

        mylast = state[0][-2]
        if action == 0:
            return [action, (1 - self.step) * mylast + self.step * compete]
        elif action == 1:
            return [action, (1 - self.step) * mylast + self.step * nash]
        return [action, (1 - self.step) * mylast + self.step * collude]

    def get_action(
        self, state
    ):  # [myq(t-3),hisq(t-3),myq(t-2),hisq(t-2),myq(t-1),hisq(t-1)]
        [compete, nash, collude] = self.get_br_nash()

        ins = torch.Tensor(state)
        intent = self.intention(ins)
        intent = intent.argmax(dim=-1).numpy()[0]
        ix = self.encode(intent)
        action = numpy.argmax(self.Q[ix])

        mylast = state[0][-2]
        if action == 0:
            return [action, (1 - self.step) * mylast + self.step * compete]
        elif action == 1:
            return [action, (1 - self.step) * mylast + self.step * nash]
        return [action, (1 - self.step) * mylast + self.step * collude]

    def encode(self, a):
        return int(a)

    def train_net(self):
        if len(self.memory) >= self.min_memory:
            qs, acts, rwrd, not_done, next_qs = self.memory.replay()

            qs = torch.Tensor(numpy.stack(qs, axis=0))
            his_intention = self.intention(qs)  # [p0,p1,p2]

            next_qs = torch.Tensor(numpy.stack(next_qs, axis=0))
            new_intention = self.intention(next_qs)

            # Intention loss
            last = qs[:, -1]
            nextlast = next_qs[:, -1]
            mat = (
                torch.stack(
                    [
                        (1 - self.step) * last + self.step * anchor
                        for anchor in self.get_br_nash()
                    ],
                    axis=1,
                )
                - nextlast[:, None]
            )
            mat = torch.abs(mat)

            y = torch.argmin(mat, axis=1)
            loss = self.loss_fn(his_intention, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # QTable update
            for a, prob, r, newprob in zip(
                acts,
                his_intention.detach().numpy(),
                rwrd,
                new_intention.detach().numpy(),
            ):
                ix = self.encode(prob.argmax())
                nextQ = sum([newprob[i] * self.Q[i] for i in range(3)])
                self.Q[ix, a] = (1 - self.alpha) * self.Q[ix, a] + self.alpha * (
                    r + self.gamma * max(nextQ)
                )
                # nextQ = sum([newprob[i] * max(self.Q[i]) for i in range(3)])
                # self.Q[ix, a] = self.alpha * self.Q[ix, a] + (1 - self.alpha) * (
                #    r + self.gamma * nextQ
                # )
            self.memory.empty()
            self.epsilon = self.eps_end + (self.epsilon - self.eps_end) * self.eps_step

    def save(self, loc):
        torch.save(self.state_dict(), loc)
        numpy.save(loc, self.Q)

    def load(self, loc):
        self.load_state_dict(torch.load(loc))
        self.Q = numpy.load(loc + ".npy")
