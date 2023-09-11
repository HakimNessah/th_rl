from collections import namedtuple
import numpy
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal
from th_rl.buffers import *
import numbers
from pylab import *

class QTable:
    def __init__(
        self,
        states=20,
        actions=20,
        action_range=[2, 4],
        gamma=0.9,
        buffer="ReplayBuffer",
        capacity=1,
        alpha=0.1,
        eps_end=2e-2,
        epsilon=0.5,
        eps_step=5e-4,
        min_memory=1,
        state_range=[0,10],
        **kwargs
    ):
        self.table = 12.5 / (1 - gamma) + numpy.random.randn(states, actions)
        self.gamma = gamma
        self.alpha = alpha
        self.a2q = numpy.linspace(action_range[0], action_range[1], actions)
        self.p2s = numpy.linspace(state_range[0], state_range[1], states)
        self.nactions = actions
        self.epsilon = epsilon
        self.eps_step = eps_step
        self.eps_end = eps_end
        self.states = states
        self.min_memory = min_memory
        self.experience = namedtuple(
            "Experience", field_names=["state", "action", "reward", "new_state"]
        )
        self.memory = eval(buffer)(capacity, self.experience)
        self.counter = 0 * self.table

    def encode_state(self, state):
        # State = Price
        if isinstance(state, numbers.Number) or (len(state.shape) == 1):
            ix = numpy.argmin(numpy.abs(state - self.p2s))
        else:
            ix = numpy.argmin(numpy.abs(state - self.p2s), axis=1)
        return ix
    
    def encode_action(self, action):
        # Action = Quantity
        if isinstance(action, numbers.Number) or (len(action.shape) == 1):
            ix = numpy.argmin(numpy.abs(action - self.a2q))
        else:
            ix = numpy.argmin(numpy.abs(action - self.a2q), axis=1)
        return ix    

    def train(self):
        if len(self.memory) >= self.min_memory:
            states, acts, rewards, next_states = self.memory.replay()
            states_ix = self.encode_state(numpy.array(states)[:, None])
            next_states_ix = self.encode_state(numpy.array(next_states)[:, None])
            actions = self.encode_action(numpy.array(acts)[:, None])
            for ns, re, st, ac in zip(next_states_ix, rewards, states_ix, actions):                
                newval = re + self.gamma * numpy.max(self.table[ns])
                self.table[st, ac] = (1 - self.alpha) * self.table[
                    st, ac
                ] + self.alpha * newval
                self.counter[st, ac] += 1
            self.memory.empty()
        self.epsilon = self.eps_end + (self.epsilon - self.eps_end) * self.eps_step

    def sample(self, obs):
        if numpy.random.rand() < self.epsilon:
            return numpy.random.choice(self.a2q), []
        ix = self.encode_state(obs)
        action = numpy.argmax(self.table[ix])
        return self.a2q[action], []

    def act(self, obs):
        ix = self.encode_state(obs)
        action = numpy.argmax(self.table[ix])
        return self.a2q[action]

    def save(self, loc):
        numpy.save(loc, self.table)
        numpy.save(loc + "_counter", self.counter)

    def load(self, loc):
        self.table = numpy.load(loc + ".npy")
        self.counter = numpy.load(loc + "_counter.npy")

class Qnet(nn.Module):
    def __init__(self, states, actions):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(states, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
      
class DQN(nn.Module):
    def __init__(
        self,
        states=20,
        actions=20,
        action_range=[2, 4],
        gamma=0.9,
        buffer="ReplayBuffer",
        capacity=1,
        alpha=0.1,
        eps_end=2e-2,
        epsilon=0.5,
        eps_step=5e-4,
        min_memory=1,
        state_range=[0,10],
        copy_freq = 10,
        counter = 0,
        **kwargs
    ):
        super(DQN, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.a2q = numpy.linspace(action_range[0], action_range[1], actions)
        self.nactions = actions
        self.epsilon = epsilon
        self.eps_step = eps_step
        self.eps_end = eps_end
        self.states = states
        self.min_memory = min_memory
        self.state_range = state_range
        self.experience = namedtuple(
            "Experience", field_names=["state", "action", "reward", "new_state"]
        )
        self.memory = eval(buffer)(capacity, self.experience)
        self.Q = Qnet(states, actions)
        self.Qtarget = Qnet(states, actions)
        self.Qtarget.load_state_dict(self.Q.state_dict())
        self.optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        self.copy_freq = copy_freq
        self.counter = counter
   
    def encode_state(self, obs):        
        x = numpy.atleast_2d(obs).astype('float32')
        x -= self.state_range[0]
        x *= 2/(self.state_range[1]-self.state_range[0])
        return torch.Tensor(x-1)
    
    def encode_action(self, action):
        # Action = Quantity
        if isinstance(action, numbers.Number) or (len(action.shape) == 1):
            ix = numpy.argmin(numpy.abs(action - self.a2q))
        else:
            ix = numpy.argmin(numpy.abs(action - self.a2q), axis=1)
        return ix  
        
    def sample(self, obs):        
        if numpy.random.rand() < self.epsilon:
            return numpy.random.choice(self.a2q)
        inputs = self.encode_state(obs)
        table = self.Q(inputs)
        return self.a2q[table.argmax().item()]
    
    def act(self, obs):
        inputs = self.encode_state(obs)
        table = self.Q(inputs)
        return self.a2q[table.argmax().item()]

    def train(self):
        if len(self.memory) >= self.min_memory:
            S, acts, rewards, S_prime = self.memory.replay()
            states = self.encode_state(numpy.array(S)[:, None])
            next_states = self.encode_state(numpy.array(S_prime)[:, None])
            actions = torch.LongTensor(self.encode_action(numpy.array(acts)[:,None])[:,None])
            rewards = torch.Tensor(numpy.array(rewards)[:,None])
            q_out = self.Q(states)
            q_a = q_out.gather(1,actions)
            max_q_prime = self.Qtarget(next_states).max(1)[0].unsqueeze(1)
            target = rewards + self.gamma * max_q_prime 
            loss = F.smooth_l1_loss(q_a, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.memory.empty()
            self.counter += 1
            if self.counter%self.copy_freq==0:
                self.Qtarget.load_state_dict(self.Q.state_dict())
        self.epsilon = self.eps_end + (self.epsilon - self.eps_end) * self.eps_step

    def save(self, loc):
        torch.save(self.state_dict(), loc)

    def load(self, loc):
        self.load_state_dict(torch.load(loc))

class Reinforce(nn.Module):
    def __init__(
        self,
        states=4,
        actions=2,
        action_range=[0, 1],
        gamma=0.98,
        buffer="ReplayBuffer",
        capacity=1000,
        min_memory=1000,
        **kwargs
    ):
        super(Reinforce, self).__init__()
        self.data = []
        self.gamma = gamma
        self.action_range = action_range
        self.actions = actions
        self.fc1 = nn.Linear(states, 128)
        self.fc_pi = nn.Linear(128, actions)
        self.optimizer = optim.AdamW(self.parameters(), lr=1e-4)
        self.experience = namedtuple(
            "Experience", field_names=["reward", "logprob"]
        )
        self.memory = eval(buffer)(capacity, self.experience)
        self.min_memory = min_memory
        self.a2q = numpy.linspace(action_range[0], action_range[1], actions)
        
    def encode_state(self, obs):        
        x = numpy.atleast_2d(obs).astype('float32')
        return torch.Tensor(x)

    def encode_action(self, action):
        # Action = Quantity
        if isinstance(action, numbers.Number) or (len(action.shape) == 1):
            ix = numpy.argmin(numpy.abs(action - self.a2q))
        else:
            ix = numpy.argmin(numpy.abs(action - self.a2q), axis=1)
        return ix    

    def pi(self, x):  # Pi=policy-> Actor
        x = torch.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=-1)
        return prob

    def sample(self, state):
        probs = self.pi(self.encode_state(state))
        highest_prob_action = numpy.random.choice(self.actions, p=numpy.squeeze(probs.detach().numpy()))
        prob = probs.squeeze(0)[highest_prob_action]
        return self.a2q[highest_prob_action], prob

    def act(self, state):
        pi = self.pi(self.encode_state(state))
        m = torch.argmax(pi)
        return self.a2q[m.item()]

    def train(self):
        if len(self.memory) >= self.min_memory:
            rewards, probs = self.memory.replay()
            logprobs = torch.log(torch.stack(probs))

            # Discount&Normalize
            R = numpy.array(rewards)
            R -= R.mean()
            for i,r in enumerate(R[::-1]):
                j = len(R)-i-1
                if i>0:
                    R[j] = r + self.gamma * R[j+1]
            R = (R-R.mean())/(R.std()+1e-9)

            # Accumulate loss
            loss = 0
            for logp,r in zip(logprobs, R):
                loss -= logp*r

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            self.optimizer.step()                
            self.memory.empty()

    def save(self, loc):
        torch.save(self.state_dict(), loc)

    def load(self, loc):
        self.load_state_dict(torch.load(loc))

class CAC(nn.Module):
    def __init__(
        self,
        states=1,
        actions=1,
        action_range=[0, 1],
        state_range=[0,10],
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
        self.state_range = state_range
        self.actions = actions
        self.fc1 = nn.Linear(states, 128)
        self.fc_mu = nn.Linear(128, 1)
        self.fc_std = nn.Linear(128, 1)
        self.fc_v = nn.Linear(128, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=2e-4)
        self.experience = namedtuple(
            "Experience", field_names=["state", "action", "reward", "new_state"]
        )
        self.memory = eval(buffer)(capacity, self.experience)
        self.min_memory = min_memory
        self.entropy = entropy

    def encode_state(self, obs):        
        x = numpy.atleast_2d(obs).astype('float32')
        x -= self.state_range[0]
        x *= 2/(self.state_range[1]-self.state_range[0])
        return torch.Tensor(x-1)
    
    def encode_action(self, actions):
        return (actions - self.action_range[0])/(self.action_range[1]-self.action_range[0])
   
    def pi(self, x):  # Pi=policy-> Actor
        x = torch.relu(self.fc1(x))
        mu = 4.0 * torch.tanh(self.fc_mu(x))
        std = F.softplus(self.fc_std(x))
        return mu, std

    def v(self, x):  # v = Value -> Critic
        y = torch.relu(self.fc1(x))
        v = self.fc_v(y)
        return v
    
    def sample(self, state):
        mu, std = self.pi(self.encode_state(state))
        dist = Normal(mu, std)
        sample = dist.sample()
        action = torch.sigmoid(sample).item()    
        return self.action_range[0] + (self.action_range[1]-self.action_range[0])*action, []

    def act(self, state):
        return self.sample(state)

    def train(self):
        if len(self.memory) >= self.min_memory:
            states, actions, rewards, s_prime = self.memory.replay()

            [actions, rewards] = [
                numpy.stack(x) for x in [actions, rewards]
            ]
            actions = self.encode_action(actions)
            breakpoint()

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

    def save(self, loc):
        torch.save(self.state_dict(), loc)

    def load(self, loc):
        self.load_state_dict(torch.load(loc))


class Exp3:
    def __init__(
        self,
        states=20,
        actions=20,
        action_range=[2, 4],
        gamma=0.9,
        buffer="ReplayBuffer",
        capacity=1,
        eta=0.1,
        min_memory=1,
        state_range=[0,10],
        alpha=0.1,
        **kwargs
    ):
        self.gamma = gamma
        self.eta = eta
        self.a2q = numpy.linspace(action_range[0], action_range[1], actions)
        self.p2s = numpy.linspace(state_range[0], state_range[1], states)
        self.nactions = actions
        self.states = states
        self.min_memory = min_memory
        self.experience = namedtuple("Experience", field_names=["action", "reward"])
        self.memory = eval(buffer)(capacity, self.experience)
        noise = numpy.random.randn(actions)
        noise -= noise.mean()
        self.weights = numpy.ones((actions,))+noise/100
        self.alpha = alpha

    @property
    def probabilities(self):
        W = self.weights
        wsum = sum(W)
        probs = (1-self.eta)*W/wsum + self.eta/self.nactions     
        return probs

    
    def encode_action(self, action):
        # Action = Quantity
        if isinstance(action, numbers.Number) or (len(action.shape) == 1):
            ix = numpy.argmin(numpy.abs(action - self.a2q))
        else:
            ix = numpy.argmin(numpy.abs(action - self.a2q), axis=1)
        return ix    


    def sample(self, data):
        return numpy.random.choice(self.a2q,p=self.probabilities), []
    
    def act(self):
        return self.sample()

    def train(self):
        if len(self.memory) >= self.min_memory:
            acts, rewards = self.memory.replay()
            actions = self.encode_action(numpy.array(acts)[:, None])
            rewards = numpy.array(rewards)
            rhat = rewards-12.5 #numpy.clip(rewards/25,0,1)
            probs = self.probabilities    
            W = numpy.zeros(self.weights.shape)            
            for i, (re,  ac) in enumerate(zip(rhat, actions)):
                W[ac] += re*self.gamma**(i+1)
            Y = self.eta*W/probs/self.nactions
            self.weights *= numpy.exp(Y)            
            self.memory.empty()


    def save(self, loc):
        numpy.save(loc, self.weights)

    def load(self, loc):
        self.weights = numpy.load(loc + ".npy")
