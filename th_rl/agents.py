from collections import namedtuple
import numpy 
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from th_rl.buffers import *


class QTable():
    def __init__(self, states=16, actions=4, action_range=[0,1], gamma=0.99, buffer='ReplayBuffer', capacity=500, max_state=10,
                alpha=0.1, eps_end=2e-2, epsilon=0.5, eps_step=5e-4, **kwargs):
        self.table =  1/gamma+numpy.random.randn(states+1, actions)
        self.gamma = gamma
        self.alpha = alpha
        self.action_space = numpy.arange(0,actions)
        self.action_range = action_range
        self.actions = actions
        self.epsilon = epsilon
        self.eps_step = eps_step
        self.eps_end = eps_end
        self.states = states
        self.max_state = max_state
        self.experience = namedtuple('Experience', field_names=['state', 'action', 'reward','done', 'new_state'])
        self.memory = eval(buffer)(capacity,self.experience)
        self.counter = 0*self.table

    def encode(self,state):
        astate = (state/self.max_state*self.states).astype('int64')
        return astate
    
    def scale(self,actions):
        return actions/(self.actions-1.)*(self.action_range[1]-self.action_range[0])+self.action_range[0]

    def train_net(self):
        price, acts, rwrd, not_done, next_state = self.memory.replay()
        state = self.encode(numpy.array(price))[:,0]
        [actions,not_done,rewards] = [numpy.reshape(x,[-1]) for x in [acts,not_done,rwrd]]
        next_state = self.encode(numpy.array(next_state))[:,0]
        old_value = self.table[state, actions]
        for i,(ns,ov,re,st,ac) in enumerate(zip(next_state,old_value,rewards,state,actions)):
            if not i%100:
              continue
            next_max = numpy.max(self.table[ns])       
            new_value = (1 - self.alpha) * ov + self.alpha * (re + self.gamma * next_max)
            self.table[st,ac] = new_value
            self.counter[st,ac] += 1

        self.epsilon = self.eps_end + (self.epsilon-self.eps_end)*self.eps_step

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
        self.table = 100/(1-self.gamma) + numpy.random.randn(self.states, self.actions) # numpy.zeros([states, actions])#
        self.epsilon = 1.
        self.eps_end = eps_end

    def reset_value(self, eps_end):
        self.table = 100/(1-self.gamma) + numpy.random.randn(self.states, self.actions) # numpy.zeros([states, actions])#

    def reset_pi(self, eps_end):
        self.epsilon = 1.
        self.eps_end = eps_end  

    def save(self,loc):
      numpy.save(loc, self.table)
      numpy.save(loc+'_counter', self.counter)

    def load(self, loc):
        self.table  = numpy.load(loc+'.npy')
        self.counter = numpy.load(loc+'_counter.npy')

class A2C(nn.Module):
    def __init__(self, states=4, actions=2, action_range=[0,1],gamma=0.98, buffer='ReplayBuffer', capacity=50000, **kwargs):
        super(A2C, self).__init__()
        self.data = []
        self.gamma = gamma
        self.action_range = action_range
        self.fc1 = nn.Linear(states,256)
        self.fc2 = nn.Linear(states+actions,256)
        self.fc_pi = nn.Linear(256,actions)
        self.fc_v = nn.Linear(256,1)        
        self.fc_v.bias.data.fill_(1000.)
        self.optimizer = optim.Adam(self.parameters(), lr=2e-4)
        self.experience = namedtuple('Experience', field_names=['state', 'action', 'reward','done', 'new_state'])
        self.cast = [torch.float, torch.int64, torch.float, torch.float, torch.float]
        self.memory = eval(buffer)(capacity,self.experience)
        
    def pi(self, x, softmax_dim = 0): # Pi=policy-> Actor
        x = torch.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob
    
    def v(self, x, pi): # v = Value -> Critic
        y = torch.cat([x,pi],axis=1)
        y = torch.relu(self.fc2(y))
        v = self.fc_v(y)
        return v

    def scale(self, action):
        return (1/(1+numpy.exp(-action))*(self.action_range[1]-self.action_range[0])+self.action_range[0])
          
    def sample_action(self, state):
        prob = self.pi(state)
        m = Categorical(prob)
        return m.sample().item()

    def get_action(self, state):
        pi = self.pi(state)
        m = torch.argmax(pi)
        return m.item()

    def train_net(self):
        states, actions, rewards, done, s_prime = self.memory.replay(self.cast)      
        [actions,done,rewards] = [torch.reshape(x,[-1,1]) for x in [actions,done,rewards]]        

        pi = self.pi(states, softmax_dim=1)
        pi_prime = self.pi(s_prime, softmax_dim=1)
        dist = Categorical(probs=pi)
        td = rewards + self.gamma * self.v(s_prime, pi_prime.detach()) - self.v(states, pi.detach())
        critic_loss = torch.mean(td**2)
        actor_loss = -torch.mean(dist.log_prob(actions[:,0])*td[:,0].detach())

        loss = critic_loss + actor_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()     
        self.memory.empty()

    def reset(self, entropy):
        self.memory.empty()
        self.entropy = entropy
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()        
        self.fc_v.bias.data.fill_(1000.)                

    def reset_value(self, entropy):
        self.memory.empty()
        self.entropy = entropy
        self.fc_v.reset_parameters()  
        self.fc_v.bias.data.fill_(1000.)  

    def reset_pi(self, entropy):
        self.memory.empty()
        self.entropy = entropy
        self.fc_pi.reset_parameters()  

    def save(self,loc):
        torch.save(self.state_dict(), loc)

    def load(self, loc):
        self.load_state_dict(loc)