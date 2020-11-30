import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal
import random
import math
from th_rl.buffers import *

class GreedyDiscrete():
    def __init__(self, env, experience):
        self.env = env
        self.last_action = 0
        self.collusion_action = 0.5/env.nplayers # Collusion
        self.monopole_action = 0.5               # Monopoly
        self.experience = experience
        self.memory = ReplayBuffer(1,self.experience)
        
    def sample_action(self, price):
        # Assumes other agents would play the same and he optimizes his action accordingly        
        others = 1-price/env.a-self.last_action
        new_action = np.clip((1-others)/2,self.collusion_action,self.monopole_action)
        self.last_action = new_action
        daction = int(new_action*self.env.nactions)
        return daction
    
    def train_net(self):
        pass

class GreedyContinuous():
    def __init__(self, env, experience):
        self.env = env
        self.last_action = 0
        self.collusion_action = 0.5/env.nplayers
        self.monopole_action = 0.5
        self.experience = experience
        self.memory = ReplayBuffer(1,self.experience)
        
    def sample_action(self, price):
        # Assumes other agents would play the same and he optimizes his action accordingly        
        others = 1-price/self.env.a-self.last_action
        new_action = np.clip((1-others)/2,self.collusion_action,self.monopole_action)
        self.last_action = new_action
        return new_action[0]
    
    def train_net(self):
        pass


class QTable():
    def __init__(self, states=16, actions=4, gamma=0.99, alpha=0.1, epsilon=0.9, epsilon_min=0.01, epsilon_decay=2e-6):
        self.table = np.zeros([states, actions])
        self.gamma = gamma
        self.alpha = alpha
        self.action_space = np.arange(0,actions)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def train_net(self, state, next_state, action):
        old_value = self.table[state, action]
        next_max = np.max(self.table[next_state])
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.table[state, action] = new_value
        self.epsilon = np.clip(self.epsilon-self.epsilon_decay, self.epsilon_min,1)

    def sample_action(self, state, epsilon=-1):
        eps = self.epsilon if epsilon==-1 else epsilon          
        if random.uniform(0, 1) < eps:
            action = random.choice(self.action_space)
        else:
            action = np.argmax(self.table[state])
        return action

class Reinforce(nn.Module):
    def __init__(self, states=4, actions=2,gamma=0.98, entropy=0, buffer='ReplayBuffer', capacity=50000):
        super(Reinforce, self).__init__()
        self.experience = namedtuple('Experience', field_names=['reward', 'action','prob'])
        self.memory = eval(buffer)(capacity,self.experience)
        self.gamma = gamma        
        self.fc1 = nn.Linear(states, 512)
        self.fc2 = nn.Linear(512, actions)
        self.optimizer = optim.Adam(self.parameters(), lr=5e-4)
        self.dist = Categorical
        self.entropy = entropy
        
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=0)
        return x

    def sample_action(self,state):
        prob = self(state)
        m = self.dist(prob)
        return m.sample(), prob

    def train_net(self):
        Rw,A,Pb = self.memory.replay()
        R,loss = 0,0
        for ret, a, prob in zip(Rw[::-1],A[::-1],Pb[::-1]):
            R = ret + self.gamma * R
            loss -= torch.log(prob[a]) * R
            entropy = -torch.sum(prob*torch.log(prob))
            loss -= self.entropy*entropy           

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()     
        self.memory.empty()           


class ActorCritic(nn.Module):
    def __init__(self, states=4, actions=2, gamma=0.98, entropy=0,buffer='ReplayBuffer', capacity=50000):
        super(ActorCritic, self).__init__()
        self.data = []
        self.gamma = gamma
        self.fc1 = nn.Linear(states,256)
        self.fc_pi = nn.Linear(256,actions)
        self.fc_v = nn.Linear(256,1)
        self.optimizer = optim.Adam(self.parameters(), lr=2e-4)
        self.experience = namedtuple('Experience', field_names=['state', 'action', 'reward','done', 'new_state'])
        self.cast = [torch.float, torch.int64, torch.float, torch.float, torch.float]
        self.memory = eval(buffer)(capacity,self.experience)
        self.dist = Categorical
        self.entropy = entropy
        
    def pi(self, x, softmax_dim = 0): # Pi=policy-> Actor
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob
    
    def v(self, x): # v = Value -> Critic
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v
          
    def sample_action(self, state):
        prob = self.pi(state)
        m = Categorical(prob)
        return m.sample().item()

    def train_net(self):
        s, a, r, done, s_prime = self.memory.replay(self.cast)
        [a,done,r] = [torch.reshape(x,[-1,1]) for x in [a,done,r]]
        
        td_target = r + self.gamma * self.v(s_prime) * done
        delta = td_target - self.v(s)

        pi = self.pi(s, softmax_dim=1)
        pi_a = pi.gather(1,a)
        loss = -torch.log(pi_a) * delta.detach() + F.smooth_l1_loss(self.v(s), td_target.detach())
        entropy = -torch.sum(pi*torch.log(pi),dim=1,keepdim=True)
        loss = loss - self.entropy*entropy

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()     
        self.memory.empty()            


class Qnet(nn.Module):
    def __init__(self, states=4, actions=2):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(states, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, actions)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x      

class DQN(nn.Module):
    def __init__(self, states=4, actions=2, batch_size=32, gamma=0.98, eps_end=2e-2, epsilon=0.1, eps_step=1e-4, copy_freq=20, buffer='ReplayBuffer', capacity=50000):
        super(DQN, self).__init__()
        self.q = Qnet(states, actions)
        self.q_target = Qnet(states, actions)
        self.q_target.load_state_dict(self.q.state_dict())
        self.optimizer = optim.Adam(self.parameters(), lr=5e-4)
        self.experience = namedtuple('Experience', field_names=['state', 'action', 'reward','done', 'new_state'])
        self.cast = [torch.float, torch.int64, torch.float, torch.float, torch.float]
        self.memory = eval(buffer)(capacity,self.experience)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_end = eps_end
        self.eps_step = eps_step
        self.copy_freq = copy_freq
        self.epoch = 0
        self.actions = actions

    def sample_action(self, obs):
        out = self.q(obs)
        coin = random.random()
        if coin < self.epsilon:
            return np.random.randint(0,self.actions)
        else : 
            return out.argmax().item()      

    def train_net(self):
        if len(self.memory)>2000:
            for i in range(10):
                s, a, r, done, s_prime = self.memory.sample(self.batch_size, self.cast)
                [a,done,r] = [torch.reshape(x,[-1,1]) for x in [a,done,r]]

                q_out = self.q(s)
                q_a = q_out.gather(1,a)
                max_q_prime = self.q_target(s_prime).max(1)[0].unsqueeze(1)
                target = r + self.gamma * max_q_prime * done
                loss = F.smooth_l1_loss(q_a, target)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            self.epsilon = np.clip(self.epsilon-self.eps_step,self.eps_end,1)
            self.epoch +=1
            if not self.epoch%self.copy_freq:
                self.q_target.load_state_dict(self.q.state_dict())        

class PPO(nn.Module):
    def __init__(self, states=4, actions=2,K_epoch=3,gamma=0.98,eps_clip=0.1,lmbda=0.95, buffer='ReplayBuffer', capacity=50000):
        super(PPO, self).__init__()
        self.gamma         = gamma
        self.lmbda         = lmbda
        self.eps_clip      = eps_clip
        self.K_epoch       = K_epoch
        self.experience = namedtuple('Experience', field_names=['state', 'action', 'reward','done', 'new_state', 'prob'])
        self.cast = [torch.float, torch.int64, torch.float, torch.float, torch.float, torch.float]
        self.memory = eval(buffer)(capacity,self.experience)

        self.fc1   = nn.Linear(states,256)
        self.fc_pi = nn.Linear(256,actions)
        self.fc_v  = nn.Linear(256,1)
        self.optimizer = optim.Adam(self.parameters(), lr=2e-4)

    def pi(self, x, softmax_dim = 0):
        x = torch.tanh(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob
    
    def v(self, x):
        x = torch.tanh(self.fc1(x))
        v = self.fc_v(x)
        return v
      
    def sample_action(self, obs):
        prob = self.pi(obs)
        m = Categorical(prob)
        a = m.sample().item()
        return a, prob[a].item()

    def train_net(self):
        s, a, r, done, s_prime, prob_a = self.memory.replay(self.cast)
        [a,done,r,prob_a] = [torch.reshape(x,[-1,1]) for x in [a,done,r,prob_a]]

        for i in range(self.K_epoch):
            td_target = r + self.gamma * self.v(s_prime) * done
            delta = td_target - self.v(s)
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = self.gamma * self.lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            pi = self.pi(s, softmax_dim=1)
            pi_a = pi.gather(1,a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s) , td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.memory.empty()                


class SACQNet(nn.Module):
    def __init__(self, states=3, actions=1, learning_rate=0.001, tau=0.01):
        super(QNet, self).__init__()
        self.tau = tau
        self.fc_s = nn.Linear(states, 64)
        self.fc_a = nn.Linear(actions,64)
        self.fc_cat = nn.Linear(128,32)
        self.fc_out = nn.Linear(32,actions)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x, a):
        h1 = F.relu(self.fc_s(x))
        h2 = F.relu(self.fc_a(a))
        cat = torch.cat([h1,h2], dim=1)
        q = F.relu(self.fc_cat(cat))
        q = self.fc_out(q)
        return q

    def train_net(self, target, mini_batch):
        s, a, r, done, s_prime = mini_batch
        loss = F.smooth_l1_loss(self.forward(s, a) , target)
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

    def soft_update(self, net_target):
        for param_target, param in zip(net_target.parameters(), self.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

class SACPolicyNet(nn.Module):
    def __init__(self, states=3, actions=1, learning_rate=0.0005, init_alpha=0.01, target_entropy=-1., lr_alpha=1e-3, activation=torch.tanh):
        super(PolicyNet, self).__init__()
        self.init_alpha = init_alpha
        self.target_entropy = target_entropy
        self.fc1 = nn.Linear(states, 128)
        self.fc_mu = nn.Linear(128,actions)
        self.fc_std  = nn.Linear(128,actions)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        self.log_alpha = torch.tensor(np.log(self.init_alpha))
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)
        self.activation = activation

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x))
        dist = Normal(mu, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action)
        real_action = self.activation(action)
        real_log_prob = log_prob - torch.log(1-real_action.pow(2) + 1e-7)
        return real_action, real_log_prob

    def train_net(self, q1, q2, mini_batch):
        s, a, r, done, s_prime = mini_batch
        a_prime, log_prob = self.forward(s_prime)
        entropy = -self.log_alpha.exp() * log_prob

        q1_val, q2_val = q1(s,a_prime), q2(s,a_prime)
        q1_q2 = torch.cat([q1_val, q2_val], dim=1)
        min_q = torch.min(q1_q2, 1, keepdim=True)[0]

        loss = -min_q - entropy # for gradient ascent
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = -(self.log_alpha.exp() * (log_prob + self.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()


class SAC(nn.Module):
    def __init__(self, states=3, actions=1, pi_learning_rate=0.0005, init_alpha=0.01, target_entropy=-1., lr_alpha=1e-3, q_learning_rate=0.001, tau=0.01, gamma=0.98, batch_size=32,activation=torch.sigmoid,buffer='ReplayBuffer',capacity=50000):
        super(SAC, self).__init__()
        self.experience = namedtuple('Experience', field_names=['state', 'action', 'reward','done', 'new_state'])
        self.cast = [torch.float, torch.float, torch.float, torch.float, torch.float]
        self.memory = eval(buffer)(capacity,self.experience)
        self.q1 = SACQNet(states, actions, q_learning_rate, tau)
        self.q2 = SACQNet(states, actions, q_learning_rate, tau)
        self.q1_target = SACQNet(states, actions, q_learning_rate, tau)
        self.q2_target = SACQNet(states, actions, q_learning_rate, tau)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        self.pi = SACPolicyNet(states, actions, pi_learning_rate, init_alpha, target_entropy, lr_alpha,activation)
        self.batch_size = batch_size
        self.gamma = gamma        

    def calc_target(self, mini_batch):
        s, a, r, done, s_prime = mini_batch
        with torch.no_grad():
            a_prime, log_prob= self.pi(s_prime)
            entropy = -self.pi.log_alpha.exp() * log_prob
            q1_val, q2_val = self.q1_target(s_prime,a_prime), self.q2_target(s_prime,a_prime)
            q1_q2 = torch.cat([q1_val, q2_val], dim=1)
            min_q = torch.min(q1_q2, 1, keepdim=True)[0]
            target = r + self.gamma * done * (min_q + entropy)
        return target

    def sample_action(self, obs):
        a, _ = self.pi(obs)
        return a.item()

    def train_net(self):
        if len(self.memory)>1000:
            for i in range(20):
                s, a, r, done, s_prime = self.memory.sample(self.batch_size, self.cast)
                [a,done,r] = [torch.reshape(x,[-1,1]) for x in [a,done,r]]
                mini_batch = [s, a, r, done, s_prime]
                td_target = self.calc_target(mini_batch)
                self.q1.train_net(td_target, mini_batch)
                self.q2.train_net(td_target, mini_batch)
                self.pi.train_net(self.q1, self.q2, mini_batch)
                self.q1.soft_update(self.q1_target)
                self.q2.soft_update(self.q2_target)        


class MuNet(nn.Module):
    def __init__(self, states, actions,activation):
        super(MuNet, self).__init__()
        self.fc1 = nn.Linear(states, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_mu = nn.Linear(64, actions)
        self.activation = activation

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.activation(self.fc_mu(x))
        return mu

class DDQNet(nn.Module):
    def __init__(self, states, actions):
        super(DDQNet, self).__init__()
        self.fc_s = nn.Linear(states, 64)
        self.fc_a = nn.Linear(actions,64)
        self.fc_q = nn.Linear(128, 32)
        self.fc_out = nn.Linear(32,actions)

    def forward(self, x, a):
        h1 = F.relu(self.fc_s(x))
        h2 = F.relu(self.fc_a(a))
        cat = torch.cat([h1,h2], dim=1)
        q = F.relu(self.fc_q(cat))
        q = self.fc_out(q)
        return q

class OrnsteinUhlenbeckNoise:
    def __init__(self, mu):
        self.theta, self.dt, self.sigma = 0.1, 0.01, 0.1
        self.mu = mu
        self.x_prev = np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x
      

class DDPG(nn.Module):
    def __init__(self,states=3,actions=1,lr_mu=5e-4, lr_q=1e-3,gamma=0.99,batch_size=32,tau=5e-3,buffer='ReplayBuffer', capacity=50000, activation=torch.sigmoid):
        super(DDPG, self).__init__()
        self.q = DDQNet(states,actions)
        self.q_target = DDQNet(states,actions)
        self.q_target.load_state_dict(self.q.state_dict())
        self.mu = MuNet(states,actions,activation)
        self.mu_target = MuNet(states,actions,activation)
        self.mu_target.load_state_dict(self.mu.state_dict())
        self.mu_optimizer = optim.Adam(self.mu.parameters(), lr=lr_mu)
        self.q_optimizer  = optim.Adam(self.q.parameters(), lr=lr_q)
        self.ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(1))
        self.experience = namedtuple('Experience', field_names=['state', 'action', 'reward','done', 'new_state'])
        self.cast = [torch.float, torch.float, torch.float, torch.float, torch.float]
        self.memory = eval(buffer)(capacity,self.experience)        
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau

    def sample_action(self, state):
        a = self.mu(state) 
        a = a.item() + self.ou_noise()[0]
        return a 

    def train(self):
        s, a, r, done, s_prime = self.memory.sample(self.batch_size, self.cast)
        [a,done,r] = [torch.reshape(x,[-1,1]) for x in [a,done,r]]      
        
        target = r + self.gamma * self.q_target(s_prime, self.mu_target(s_prime)) * done
        q_loss = F.smooth_l1_loss(self.q(s,a), target.detach())
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        
        mu_loss = -self.q(s,self.mu(s)).mean() # That's all for the policy loss.
        self.mu_optimizer.zero_grad()
        mu_loss.backward()
        self.mu_optimizer.step()
        self.soft_update()        
        
    def soft_update(self):
        for param_target, param in zip(self.mu_target.parameters(), self.mu.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)            
        for param_target, param in zip(self.q_target.parameters(), self.q.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)                