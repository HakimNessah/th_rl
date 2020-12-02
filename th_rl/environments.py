import numpy as np

class PriceState():
    def __init__(self, nactions, nplayers, cost,action_range=[0,1], a=10, b=1,  max_steps=1, discrete=True, encoder='none'):
        self.nactions = nactions
        self.nplayers = nplayers
        self.cost = cost
        self.action_range = action_range
        self.b = b
        self.a = a
        self.max_steps = max_steps
        self.discrete = discrete                
        self.state = self.sample_state()      
        self.episode = 0 
        self.encoder = encoder
        assert encoder=='none', 'Discrete Prices are not supported!'

    def sample_state(self):
        return np.random.uniform(0, self.a)
    
    def encode(self):
        return np.atleast_1d(self.state)

    def scale_actions(self, actions):
        A = np.array(actions)
        if self.discrete:
            return (self.a/self.b)*(A/(self.nactions-1.)*(self.action_range[1]-self.action_range[0])+self.action_range[0])
        else:
            return (self.a/self.b)*(1/(1+np.exp(-A))*(self.action_range[1]-self.action_range[0])+self.action_range[0])

    def step(self, actions):
        A = self.scale_actions(actions)
        Q = sum(A)
        price = np.max([0,self.a - self.b*Q])
        welfare = self.a * Q - 0.5 * self.b * Q**2 - sum([c*a for c,a in zip(self.cost,A)])
        rewards = [(price-c)*a for c,a in zip(self.cost,A)]

        self.state = price  
        self.episode += 1
        done = self.episode>=self.max_steps
        return np.atleast_1d(self.state), np.array(rewards), welfare, done 
    
    def get_optimal(self):
        anash = (self.a/self.b)*np.ones(self.nplayers,)/(self.nplayers+1)
        price = np.max([0,self.a - self.b*sum(anash)])
        rnash = [(price-c)*a for c,a in zip(self.cost,anash)]        
        acoll = (self.a/self.b)*0.5*np.ones(self.nplayers,)/self.nplayers
        price = np.max([0,self.a - self.b*sum(acoll)])
        rcoll = [(price-c)*a for c,a in zip(self.cost,acoll)]  
        return sum(rnash), sum(rcoll)

    def reset(self):
        self.episode = 0
        self.state = self.sample_state()
        return self.encode()


class ActionState():
    def __init__(self, nactions, nplayers, cost,action_range=[0,1], a=10, b=1,  max_steps=1, discrete=True, encoder='none'):
        self.nactions = nactions
        self.nplayers = nplayers
        self.cost = cost
        self.action_range = action_range
        self.b = b
        self.a = a
        self.max_steps = max_steps
        self.discrete = discrete                
        self.state = self.sample_state()      
        self.episode = 0 
        self.encoder = encoder
        if not discrete:
            assert action_range==[0,1], 'In continuous action space range is assumed to be [0,1]'

    def encode(self):
        if self.discrete:
            if self.encoder=='none':
                state = np.array(self.state)
            if self.encoder=='one_hot':
                state = np.eye(self.nactions)[self.state]
            if self.encoder=='full_one_hot':
                act = sum([a*self.nactions**i for i,a in enumerate(self.state)])
                state = np.eye(self.nactions**len(self.state))[act]
            return np.reshape(state,(1,-1))
        return np.array(self.state)

    def sample_state(self):
        if self.discrete:
            state = np.random.randint(0,self.nactions, self.nplayers)
        else:
            state = np.random.uniform(self.action_range[0],self.action_range[1], self.nplayers)
        return state
    
    def scale_actions(self, actions):
        A = np.array(actions)
        if self.discrete:
            return (self.a/self.b)*(A/(self.nactions-1.)*(self.action_range[1]-self.action_range[0])+self.action_range[0])
        else:
            return (self.a/self.b)*(1/(1+np.exp(-A))*(self.action_range[1]-self.action_range[0])+self.action_range[0])

    def step(self, actions):
        A = self.scale_actions(actions)
        Q = sum(A)
        price = np.max([0,self.a - self.b*Q])
        welfare = self.a * Q - 0.5 * self.b * Q**2 - sum([c*a for c,a in zip(self.cost,A)])
        rewards = [(price-c)*a for c,a in zip(self.cost,A)]

        self.state = actions  
        self.episode += 1
        done = self.episode>=self.max_steps
        return np.atleast_1d(self.state), np.array(rewards), welfare, done 
    
    def get_optimal(self):
        anash = (self.a/self.b)*np.ones(self.nplayers,)/(self.nplayers+1)
        price = np.max([0,self.a - self.b*sum(anash)])
        rnash = [(price-c)*a for c,a in zip(self.cost,anash)]        
        acoll = (self.a/self.b)*0.5*np.ones(self.nplayers,)/self.nplayers
        price = np.max([0,self.a - self.b*sum(acoll)])
        rcoll = [(price-c)*a for c,a in zip(self.cost,acoll)]  
        return sum(rnash), sum(rcoll)

    def reset(self):
        self.episode = 0
        self.state = self.sample_state()
        return self.encode()      