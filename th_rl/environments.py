import numpy 


class NoisyPriceState():
    def __init__(self, nplayers, action_range=[0,1], a=10, b=1,  max_steps=1, noise_prob = 0.05, **kwargs):
        self.nplayers = nplayers
        self.action_range = action_range
        self.b = b
        self.a = a
        self.max_steps = max_steps
        self.state = self.sample_state()      
        self.episode = 0 
        self.noise_prob = noise_prob

    def sample_state(self):
        return numpy.random.uniform(0, self.a)
    
    def encode(self):
        return numpy.atleast_1d(self.state)

    # Handles proper scaling for both discete and continuous action spaces
    def scale_actions(self, actions):
        return [self.a/self.b*a for a in actions]

    def step(self, actions):
        A = self.scale_actions(actions)
        Q = sum(A)
        if numpy.random.uniform(0,1)<self.noise_prob:
            new_a = numpy.random.uniform(self.a*0.7, self.a)
        else:
            new_a = self.a
        price = numpy.max([0,new_a - self.b*Q])
            
        rewards = [price*a for a in A]

        self.state = price  
        self.episode += 1
        done = self.episode>=self.max_steps
        return self.encode(), numpy.array(rewards), done 
    
    def get_optimal(self):
        anash = (self.a/self.b)*numpy.ones(self.nplayers,)/(self.nplayers+1)
        price = numpy.max([0,self.a - self.b*sum(anash)])
        rnash = [price*a for a in anash]
        acoll = (self.a/self.b)*0.5*numpy.ones(self.nplayers,)/self.nplayers
        price = numpy.max([0,self.a - self.b*sum(acoll)])
        rcoll = [price*a for a in acoll]  
        return sum(rnash), sum(rcoll)

    def reset(self):
        self.episode = 0
        self.state = self.sample_state()
        return self.encode()        

