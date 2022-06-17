import numpy
import torch


class Hotelling:
    def __init__(self, nplayers=2, max_steps=100, phi=0.8, t=1, u=2.96, **kwargs):
        self.nplayers = nplayers
        self.max_steps = max_steps
        self.episode = 0
        self.t = t
        self.u = u
        self.phi = phi
        self.state = self.sample_state()
        assert nplayers == 2, "Env is only defined for exactly two players"

    def sample_state(self):
        return torch.rand(self.nplayers) * self.u
        # numpy.random.uniform(0, self.u, [self.nplayers])

    def calc_demand(self, p0, p1):
        if p0 < p1 - self.t:
            return 0.5 * (1 + self.phi)
        if p0 < p1 + self.t:
            return 0.5 * (self.phi * (p1 - p0) / self.t + 1)
        if p0 <= self.u - 0.5 * self.t:
            return 0.5 * (1 - self.phi)
        assert 1 == 0, "Not supposed to be here"

    def get_Nash(self):
        price = self.t / self.phi
        demand = self.calc_demand(price, price)
        profit = price * demand
        return price, demand, profit

    def step(self, prices):
        self.state[0] = self.calc_demand(prices[0], prices[1])
        self.state[1] = self.calc_demand(prices[1], prices[0])
        rewards = self.state * prices
        self.episode += 1
        done = self.episode >= self.max_steps
        return self.state, rewards, done

    def reset(self):
        self.episode = 0
        self.state = self.sample_state()

    def get_surface(self):
        prices = self.get_optimal()
        res = numpy.arange(0, prices["cartel"], 0.01)
        num = res.shape[0]
        R = numpy.zeros((num, num))
        for ii in range(num):
            for jj in range(num):
                _, rew, _ = self.step([res[ii], res[jj]])
                R[ii, jj] = rew[0]
        return R

    def get_optimal(self):
        return {"nash": self.t / self.phi, "cartel": self.u - 0.5 * self.t}
