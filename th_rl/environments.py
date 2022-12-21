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


class Bertrand:
    def __init__(self, nplayers=2, max_steps=100, **kwargs):
        self.nplayers = nplayers
        self.max_steps = max_steps
        self.episode = 0
        self.demand = torch.tensor([0, 0]).type(torch.float)
        assert nplayers == 2, "Env is only defined for exactly two players"

    def calc_demand(self, p):
        if p[0] < p[1] and p[0] <= 1:
            return [1, 0]
        elif p[0] > p[1] and p[1] <= 0:
            return [0, 1]
        elif p[0] == p[1] and p[0] <= 1:
            return [0.5, 0.5]
        else:
            return [0, 0]

    def step(self, prices):
        demand = self.calc_demand(prices)
        self.demand[0] = demand[0]
        self.demand[1] = demand[1]

        rewards = self.demand * prices
        self.episode += 1
        done = self.episode >= self.max_steps
        return self.demand, rewards, done

    def reset(self):
        self.episode = 0


class NoisyCournot:
    def __init__(
        self,
        nplayers,
        action_range=[0, 1],
        a=10,
        b=1,
        max_steps=100,
        noise_prob=0.0,
        **kwargs
    ):
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

    def step(self, actions):
        A = numpy.array(actions)
        Q = sum(A)
        if numpy.random.uniform(0, 1) < self.noise_prob:
            new_a = numpy.random.uniform(self.a * 0.7, self.a)
        else:
            new_a = self.a
        price = numpy.max([0, new_a - self.b * Q])

        rewards = price * A

        self.state = price
        self.episode += 1
        done = self.episode >= self.max_steps
        return self.encode(), rewards, done

    def get_optimal(self):
        anash = (
            (self.a / self.b)
            * numpy.ones(
                self.nplayers,
            )
            / (self.nplayers + 1)
        )
        price = numpy.max([0, self.a - self.b * sum(anash)])
        rnash = [price * a for a in anash]
        acoll = (
            (self.a / self.b)
            * 0.5
            * numpy.ones(
                self.nplayers,
            )
            / self.nplayers
        )
        price = numpy.max([0, self.a - self.b * sum(acoll)])
        rcoll = [price * a for a in acoll]
        return {
            "nash": {"action": anash, "reward": rnash},
            "cartel": {"action": acoll, "reward": rcoll},
        }

    def reset(self):
        self.episode = 0
        self.state = self.sample_state()
        return self.encode()
