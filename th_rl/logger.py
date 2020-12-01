import os
from pathlib import Path
import pandas as pd
import random
import string
import matplotlib.pyplot as plt
import torch 
import numpy as np

class Logger():
    def __init__(self, loc, config, id):    
        self.id = id
        self.dir = os.path.join(loc, self.id)
        Path(self.dir).mkdir(parents=True, exist_ok=True)
        self.loc = os.path.join(self.dir, 'result.csv')
        D = config.copy()
        D.update({'id':self.id})
        c = pd.DataFrame.from_dict(D, orient='index')
        c.to_csv(os.path.join(loc, 'dashboard.csv'), mode='a')
    
    def log(self, df):
        df.to_csv(self.loc)

    def append(self, df):
        df.to_csv(self.loc, mode='a', header=False)

    def plot_state_sweep(self, env, agents):
        labels = [A.__class__.__name__ for A in agents]
        scenarios = np.linspace(0,10,40)
        Actions, Rewards = [],[]
        for i in range(scenarios.shape[0]):
            action, reward = [],[]
            for j in range(100):
                act = [agent.sample_action(torch.from_numpy(scenarios[[i]]).float()) for agent in agents]
                if labels[0] in ['PPO','Reinforce']:
                    act = [a[0] for a in act]
                env.reset()
                _, r, _, _ = env.step(act)
                action.append(np.array(act))
                reward.append(np.array(r))
            Rewards.append(np.array(reward))
            Actions.append(np.array(action))
        Actions = np.array(Actions)
        Rewards = np.array(Rewards)

        f,ax = plt.subplots(1,2,figsize=(12,5))
        mi = np.percentile(Actions,5,axis=1)
        ma = np.percentile(Actions,95,axis=1)
        ax[0].plot(scenarios, np.mean(Actions,axis=1))
        ax[0].fill_between(scenarios, mi[:,0], ma[:,0], alpha=.2)
        ax[0].fill_between(scenarios, mi[:,1], ma[:,1], alpha=.2)
        ax[0].fill_between(scenarios, mi[:,2], ma[:,2], alpha=.2)
        ri = np.percentile(Rewards,5,axis=1)
        ra = np.percentile(Rewards,95,axis=1)
        ax[1].plot(scenarios, np.mean(Rewards,axis=1))
        ax[1].fill_between(scenarios, ri[:,0], ra[:,0], alpha=.2)
        ax[1].fill_between(scenarios, ri[:,1], ra[:,1], alpha=.2)
        ax[1].fill_between(scenarios, ri[:,2], ra[:,2], alpha=.2)
        ax[1].plot(scenarios, np.sum(np.mean(Rewards,axis=1),axis=1))

        [a.grid() for a in ax]
        [a.set_title(c) for a,c in zip(ax, ['Actions','Rewards'])]
        [a.set_xlabel('Price') for a in ax]
        ax[0].legend(labels)
        _=ax[1].legend(labels+['Total'])
        plt.savefig(os.path.join(self.dir, 'state_sweep.png')) 
        plt.close()

