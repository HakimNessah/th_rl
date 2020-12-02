import os
from pathlib import Path
import pandas as pd
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

    def plot_learning(self, env, values, label):
        f,ax = plt.subplots(1,1,figsize=(5,5))
        vals = np.array(values)/env.max_steps
        ax.plot(vals)
        ax.set_title(label)
        [n,c] = env.get_optimal()
        ax.plot((0,vals.shape[0]),(n,n))
        ax.plot((0,vals.shape[0]),(c,c))
        ax.legend(['Reward','Nash','Collusion'])
        ax.grid()
        plt.savefig(os.path.join(self.dir, 'learning.png')) 
        plt.close()        


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

        if labels[0] in ['TD3','SAC']:
            Actions = (1/(1+np.exp(-Actions)))*(env.action_range[1]-env.action_range[0])+env.action_range[0]

        f,ax = plt.subplots(1,2,figsize=(12,5))
        mi = np.percentile(Actions,5,axis=1)
        ma = np.percentile(Actions,95,axis=1)
        ax[0].plot(scenarios, np.mean(Actions,axis=1))
        for i in range(len(agents)):
            ax[0].fill_between(scenarios, mi[:,i], ma[:,i], alpha=.2)
        ri = np.percentile(Rewards,5,axis=1)
        ra = np.percentile(Rewards,95,axis=1)
        ax[1].plot(scenarios, np.mean(Rewards,axis=1))
        for i in range(len(agents)):
            ax[1].fill_between(scenarios, ri[:,i], ra[:,i], alpha=.2)
        ax[1].plot(scenarios, np.sum(np.mean(Rewards,axis=1),axis=1))

        [a.grid() for a in ax]
        [a.set_title(c) for a,c in zip(ax, ['Actions','Rewards'])]
        [a.set_xlabel('Price') for a in ax]
        ax[0].legend(labels)
        _=ax[1].legend(labels+['Total'])
        plt.savefig(os.path.join(self.dir, 'state_sweep.png')) 
        plt.close()

    def plot_trajectory(self, env, agents):
        labels = [A.__class__.__name__ for A in agents]

        Actions, Rewards, States = [],[],[]
        for _ in range(10):
            state = env.reset()
            state = np.array([0.])
            A, R, S = [],[],[]
            for e in range(env.max_steps):
                ep_r = 0    
                # Get probabilities
                S.append(state)
                action = [agent.sample_action(torch.from_numpy(state).float()) for agent in agents]
                if labels[0] in ['PPO','Reinforce']:
                    action = [a[0] for a in action]
                next_state, reward, welfare, done = env.step(action)
                state = next_state
                A.append(np.array(action))
                R.append(sum(reward))
                
            Actions.append(np.array(A))
            Rewards.append(np.array(R))
            States.append(np.array(S))
        
        Actions = np.stack(Actions,axis=2)
        Rewards = np.stack(Rewards,axis=1)
        States = np.stack(States,axis=2)

        if labels[0] in ['TD3','SAC']:
            Actions = (1/(1+np.exp(-Actions)))*(env.action_range[1]-env.action_range[0])+env.action_range[0]

        f,ax = plt.subplots(1,3,figsize=(16,4))
        ax[0].plot(np.mean(Actions,axis=2))
        ax[1].plot(np.mean(Rewards,axis=1))
        [n,c] = env.get_optimal()
        ax[1].plot((0,Rewards.shape[0]),(n,n))
        ax[1].plot((0,Rewards.shape[0]),(c,c))
        ax[2].plot(np.mean(States,axis=2))
        [a.grid() for a in ax]
        [a.set_title(c) for a,c in zip(ax, ['Actions','Total Reward', 'Price'])]
        [a.set_xlabel('Step') for a in ax]
        _=ax[0].legend(labels)
        plt.savefig(os.path.join(self.dir, 'trajectory.png')) 
        plt.close()