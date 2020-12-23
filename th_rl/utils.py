import pandas as pd
import os

from th_rl.agents import *
from th_rl.environments import *
from th_rl.logger import *

def load_scenarios():
    filepath = os.path.realpath(__file__)
    csvpath = os.path.join(os.path.dirname(filepath), 'scenarios.csv')
    return pd.read_csv(csvpath)


def load_experience(id,edir = r'C:\Users\niki\Source\electricity_rl\Experiments'):   
    config = load_scenarios().iloc[id].to_dict()
    config.update({'batch_size':128,'gamma':'0.995', 'dir':r'C:\Users\niki\Source\electricity_rl\Experiments','env':'PriceState'})
    globals().update(**config)
      
    if ',' in config['gamma']:
        gamma = config['gamma'].split(',')
    else:
        gamma = [config['gamma']]*nplayers

    if ',' in agent:
        agents,nact,states = [],[],[],
        for ag in agent.split(','):
            if ag=='SAC':
                nact.append(1)
            else:
                nact.append(nactions)
            if ag=='QTable':
                states.append(nstates)
            else:
                states.append(1)
            agents.append(ag)
            total_players = len(agents)
    else:
        agents = [agent]*nplayers
        nact = [nactions]*nplayers
        total_players = nplayers+greedy
        nact = nact+[1]*greedy
        states = [nstates]*nplayers

    environment = eval(env)(nactions=nact, 
                            nplayers=total_players, 
                            cost = np.zeros((total_players)), 
                            action_range = [action_min,action_max], 
                            max_steps=max_steps)

    agents = [eval(ag)(states=ns, 
                          actions=na, 
                          gamma=float(g),
                          max_state = environment.a,
                          batch_size=batch_size,
                          eps_step=1-20/epochs,
                          epsilon=eps_end,
                          eps_end=eps_end) for ag,na,ns,g in zip(agents,nact,states,gamma)]

    # Add Greedy Agents
    for _ in range(greedy):
        agents.append(GreedyContinuous(environment, agents[0].experience))    

    logdir = os.path.join(dir, str(id))
    # Load Pre-Trained models
    for i,A in enumerate(agents):
        if A.__class__.__name__ in ['QTable']:
            table = pd.read_csv(os.path.join(logdir,'model_'+str(i)+'.csv'))
            A.table = table.values
        elif not A.__class__.__name__ in ['GreedyDiscrete','GreedyContinuous']:
            state_dict = torch.load(os.path.join(logdir,'model_'+str(i)+'.pt'))
            A.load_state_dict(state_dict)       

    return environment, agents


def sample_trajectory(env, agents, init_state = []):
    labels = [a.__class__.__name__ for a in agents]
    if not init_state:
        state = env.reset()
    else:
        state = init_state
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
        R.append(reward)
    Actions = np.array(A)
    Rewards = np.array(R)
    States = np.array(S)

    for i,L in enumerate(labels):
        if L in ['TD3','SAC','GreedyContinuous']:
            Actions[:,i] = (1/(1+np.exp(-Actions[:,i])))*(env.action_range[1]-env.action_range[0])+env.action_range[0]
        else:
            Actions[:,i] = Actions[:,i]/(env.nactions[i]-1.)*(env.action_range[1]-env.action_range[0])+env.action_range[0]
    return Rewards, Actions, States
