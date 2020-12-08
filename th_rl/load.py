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
    #scen = pd.read_csv(os.path.join(edir,'dashboard.csv'))
    #scen.columns = ['param','value']
    #six = scen.query("param=='id' & value=='0'").index.values[-1]
    #eix = np.argmax(pd.isnull(scen.iloc[six:]['param']).values)
    #scen = scen.iloc[six:six+eix].set_index('param').rename(columns={'value':''})
    #config = scen.to_dict()['']

    config = load_scenarios().iloc[id].to_dict()
    config.update({'batch_size':128,'gamma':0.995, 'dir':r'C:\Users\niki\Source\electricity_rl\Experiments','env':'PriceState'})
    globals().update(**config)
      
    if ',' in agent:
        agents = []
        nact = []
        for ag in agent.split(','):
            assert ag in ['SAC','ActorCritic'], 'Agent '+ag+' not implemented!'
            if ag=='SAC':
                nact.append(1)
            else:
                nact.append(nactions)
            agents.append(ag)
            total_players = len(agents)
            assert greedy==0, 'Greedy not implemented in mixed agents'
    else:
        agents = [agent]*nplayers
        nact = [nactions]*nplayers
        total_players = nplayers+greedy

    environment = eval(env)(nactions=nact, 
                            nstates=nstates, 
                            nplayers=total_players, 
                            cost = np.zeros((total_players)), 
                            action_range = [action_min,action_max], 
                            max_steps=max_steps)

    agents = [eval(ag)(states=environment.encode().shape[-1], 
                          actions=na, 
                          gamma=gamma,
                          batch_size=batch_size,
                          eps_step=1-10/epochs,
                          epsilon=1.,
                          eps_end=eps_end) for ag,na in zip(agents,nact)]
    
    # Add Greedy Agents
    for _ in range(greedy):
        assert nprices==1, 'Greedy not implemented in discrete price space'
        if nactions>1:
            agents.append(GreedyDiscrete(environment, agents[0].experience))
        else:
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