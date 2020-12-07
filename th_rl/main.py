import click
import time
import random
import string

from th_rl.agents import *
from th_rl.environments import *
from th_rl.logger import *

@click.command()
@click.option('--agent', default='ActorCritic', help='Agent', type=str)
@click.option('--env', default='PriceState', help='Environment', type=str)
@click.option('--dir', default=r'C:\Users\niki\Source\electricity_rl\Experiments', help='experiment save location', type=str)
@click.option('--epochs', default=500, help='training epochs', type=int)
@click.option('--max_steps', default=100, help='environment trajectory length.', type=int)
@click.option('--greedy', default=0, help='number of greedy agents.',type=int)
@click.option('--gamma', default=0.995, help='gamma', type=float)
@click.option('--nactions', default=100, help='number of actions',type=int)
@click.option('--nstates', default=1, help='number of actions',type=int)
@click.option('--nplayers', default=3, help='number of players',type=int)
@click.option('--action_min', default=0., help='min action',type=float)
@click.option('--action_max', default=1., help='max action',type=float)
@click.option('--load', default=0, help='Load pre-trained agents',type=int)
@click.option('--print_freq', default=20, help='Print Frequency',type=int)
@click.option('--batch_size', default=20, help='Print Frequency',type=int)
@click.option('--id', default=''.join(random.choices(string.ascii_lowercase,k=8)) , help='scenario ID',type=str)
def train(**config):
    globals().update(config)
    
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
                          epsilon=1.) for ag,na in zip(agents,nact)]
    
    # Add Greedy Agents
    for _ in range(greedy):
        assert nprices==1, 'Greedy not implemented in discrete price space'
        if nactions>1:
            agents.append(GreedyDiscrete(environment, agents[0].experience))
        else:
            agents.append(GreedyContinuous(environment, agents[0].experience))

    log = Logger(dir, config, id)
    if load:
        # Load Pre-Trained models
        for i,A in enumerate(agents):
            if not A.__class__.__name__ in ['GreedyDiscrete','GreedyContinuous']:
                state_dict = torch.load(os.path.join(log.dir,'model_'+str(i)+'.pt'))
                A.load_state_dict(state_dict)       

    scores, episodes, t = [], [], time.time()
    for e in range(epochs):
        done = False
        state = environment.reset()
        ep_r = 0    
        while not done:
            # choose actions and step through environment
            action = [agent.sample_action(torch.from_numpy(state).float()) for agent in agents]
            if agent in ['PPO','Reinforce']:
                prob  = [a[1] for a in action]                
                action = [a[0] for a in action]                
            next_state, reward, welfare, done = environment.step(action)

            # save transition to the replay memory                 
            if agent=='Reinforce':
                for A,r,a,p in zip(agents, reward, action, prob):          
                    A.memory.append(r,a,p)                  
            elif agent=='PPO':
                for A,r,a,p in zip(agents, reward, action, prob):          
                    A.memory.append(state,a,r,not done,next_state,p)  
            else:                    
                for A,r,a in zip(agents, reward, action):
                    A.memory.append(state,a,r,not done,next_state)
            state = next_state
            ep_r += sum(reward)
        scores.append(ep_r)
        episodes.append(e)   
        # Train
        [A.train_net() for A in agents]

        if not (e%print_freq):
            tr = np.mean(scores[-print_freq:])
            sr = tr/environment.max_steps
            if agents[0].__class__.__name__=='QTable':
                print("time:{:2.2f} | episode:{:3d} | mean trajectory reward:{:2.2f} | mean step reward:{:2.2f} | epsilon:{:0.4f}".format(
                    time.time()-t,e,tr,sr,agents[0].epsilon))
            else:
                print("time:{:2.2f} | episode:{:3d} | mean trajectory reward:{:2.2f} | mean step reward:{:2.2f}".format(
                    time.time()-t,e,tr,sr))

            t = time.time()

        #if np.mean(scores[-10:])/environment.max_steps > 24.5:
        #    break # Collusion was found

    # Save results
    logdata = np.stack([np.array(x) for x in [episodes, scores]],axis=1)
    logdata = pd.DataFrame(data=logdata, columns = ['Episodes', 'Scores'])
    if load:
        log.append(logdata)
    else:
        log.log(logdata)

    # Save figures
    log.plot_state_sweep(environment, agents)
    log.plot_trajectory(environment, agents)
    log.plot_learning(environment, scores, 'Total Step Reward')

    # Save trained models
    for i,A in enumerate(agents):
        if not A.__class__.__name__ in ['GreedyDiscrete','GreedyContinuous','QTable']:
            torch.save(A.state_dict(), os.path.join(log.dir,'model_'+str(i)+'.pt'))
        if A.__class__.__name__=='QTable':
            T = pd.DataFrame(A.table)
            T.to_csv(os.path.join(log.dir,'model_'+str(i)+'.csv'))

if __name__=='__main__':
    train()
    