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
@click.option('--nplayers', default=3, help='number of players',type=int)
@click.option('--action_min', default=0., help='min action',type=float)
@click.option('--action_max', default=1., help='max action',type=float)
@click.option('--encoder', default='none', help='Action Encoder',type=str)
@click.option('--load', default=0, help='Load pre-trained agents',type=int)
@click.option('--print_freq', default=20, help='Print Frequency',type=int)
@click.option('--id', default=''.join(random.choices(string.ascii_lowercase,k=8)) , help='scenario ID',type=str)
def train(**config):
    globals().update(config)
    total_players = nplayers+greedy
    if agent in ['SAC', 'DDPG']:
        actions = 1
        discrete = False
        action_range = [0,1]
    else:
        actions = nactions
        discrete = True
        action_range = [action_min,action_max]

    environment = eval(env)(nactions=nactions, 
                            nplayers=total_players, 
                            cost = np.zeros((total_players)), 
                            action_range = action_range, 
                            max_steps=max_steps, 
                            encoder=encoder,
                            discrete=discrete)

    agents = [eval(agent)(states=environment.encode().shape[-1], 
                          actions=actions, 
                          gamma=gamma) for _ in range(config['nplayers'])]
    
    # Add Greedy Agents
    for _ in range(greedy):
        if discrete:
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
                    A.memory.append(r/10,a,p)                  
            elif agent=='PPO':
                for A,r,a,p in zip(agents, reward, action, prob):          
                    A.memory.append(state,a,r/10,not done,next_state,p)  
            else:                    
                for A,r,a in zip(agents, reward, action):
                    A.memory.append(state,a,r/10,not done,next_state)              
            state = next_state
            ep_r += sum(reward)
        scores.append(ep_r)
        episodes.append(e)   
        
        # Train
        [A.train_net() for A in agents]

        if not (e%print_freq):
            tr = np.mean(scores[-print_freq:])
            sr = tr/environment.max_steps
            print("time:{:2.2f} | episode:{:3d} | mean trajectory reward:{:2.2f} | mean step reward:{:2.2f}".format(time.time()-t,e,tr,sr))
            t = time.time()  

    # Save results
    logdata = np.stack([np.array(x) for x in [episodes, scores]],axis=1)
    logdata = pd.DataFrame(data=logdata, columns = ['Episodes', 'Scores'])
    if load:
        log.append(logdata)
    else:
        log.log(logdata)

    # Save figures
    log.plot_state_sweep(environment, agents)

    # Save trained models
    for i,A in enumerate(agents):
        if not A.__class__.__name__ in ['GreedyDiscrete','GreedyContinuous']:
            torch.save(A.state_dict(), os.path.join(log.dir,'model_'+str(i)+'.pt'))


if __name__=='__main__':
    train()
    