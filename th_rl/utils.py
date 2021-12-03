import numpy
from th_rl.trainer import create_game
import os
import pandas
import click

import plotly.graph_objects as go
from plotly.subplots import make_subplots

def load_experiment(loc):
    cpath  = os.path.join(loc, 'config.json')
    config, agents, environment = create_game(cpath)
    for i,agent in enumerate(agents):
        agent.load(os.path.join(loc,str(i)))
    return config, agents, environment

def play_game(agents, environment, iters=1):
    rewards,actions = [],[]
    for i in range(iters):
        done = False
        state = environment.reset()
        next_state = state
        while not done:
            # choose actions 
            acts = [ agent.get_action(next_state) for agent in agents]
            #acts = [ agent.sample_action(torch.from_numpy(next_state).float()) for agent in agents]
            scaled_acts = [agent.scale(act) for agent, act in zip(agents, acts)]
            
            # Step through environment
            next_state, reward, done = environment.step(scaled_acts)
            rewards.append(reward)
            actions.append(scaled_acts)

    return numpy.array(actions), numpy.array(rewards)

def plot_matrix(x,y,z,title='',xlabel='Actions',ylabel='States',zlabel='Values'):
    fig = go.Figure()
    fig.add_trace(go.Surface(z=z, x=x,y=y))
    fig.update_layout(scene = dict(
                        xaxis_title=xlabel,
                        yaxis_title=ylabel,
                        zaxis_title=zlabel),
                        title=title,
                        width=700,
                        height = 600,
                        margin=dict(r=20, b=10, l=10, t=30))
    fig.show()

def plot_qagent(agent, title='',field='value'):
    if field=='value':
        z = agent.table
    else:
        z = agent.counter
    
    y = numpy.arange(0,agent.states)/agent.states*agent.max_state
    x = agent.action_range[0]+agent.action_space/agent.actions*(agent.action_range[1]-agent.action_range[0])
    plot_matrix(x,y,z,title=title)

def plot_trajectory(actions, rewards, title=''):
    rpd = pandas.DataFrame(data = rewards, columns=numpy.arange(actions.shape[1]))
    apd = pandas.DataFrame(data = actions, columns=numpy.arange(actions.shape[1]))
    rpd['Total'] = rpd.sum(axis=1)
    rpd['Nash'] = 22.22
    rpd['Cartel'] = 25

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,vertical_spacing=0.1, subplot_titles=("Rewards", "Actions"))
    for col in rpd.columns:
        fig.add_trace(go.Scatter(x=rpd.index.values, y=rpd[col].values, name='Reward {}'.format(col)),row=1, col=1)
    for col in apd.columns:
        fig.add_trace(go.Scatter(x=rpd.index.values, y=rpd[col].values, name='Action {}'.format(col)),row=2, col=1)
    fig.update_layout(height=600, width=600, title_text=title)
    fig.show()    


def plot_experiment(loc):
    config, agents, environment = load_experiment(loc)
    rewards, actions = play_game(agents, environment)
    plot_trajectory(rewards,actions,loc)

def plot_mean_result(loc):
    expi = os.listdir(loc)
    rewards, actions = 0,0
    for exp in expi:
        config, agents, environment = load_experiment(os.path.join(loc,exp))
        acts, rwds = play_game(agents, environment)   
        rewards += rwds
        actions += acts
    plot_trajectory(actions/len(expi), rewards/len(expi), title=loc)

def plot_visits(loc):
    config, agents, environment = load_experiment(loc)
    [plot_qagent(a,loc, 'counter') for a in agents]

def plot_values(loc):
    config, agents, environment = load_experiment(loc)
    [plot_qagent(a,loc, 'counter') for a in agents]

@click.command()
@click.option('--dir', help='Experiment dir', type=str)
@click.option('--fun', default='plot_mean_result',help='Experiment dir', type=str)

def main(**params):
    eval(params['fun'])(params['dir'])

if __name__=='__main__':
    main()
