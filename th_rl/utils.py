import numpy
from th_rl.trainer import create_game
import os
import pandas
import click
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


def load_experiment(loc):
    cpath = os.path.join(loc, "config.json")
    config, agents, environment = create_game(cpath)
    for i, agent in enumerate(agents):
        agent.load(os.path.join(loc, str(i)))
    log = pandas.read_csv(os.path.join(loc, "log.csv"))
    names = [a["name"] + str(i) for i, a in enumerate(config["agents"])]

    rewards = log[["rewards", "rewards.1"]].ewm(halflife=1000).mean()
    actions = log[["actions", "actions.1"]].ewm(halflife=1000).mean()
    rewards.columns = names
    actions.columns = names
    return config, agents, environment, actions, rewards


def play_game(agents, environment, iters=1):
    rewards, actions = [], []
    for i in range(iters):
        done = False
        state = environment.reset()
        next_state = state
        while not done:
            # choose actions
            acts = [agent.get_action(next_state) for agent in agents]
            # acts = [
            #    agent.sample_action(torch.from_numpy(next_state).float())
            #    for agent in agents
            # ]
            scaled_acts = [agent.scale(act) for agent, act in zip(agents, acts)]

            # Step through environment
            next_state, reward, done = environment.step(scaled_acts)
            rewards.append(reward)
            actions.append(scaled_acts)

    return numpy.array(actions), numpy.array(rewards)


def plot_matrix(
    x,
    y,
    z,
    title="",
    xlabel="Actions",
    ylabel="States",
    zlabel="Values",
    return_fig=False,
):
    fig = go.Figure()
    fig.add_trace(go.Surface(z=z, x=x, y=y))
    fig.update_layout(
        scene=dict(xaxis_title=xlabel, yaxis_title=ylabel, zaxis_title=zlabel),
        title=title,
        width=700,
        height=600,
        margin=dict(r=20, b=10, l=10, t=30),
    )
    if return_fig:
        return fig
    fig.show()


def plot_qagent(agent, title="", field="value", return_fig=False):
    if field == "value":
        z = agent.table
    else:
        z = agent.counter

    y = numpy.arange(0, agent.states) / agent.states * agent.max_state
    x = agent.action_range[0] + agent.action_space / agent.actions * (
        agent.action_range[1] - agent.action_range[0]
    )
    return plot_matrix(x, y, z, title=title, return_fig=return_fig)


def plot_trajectory(actions, rewards, title="", return_fig=False):
    rpd = rewards
    apd = actions
    rpd["Total"] = rpd.sum(axis=1)
    rpd["Nash"] = 22.22
    rpd["Cartel"] = 25

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Rewards", "Actions"),
    )
    for col in rpd.columns:
        fig.add_trace(
            go.Scatter(
                x=rpd.index.values, y=rpd[col].values, name="Reward {}".format(col)
            ),
            row=1,
            col=1,
        )
    for col in apd.columns:
        fig.add_trace(
            go.Scatter(
                x=rpd.index.values, y=apd[col].values, name="Action {}".format(col)
            ),
            row=2,
            col=1,
        )
    fig.update_layout(height=600, width=600, title_text=title)
    if return_fig:
        return fig
    fig.show()


def plot_learning_curve(loc, return_fig=False):
    config, agents, environment, actions, rewards = load_experiment(loc)
    fig = plot_trajectory(
        actions,
        rewards,
        title=os.path.basename(loc),
        return_fig=return_fig,
    )
    return fig


def plot_learning_curve_conf(loc, return_fig=False):
    rewards = []
    for f in os.listdir(loc):
        log = pandas.read_csv(os.path.join(loc, f, "log.csv"))
        rewards.append(
            log[["rewards", "rewards.1"]].ewm(halflife=1000).mean().sum(axis=1)
        )
    rewards = pandas.concat(rewards, axis=1)
    plotdata = pandas.DataFrame()
    plotdata["median"] = rewards.quantile(0.5, axis=1)
    plotdata["75th"] = rewards.quantile(0.75, axis=1)
    plotdata["25th"] = rewards.quantile(0.25, axis=1)
    plotdata["Nash"] = 22.22
    plotdata["Cartel"] = 25
    fig = px.line(plotdata, width=500, height=500, title=os.path.basename(loc))
    fig.update_yaxes(range=[10, 25])
    if return_fig:
        return fig
    fig.show()


def plot_learning_curve_sweep(loc, return_fig=False):
    plotdata = pandas.DataFrame()
    for e in os.listdir(loc):
        rewards = []
        for f in os.listdir(os.path.join(loc, e)):
            log = pandas.read_csv(os.path.join(loc, e, f, "log.csv"))
            rewards.append(
                log[["rewards", "rewards.1"]].ewm(halflife=1000).mean().sum(axis=1)
            )
        rewards = pandas.concat(rewards, axis=1)
        plotdata[e + "-median"] = rewards.quantile(0.5, axis=1)
        # plotdata[e+'-75th'] = rewards.quantile(0.75,axis=1)
        # plotdata[e+'-25th'] = rewards.quantile(0.25,axis=1)
    plotdata["Nash"] = 22.22
    plotdata["Cartel"] = 25
    fig = px.line(
        plotdata, width=500, height=500, title="Learning Curve " + os.path.basename(loc)
    )
    fig.update_yaxes(range=[10, 25])
    #  position legends inside a plot
    fig.update_layout(
        legend=dict(
            x=0.3,  # value must be between 0 to 1.
            y=0.02,  # value must be between 0 to 1.
            traceorder="normal",
            font=dict(family="sans-serif", size=10, color="black"),
        )
    )
    if return_fig:
        return fig
    fig.show()


def plot_experiment(loc, return_fig=False):
    config, agents, environment, _, _ = load_experiment(loc)
    rewards, actions = play_game(agents, environment)
    return plot_trajectory(rewards, actions, loc, return_fig)


def plot_mean_result(loc, return_fig=False):
    expi = os.listdir(loc)
    rewards, actions = 0, 0
    for exp in expi:
        config, agents, environment, _, _ = load_experiment(os.path.join(loc, exp))
        acts, rwds = play_game(agents, environment)
        rewards += rwds
        actions += acts
    return plot_trajectory(
        actions / len(expi),
        rewards / len(expi),
        title=os.path.basename(loc),
        return_fig=return_fig,
    )


def plot_mean_conf(loc, return_fig=False):
    expi = os.listdir(loc)
    rewards, actions = [], []
    for exp in expi:
        config, agents, environment, _, _ = load_experiment(os.path.join(loc, exp))
        acts, rwds = play_game(agents, environment)
        rewards.append(numpy.sum(rwds, axis=1))
        actions.append(acts)
    rewards = pandas.DataFrame(data=numpy.stack(rewards, axis=0))
    rewards = rewards.ewm(halflife=5, axis=1, min_periods=0).mean()
    plotdata = pandas.DataFrame()
    plotdata["median"] = rewards.quantile(0.5, axis=0)
    plotdata["75th"] = rewards.quantile(0.75, axis=0)
    plotdata["25th"] = rewards.quantile(0.25, axis=0)
    plotdata["Nash"] = 22.22
    plotdata["Cartel"] = 25
    fig = px.line(plotdata, width=500, height=500, title=os.path.basename(loc))
    fig.update_yaxes(range=[10, 25])
    if return_fig:
        return fig
    fig.show()


def plot_visits(loc, return_fig=False):
    config, agents, environment, _, _ = load_experiment(loc)
    return [plot_qagent(a, loc, "counter", return_fig=return_fig) for a in agents]


def plot_values(loc, return_fig=False):
    config, agents, environment, _, _ = load_experiment(loc)
    return [plot_qagent(a, loc, "value", return_fig=return_fig) for a in agents]


def plot_sweep_conf(loc, return_fig=False):
    ptiles = []
    for iloc in os.listdir(loc):
        exp_loc = os.path.join(loc, iloc)
        rewards = []
        for exp in os.listdir(exp_loc):
            config, agents, environment, _, _ = load_experiment(os.path.join(exp_loc, exp))
            acts, rwds = play_game(agents, environment)
            rewards.append(numpy.sum(rwds, axis=1))
        rewards = numpy.stack(rewards, axis=0)
        pt = numpy.percentile(rewards, 50, axis=1)
        ptiles.append([numpy.percentile(pt, p) for p in [25, 50, 75]])
    plotdata = pandas.DataFrame(data=ptiles, columns=["25th", "median", "75th"])
    plotdata["Nash"] = 22.22
    plotdata["Cartel"] = 25
    plotdata.index = os.listdir(loc)
    fig = px.line(plotdata, width=500, height=500, title=os.path.basename(loc))
    fig.update_yaxes(range=[10, 25])
    if return_fig:
        return fig
    fig.show()


def calc_discount_nash(discount, freq):
    return 22.22222 * (
        freq * (1 + (1 - discount) + (1 - discount) ** 2) / 3 + (1 - freq)
    )


@click.command()
@click.option("--dir", help="Experiment dir", type=str)
@click.option("--fun", default="plot_mean_result", help="Experiment dir", type=str)
def main(**params):
    eval(params["fun"])(params["dir"])


if __name__ == "__main__":
    main()
