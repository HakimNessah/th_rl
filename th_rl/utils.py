from re import I
from turtle import fillcolor
from matplotlib.pyplot import axis, xlim, ylim
import numpy
from th_rl.trainer import create_game
import os
import pandas
import click
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from tqdm import tqdm
from plotly.graph_objects import Layout

HEIGHT = 600
WIDTH = 600
WRITE = r"C:\temp"


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
        A, R = [], []
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
            R.append(reward)
            A.append(scaled_acts)
        rewards.append(R)
        actions.append(A)
    return numpy.stack(actions, axis=2), numpy.stack(rewards, axis=2)


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
        height=HEIGHT,
        width=WIDTH,
        margin=dict(r=20, b=10, l=10, t=30),
    )
    if return_fig:
        return fig
    fig.show()


def plot_qagent(
    agent,
    title="",
    field="value",
    return_fig=False,
    state_range=[0, 100],
    action_range=[0, 101],
):
    if field == "value":
        z = agent.table
    else:
        z = agent.counter

    y = numpy.arange(0, agent.states) / agent.states * agent.max_state
    x = agent.action_range[0] + agent.action_space / agent.actions * (
        agent.action_range[1] - agent.action_range[0]
    )
    return plot_matrix(
        x[action_range[0] : action_range[1]],
        y[state_range[0] : state_range[1]],
        z[action_range[0] : action_range[1], state_range[0] : state_range[1]],
        title=title,
        return_fig=return_fig,
    )


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
    fig.update_layout(
        height=HEIGHT,
        width=WIDTH,
        # title_text=title
    )
    if return_fig:
        return fig
    fig.show()
    if WRITE:
        fig.write_image(os.path.join(WRITE, title + ".svg"))


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
    fig = px.line(plotdata, height=HEIGHT, width=WIDTH, title=os.path.basename(loc))
    fig.update_yaxes(range=[10, 25])
    if return_fig:
        return fig
    fig.show()


def plot_multi_lc_area(
    loc=r"C:\Users\nikolay.tchakarov\Data\Collusion\runs",
    exp=["qtable_001", "qtable_01", "qtable_1"],
    names=["QTable 0.1%", "QTable 1%", "QTable 10%"],
    colors=[
        "rgba(50,100,220,0.5)",
        "rgba(220,50,100,0.5)",
        "rgba(50,220,100,0.5)",
    ],
    title="",
    return_fig=False,
    x=0.52,
    y=0.02,
    xlabel="",
    ylabel="",
):
    fig = go.Figure()
    for i, e in enumerate(exp):
        rewards = []
        for f in os.listdir(os.path.join(loc, e)):
            config, _, _, _, rew = load_experiment(os.path.join(loc, e, f))
            rewards.append(rew)
        rewards = numpy.stack(rewards, axis=2).sum(axis=1)
        data = numpy.percentile(a=rewards, q=[10, 50, 90], axis=-1)
        add_conf_area(fig, data, colors[i], names[i])
    fig.add_trace(
        go.Scatter(
            y=22.2222 + 0 * data[1, :],
            fill=None,
            mode="lines",
            line_color="red",
            name="Nash",
        )
    )
    fig.add_trace(
        go.Scatter(
            y=25 + 0 * data[1, :],
            fill=None,
            mode="lines",
            line_color="black",
            name="Collusion",
        )
    )
    fig.update_yaxes(range=[15, 25.1])
    fig.update_layout(
        height=HEIGHT,
        width=WIDTH,
        # title_text=title,
        title_x=0.5,
        legend=dict(y=y, x=x),
        xaxis_title=xlabel,
        yaxis_title=ylabel,
    )

    if return_fig:
        return fig
    fig.show()
    if WRITE:
        fig.write_image(os.path.join(WRITE, title + ".svg"))


def plot_learning_curve_area(
    loc, names, title="", return_fig=False, x=0.52, y=0.02, xlabel="", ylabel=""
):
    fig = go.Figure()
    rewards = []
    for f in os.listdir(loc):
        config, _, _, _, rew = load_experiment(os.path.join(loc, f))
        rewards.append(rew)

    rewards = numpy.stack(rewards, axis=2)
    rewards = numpy.concatenate([rewards, rewards.sum(axis=1, keepdims=True)], axis=1)
    data = numpy.percentile(a=rewards, q=[10, 50, 90], axis=-1)
    colors = ["rgba(60,100,220,0.5)", "rgba(200,60,200,0.5)", "rgba(120,120,120,0.5)"]
    for i in range(3):
        add_conf_area(fig, data[:, :, i], colors[i], names[i])
    fig.add_trace(
        go.Scatter(
            y=22.2222 + 0 * data[1, :, 0],
            fill=None,
            mode="lines",
            line_color="red",
            name="Nash",
        )
    )
    fig.add_trace(
        go.Scatter(
            y=25 + 0 * data[1, :, 0],
            fill=None,
            mode="lines",
            line_color="black",
            name="Collusion",
        )
    )
    fig.update_yaxes(range=[0, 25.1])
    fig.update_layout(
        height=HEIGHT,
        width=WIDTH,
        # title_text=title,
        title_x=0.5,
        legend=dict(y=y, x=x),
        xaxis_title=xlabel,
        yaxis_title=ylabel,
    )

    if return_fig:
        return fig
    fig.show()
    if WRITE:
        fig.write_image(os.path.join(WRITE, title + ".svg"))


def add_conf_area(fig, data, color, name):
    fig.add_trace(
        go.Scatter(
            y=data[1],
            fill=None,
            mode="lines",
            line_color=color,
            name=None,
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            y=data[0],
            fill=None,
            mode="lines",
            line_color=color,
            name=None,
            showlegend=False,
        ),
    )
    fig.add_trace(
        go.Scatter(
            y=data[2],
            fill="tonexty",
            mode="lines",
            fillcolor=color,
            line_color=color,
            name=name,
        )
    )
    return fig


def plot_learning_curve_sweep(loc, return_fig=False, x=0.3, y=0.02):
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
        plotdata,
        height=HEIGHT,
        width=WIDTH,
        title="Learning Curve " + os.path.basename(loc),
    )
    fig.update_yaxes(range=[10, 25])
    #  position legends inside a plot
    fig.update_layout(
        legend=dict(
            x=x,  # value must be between 0 to 1.
            y=y,  # value must be between 0 to 1.
            traceorder="normal",
            font=dict(family="sans-serif", size=10, color="black"),
        )
    )
    if return_fig:
        return fig
    fig.show()


def plot_experiment(loc, return_fig=False):
    config, agents, environment, _, _ = load_experiment(loc)
    actions, rewards = play_game(agents, environment)
    breakpoint()
    return plot_trajectory(rewards, actions, loc, return_fig)


def plot_mean_result(loc, return_fig=False):
    expi = os.listdir(loc)
    rewards, actions = 0, 0
    for exp in expi:
        config, agents, environment, _, _ = load_experiment(os.path.join(loc, exp))
        acts, rwds = play_game(agents, environment)
        rewards += rwds.mean(axis=-1)
        actions += acts.mean(axis=-1)
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
        rewards.append(rwds.mean(axis=-1).sum(axis=1))
        actions.append(acts.mean(axis=-1))
    rewards = pandas.DataFrame(data=numpy.stack(rewards, axis=0))
    rewards = rewards.ewm(halflife=5, axis=1, min_periods=0).mean()
    plotdata = pandas.DataFrame()
    plotdata["median"] = rewards.quantile(0.5, axis=0)
    plotdata["75th"] = rewards.quantile(0.75, axis=0)
    plotdata["25th"] = rewards.quantile(0.25, axis=0)
    plotdata["Nash"] = 22.22
    plotdata["Cartel"] = 25
    fig = px.line(plotdata, height=HEIGHT, width=WIDTH, title=os.path.basename(loc))
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
        print("Playing {}...".format(iloc))
        for exp in os.listdir(exp_loc):
            config, agents, environment, _, _ = load_experiment(
                os.path.join(exp_loc, exp)
            )
            acts, rwds = play_game(agents, environment)
            rewards.append(rwds.mean(axis=-1).sum(axis=1))
        rewards = numpy.stack(rewards, axis=0)
        pt = numpy.percentile(rewards, 50, axis=1)
        ptiles.append([numpy.percentile(pt, p) for p in [25, 50, 75]])
    plotdata = pandas.DataFrame(data=ptiles, columns=["25th", "median", "75th"])
    plotdata["Nash"] = 22.22
    plotdata["Cartel"] = 25
    plotdata.index = os.listdir(loc)
    fig = px.line(plotdata, height=HEIGHT, width=WIDTH, title=os.path.basename(loc))
    fig.update_yaxes(range=[10, 25])
    if return_fig:
        return fig
    fig.show()


def calc_discount_nash(discount, freq):
    return 22.22222 * (
        freq * (1 + (1 - discount) + (1 - discount) ** 2) / 3 + (1 - freq)
    )


def box_plot_sweep(
    loc=r"C:\Users\nikolay.tchakarov\Data\Collusion\runs",
    return_fig=False,
    exp=["qtable_001", "qtable_01", "qtable_1", "cac", "qtable_cac"],
    names=["QTable 0.1%", "QTable 1%", "QTable 10%", "CAC", "QTable vs CAC"],
    coll_rate_scale=True,
    title="",
    iters=1,
    x=0.01,
    y=0.01,
    xlabel="",
    ylabel="",
):
    rewards = []
    for iloc in exp:
        exp_loc = os.path.join(loc, iloc)
        exp_reward = []
        print("Playing {}...".format(iloc))

        for exp in tqdm(os.listdir(exp_loc)):
            config, agents, environment, _, _ = load_experiment(
                os.path.join(exp_loc, exp)
            )
            _, rwds = play_game(agents, environment, iters=iters)
            exp_reward.append(
                numpy.percentile(rwds.sum(axis=1), 50, axis=0, keepdims=True)
            )
        exp_reward = numpy.stack(exp_reward, axis=0)
        # Reformat to 100
        thisr = exp_reward.reshape([-1, exp_reward.shape[1]])
        if thisr.shape[0] > 100:
            thisr = thisr[:100]
        if thisr.shape[0] < 100:
            num = int(100 / thisr.shape[0]) + 1
            thisr = numpy.concatenate([thisr] * num, axis=0)
            thisr = thisr[:100]
        rewards.append(thisr)

    rewards = numpy.concatenate(rewards, axis=-1)

    if coll_rate_scale:
        ylim = [-1, 1]
        rewards = (rewards - 22.22222) / (25 - 22.22222)
    else:
        ylim = [20, 25]

    layout = Layout(plot_bgcolor="rgba(0,0,0,0)")

    fig = go.Figure(layout=layout)
    [
        fig.add_trace(
            go.Box(
                y=rewards[:, i],
                name=name,
                fillcolor="rgb(180,180,180)",
                marker={"color": "rgb(50,50,50)", "size": 1},
            )
        )
        for i, name in enumerate(names)
    ]
    fig.update_yaxes(
        range=ylim,
        linecolor="black",
        showgrid=True,
        gridwidth=1,
        linewidth=1,
        gridcolor="rgb(150,150,150)",
        zerolinecolor="black",
    )
    fig.update_xaxes(
        linecolor="black",
        showgrid=True,
        gridwidth=1,
        linewidth=1,
        gridcolor="rgb(150,150,150)",
    )
    fig.update_layout(
        height=HEIGHT,
        width=WIDTH,
        # title_text=title,
        title_x=0.5,
        # legend=dict(x=x, y=y),
        showlegend=False,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
    )
    if return_fig:
        return fig
    fig.show()
    if WRITE:
        fig.write_image(os.path.join(WRITE, title + ".svg"))


def box_plot_player(
    loc=r"C:\Users\nikolay.tchakarov\Data\Collusion\runs",
    return_fig=False,
    exp="qtable_cac",
    names=["QTable", "CAC"],
    title="",
    iters=1,
    x=0.6,
    y=0.02,
    xlabel="",
    ylabel="",
):
    exp_loc = os.path.join(loc, exp)
    exp_reward = []
    layout = Layout(plot_bgcolor="rgba(0,0,0,0)")

    for exp in tqdm(os.listdir(exp_loc)):
        config, agents, environment, _, _ = load_experiment(os.path.join(exp_loc, exp))
        _, rwds = play_game(agents, environment, iters=iters)
        exp_reward.append(numpy.percentile(rwds, 50, axis=0))

    exp_reward = numpy.stack(exp_reward, axis=0)
    rewards = exp_reward.reshape([-1, exp_reward.shape[1]])

    ylim = [7, 15]
    fig = go.Figure(layout=layout)
    [
        fig.add_trace(
            go.Box(
                y=rewards[:, i],
                name=name,
                fillcolor="rgb(180,180,180)",
                marker={"color": "rgb(50,50,50)", "size": 1},
            )
        )
        for i, name in enumerate(names)
    ]

    fig.update_yaxes(
        range=ylim,
        linecolor="black",
        showgrid=True,
        gridwidth=1,
        linewidth=1,
        gridcolor="rgb(150,150,150)",
        zerolinecolor="black",
    )
    fig.update_xaxes(
        linecolor="black",
        showgrid=True,
        gridwidth=1,
        linewidth=1,
        gridcolor="rgb(150,150,150)",
    )
    fig.update_layout(
        height=HEIGHT,
        width=WIDTH,
        # title_text=title,
        title_x=0.5,
        # legend=dict(x=x, y=y),
        showlegend=False,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
    )

    if return_fig:
        return fig
    fig.show()
    if WRITE:
        fig.write_image(os.path.join(WRITE, title + ".svg"))


@click.command()
@click.option("--dir", help="Experiment dir", type=str)
@click.option("--fun", default="plot_mean_result", help="Experiment dir", type=str)
def main(**params):
    eval(params["fun"])(params["dir"])


if __name__ == "__main__":
    main()
