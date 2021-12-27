import sys
from streamlit import cli as stcli
import streamlit as st
import streamlit
from th_rl import utils

def main():
    plots = {
        "Plot Single Learning Curve":utils.plot_learning_curve,
        "Plot Learning Curve Confidence":utils.plot_learning_curve_conf,
        "Plot Learning Curve Sweep":utils.plot_learning_curve_sweep,
        "Plot Experiment":utils.plot_experiment,
        "Plot Mean Result":utils.plot_mean_result,
        "Plot Mean Result with Confidence":utils.plot_mean_conf,
        "Plot Sweep Result with Confidence":utils.plot_sweep_conf,
        "Plot QTable Values":utils.plot_values,
        "Plot QTable Visits":utils.plot_visits,
    }

    for k,f in plots.items():
        with st.expander(k):
            loc = st.text_input("Path to experiment",key=k)
            if loc:
                try:
                    fig=f(loc, return_fig=True)
                    if type(fig)==list:
                        for g in fig:
                            st.plotly_chart(g)
                    else:
                        st.plotly_chart(fig)
                except:
                    streamlit.write("Incorrect folder structure for plot")


if __name__ == '__main__':
    if st._is_running_with_streamlit:
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())