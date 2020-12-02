import pandas as pd
import os
import click

@click.command()
@click.option('--scenariofile', default=r'C:\Users\niki\Source\th_rl\th_rl\scenarios.csv', help='Location of the scenario file', type=str)
@click.option('--start', default=0, help='Start index', type=int)
def run(**params):
    globals().update(params)
    scenarios = pd.read_csv(scenariofile)
    for i in range(scenarios.shape[0]):
        scen = scenarios.iloc[i,:]
        command = 'python -m th_rl.main --id='+str(i)
        for s in scen.index:
            opt = ' --'+s+'='+str(scen[s])
            command += opt

        if i>=start:
            print(command)
            os.system(command)

if __name__=='__main__':
    run()