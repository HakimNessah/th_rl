import pandas as pd
import os
import click

@click.command()
@click.option('--scenariofile', default=r'C:\Users\niki\Source\th_rl\th_rl\scenarios.csv', help='Location of the scenario file', type=str)
@click.option('--start', default=0, help='Start index', type=int)
@click.option('--end', default=-1, help='End index', type=int)
def run(**config):
    scenarios = pd.read_csv(config['scenariofile'])
    if config['end']==-1:
        config['end'] = len(scenarios)
    for i in range(scenarios.shape[0]):
        scen = scenarios.iloc[i,:]
        command = 'python -m th_rl.main --id='+str(i)
        for s in scen.index:
            opt = ' --'+s+'='+str(scen[s])
            command += opt

        if i>=config['start'] and i<config['end']:
            print(command)
            os.system(command)

if __name__=='__main__':
    run()