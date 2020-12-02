import pandas as pd
import os

scenarios = pd.read_excel(r'C:\Users\niki\Source\th_rl\th_rl\scenarios.xlsx')

for i in range(scenarios.shape[0]):
    scen = scenarios.iloc[i,:]
    command = 'python -m th_rl.main --id='+str(i)
    for s in scen.index:
        opt = ' --'+s+'='+str(scen[s])
        command += opt

    print(command)
    os.system(command)
