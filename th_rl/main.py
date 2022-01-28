import click
import os

from th_rl.trainer import train_one

@click.command()
@click.option('--runs', default=1, help='Runs per config', type=int)
@click.option('--dir', default=r'C:\Users\nikolay.tchakarov\Data\Collusion\configs' ,help='Configs dir', type=str)
def main(**params):
    home = os.path.join(os.path.abspath(params['dir']),'..','runs')
    if not os.path.exists(home):
        os.mkdir(home)
    for confname in os.listdir(params['dir']):
        if '.json' in confname:
            cpath = os.path.join(home, confname.replace('.json',''))
            if confname.replace('.json','') not in os.listdir(home):
                if not os.path.exists(cpath):
                    os.mkdir(cpath)
                for i in range(params['runs']):
                    exp_path = os.path.join(cpath, str(i))
                    train_one(exp_path, os.path.join(params['dir'],confname))
                else:
                    print('Skipping {}'.format(confname))

if __name__=='__main__':
    main()
    