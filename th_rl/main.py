import click
import os

from th_rl.trainer import train_one

@click.command()
@click.option('--runs', default=2, help='Runs per config', type=int)
@click.option('--cdir', help='Configs dir', type=str)
def main(**params):
    home = os.path.join(os.path.abspath(params['cdir']),'..','runs')
    if not os.path.exists(home):
        os.mkdir(home)

    for confname in os.listdir(params['cdir']):
        cpath = os.path.join(home, confname.replace('.json',''))
        if not os.path.exists(cpath):
            os.mkdir(cpath)
        for i in range(params['runs']):
            exp_path = os.path.join(cpath, str(i))
            train_one(exp_path, os.path.join(params['cdir'],confname))

if __name__=='__main__':
    main()
    