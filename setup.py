from setuptools import setup

setup(name='th_rl',
      version='0.1',
      description='RL Repo in Pytorch',
      url='http://github.com/ntchakarov/th_rl',
      author='Nikolay Tchakarov',
      license='MIT',
      packages=['th_rl'],
      install_requires=[
        "click==7.1.2",
        "numpy==1.19.4",
        "pandas==1.3.4",
        "plotly==5.4.0",
        "torch==1.10.0"
      ],      
      zip_safe=False)