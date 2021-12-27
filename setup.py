from setuptools import setup

setup(name='th_rl',
      version='0.1',
      description='RL Repo in Pytorch',
      url='http://github.com/ntchakarov/th_rl',
      author='Nikolay Tchakarov',
      license='MIT',
      packages=['th_rl'],
      install_requires=[
        "click",
        "numpy",
        "pandas",
        "plotly",
        "torch",
        "streamlit"
      ],      
      zip_safe=False)