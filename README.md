# th_rl
Pytorch Reinforcement Learning agents in an iterated prisonner dilemma setting.
Application to electricity/power market dynamics.
All experiments stored in https://github.com/nikitcha/electricity_rl

## Agents
- Discrete:
    - QTable
    - DQN
    - Actor Critic
    - PPO
- Continuous    
    - SAC
    - TD3

## Multi-agent environments
- State-Price (v)
    - Discrete or continuous action spaces
    - Discrete or continuous state (price) space
    - Linear price as function of production
- State-Action (x)
    - Not implemented

## Buffers
- Standard replay buffer 