# th_rl
Pytorch-based package for multi-agent reinforcement learning in an iterated prisonner dilemma setting.

<br> Application to electricity/power market dynamics.
<br> All experiments stored in https://github.com/nikitcha/electricity_rl

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