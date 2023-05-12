# Ant Race Environment

## Overview

This repository contains the implementation of a multi-agent ant race environment based on OpenAI Gym and MuJoCo. The agents are trained using the Multi-Agent Deep Deterministic Policy Gradient (MADDPG) algorithm.

## Requirements

- Python 3.7 or later
- OpenAI Gym
- MuJoCo
- PyTorch

## Usage

To start the training process, run the following command:

```
python run.py
```

This will start the training process using the MADDPG algorithm. The agents' performance will be saved in the `results/` directory.

## Environment Details

In the ant race environment, multiple ant agents are placed in a race track. Each agent's goal is to reach the finish line as quickly as possible. The agents receive a reward based on their distance to the finish line and are penalized for falling behind the other agents.

The state space of each agent includes its own position and velocity, as well as the relative positions and velocities of the other agents. The action space consists of the forces applied by the ant's legs.

![image](https://github.com/Praniyendeev/MARL_ANT/assets/49686621/6aff35ab-3257-455a-ba91-add2a030ac8c)


## Algorithm

The agents are trained using the MADDPG algorithm, which is an extension of the DDPG algorithm for multi-agent settings. In MADDPG, each agent has its own actor network that determines its actions, and a critic network that estimates the value of the action given the full state of the environment. The critic is trained using the Bellman equation, and the actor is trained to maximize the output of the critic.

## Contributing

We welcome contributions to this project. If you have a feature request, bug report, or proposal for improvement, please open an issue on GitHub.

## License

This project is licensed under the MIT License. See the LICENSE file for details.



Citations: 
```
@misc{triplesumo,
  howpublished = {Wang, N., Das, G.P., Millard, A.G. (2022). Learning Cooperative Behaviours in Adversarial Multi-agent Systems. In: Pacheco-Gutierrez, S., Cryer, A., Caliskanelli, I., Tugal, H., Skilton, R. (eds) Towards Autonomous Robotic Systems. TAROS 2022. Lecture Notes in Computer Science(), vol 13546. Springer, Cham. https://doi.org/10.1007/978-3-031-15908-4_15} 

@misc{1606.01540,
  Author = {Greg Brockman and Vicki Cheung and Ludwig Pettersson and Jonas Schneider and John Schulman and Jie Tang and Wojciech Zaremba},
  Title = {OpenAI Gym},
  Year = {2016},
  Eprint = {arXiv:1606.01540},
}

@misc{bansal2018emergent,
      title={Emergent Complexity via Multi-Agent Competition}, 
      author={Trapit Bansal and Jakub Pachocki and Szymon Sidor and Ilya Sutskever and Igor Mordatch},
      year={2018},
      eprint={1710.03748},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}

```  
