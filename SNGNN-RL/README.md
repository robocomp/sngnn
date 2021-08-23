# SNGNN-RL

SNGNN-RL: Reinforcement Learning based Path Planning Policy for Social Settings

## Introduction

Mobile robots that come across people on a daily basis must react to them in an appropriate way. While conventional control algorithms treat all sensor readings as objects, however the recent trend argues that robots that operate around people should react socially to those people, following the same social conventions that people use around each other. In order to achieve this task, one cannot really rely on hand-engineered heuristics for path planning algorithms. With the advent of Machine Learning, one can learn such heuristics and use them as cost functions in conventional path planners. Social Navigation Graph Neural Network - 1D (SNGNN-1D) is one such Graph Neural Network based Machine Learning algorithm which evaluates the static scenes generated randomly from the point of view of the robot. In short, it outputs a score between 0 and 1 which shows how good the robot’s position is with respect to its surroundings. This score, for every state in the room, is used to generate a heat map that is used by A* algorithm to plan a path to the goal position. The problem with this implementation is that it is computationally expensive and is relatively slow, also SNGNN has been trained using static scenes and does not factor in dynamic settings for example, walking humans, which is a very common scenario in a social setting. A more cost effective way to carry out this task would be to train a policy to reach the goal. However, it is difficult to structure a reward function for complex environments (social settings). Hence to overcome these problems, we made use of a Reinforcement Learning algorithm (DUELING DQN) that made use of SNGNN-2D as a reward function to learn a policy that would allow a robot to move around in social settings by causing minimal to no disturbance.


## Software requirements
1. PyTorch [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
2. DGL [https://www.dgl.ai/pages/start.html](https://www.dgl.ai/pages/start.html)
3. PyTorch Geometric [https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
4. Webots [Installation Procedure](https://cyberbotics.com/doc/guide/installation-procedure)
5. ZerocIce [Installation Procedure](https://doc.zeroc.com/ice/3.7/release-notes/using-the-python-distribution)

# Running the repo

After cloning the directory, execute the following commands:
1. Build the controller.
    ```
   cd controllers/simulator_supervisor_cpp
   cmake . && make
    ```
2. Run the simulator and the algorithm. You might want to open 2 terminals to do the same
    TERMINAL 1
    ```
    webots
    ```
    TERMINAL 2
    ```
    cd agent/
    python3 agent.py --play=models/third_try_dueling_dqn_20200818.dat
    ```

