# Introduction

This repository contains the work done by Robocomp on social navigation and particularly using Graph neural networks.
It contains the following repository.
1. SNGNN1D
2. SONATA
3. SNGNN-RL

SNGNN1D (Social Navigation Graph Neural Network 1D) is used to evaluate the robot based on the static social scene 
generated randomly. The work done was a part of GSoC '19 program.

SONATA is a toolkit used to collect data for dynamic social scenario. This is currently under work and part of the GSoC 
'20 program.

SNGNN-RL makes use of Dueling DQN, to learn a policy (path planner) that would not require to make use of heavily engineered heuristics pertaining to the controller of the robot and that is invariant to the robot type and the control policy that it uses to achieve the task of path planning in social settings.

For more details check the corresponding directories.