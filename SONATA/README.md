# SONATA

SONATA: A Toolkit to Generate Social Navigation Datasets and HRI

## Introduction

The toolkit is used to generate the dataset by simulating the scenarios for robot's navigation in a social setting. We show an usecase of this data collected from the toolkit by converting into graphs and feeding it into the GNNs to predicts the robot's path in a given scene.

## Software requirements
1. PyTorch [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
2. DGL [https://www.dgl.ai/pages/start.html](https://www.dgl.ai/pages/start.html)
3. PyTorch Geometric [https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

# Running the repo

After cloning the directory, execute the following commands:
1. Shift the interfaces to proper location.
    ```
    cp interfaces/* /opt/robocomp/interfaces
    cp interfaces/* /opt/robocomp/interfaces/IDSLs
    ```
2. Run the tool.
    ```
    bash run.sh
    ```

# Usage

After you run the above commands the SONATA GUI opens up.

1. Write the contributor's(user's) name so that the data saved can be marked by the users name.
![contributor](./images/get_contributor.png)

2. After this the simulation will start with a green tint, and until the mouse controller is clicked robot will not move. 
![simulator](./images/simulation_green_start.png)

3. Select the configuration from the top bar of the SONATA GUI to select the range of different entities you want to add in the scene.
![configuration](./images/select_range.png)

4. click on the mouse controller and hold the left key and drag the mouse to move the robot in that direction.
![mouse](./images/click_joystick.png)

5. Move the robot to the goal marked by the cone.
![goal](./images/reach_goal.png)

6. Save your data.
![save](./images/save.png)

