import os
import sys
import math
from controller import Robot

os.environ['WEBOTS_ROBOT_NAME'] = 'TIAGo Iron'

MAX_SPEED = 6.28

def speeds_to_wheels(forward_speed, rotation_speed, r, l):
    left_wheel = (forward_speed - (-rotation_speed*l)/2.) / r
    right_wheel = (forward_speed + (-rotation_speed*l)/2.) / r

    # we want the network to predict permissible speed values for each wheel
    return max(min(left_wheel,MAX_SPEED),-MAX_SPEED), max(min(right_wheel,MAX_SPEED),-MAX_SPEED)



# create the Robot instance.
robot = Robot()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# Enable touch sensor to detect collisions:
Tsensor = robot.getDevice("main_sensor")
Tsensor.enable(200)  # Sampling period in milliseconds as parameter

left_wheel = robot.getDevice("wheel_left_joint")
right_wheel = robot.getDevice("wheel_right_joint")
left_wheel.setPosition(float('inf'))
right_wheel.setPosition(float('inf'))

radius = 0.0985
distance_between_wheels = 0.404

collided = False

while robot.step(timestep) != -1:
    data = robot.getCustomData()
    if len(data) == 0:
        continue
    customData = [float(x) for x in data.split(',')]
    left_wheel_speed, right_wheel_speed = speeds_to_wheels(customData[0], customData[2], radius, distance_between_wheels)
    left_wheel.setVelocity(left_wheel_speed)
    right_wheel.setVelocity(right_wheel_speed)
    # Readings from the touch sensor:
    collision = Tsensor.getValue()  # Value is 1 if there is a collision and 0 otherwise
    if math.isnan(collision) and not collided:
        collision = '0'
    elif collision > 0 or collided:
        collision = '1'
        collided = True
    else:
        collision = '0'
    data = ','.join(map(str, customData[0:3]))+','+collision
    robot.setCustomData(data)




