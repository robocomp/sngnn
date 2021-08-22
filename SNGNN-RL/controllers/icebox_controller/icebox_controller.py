import os
import time

import psutil

os.environ['WEBOTS_ROBOT_NAME'] = 'icebox'

found_icebox = False
for proc in psutil.process_iter():
    try:
        if 'icebox' in proc.name().lower():
            found_icebox = True
            break
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        pass

if found_icebox is False:
    # print('icebox controller: killing previous instances')
    # os.system("killall -9 icebox")
    print('icebox controller: running new')
    os.system("icebox --Ice.Config=icebox.conf &")


from controller import Robot
robot = Robot()

while robot.step(int(robot.getBasicTimeStep())) != -1:
    pass
