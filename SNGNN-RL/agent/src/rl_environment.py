import sys
import os
import time
import math

from threading import Thread, Lock
import numpy as np
import _pickle as pickle
import Ice
import torch
import torch.nn.functional as F

from collections import deque 

sys.path.append(os.path.join(os.path.dirname(__file__), '../socnav/'))
from socnav import *

from shapely.geometry import Point, Polygon

MAX_EPISODE_LENGTH = 500

# match it with rl_constant files
TIME_STEP = -1


class RLEnvironment(object):
    def __init__(self, simulator_proxy, omnirobot_proxy, ui):
        super(RLEnvironment, self).__init__()
        if torch.cuda.is_available() is True:
            self.device_str = 'cuda'
        else:
            self.device_str = 'cpu'

        # Get reference to UI
        self.ui = ui

        # Make local copies of the proxies
        self.simulator_proxy = simulator_proxy
        self.omnirobot_proxy = omnirobot_proxy

        # Initialise people's data and mutex
        self.people_data = []
        # Initialise objec's data and mutex
        self.object_data = []
        # Initialise interactions' data and mutex
        self.interactions_data = []
        # Initialise walls' data and mutex
        self.walls_data = []
        # Initialise goal's data and mutex
        self.goal_data = {}
        self.ALLDATA_mutex = Lock()
        self.ALLDATA_mutex.acquire()

        # Most recent observations
        self.observations = []
        # The most recent observations are those received for the last self.MAX_OBSERVATION_TIME seconds
        self.MAX_OBSERVATION_TIME = 5

        # Proximity to goal
        self.threshold_distance = 0.5

        # STEPS
        self.steps = 0
        self.max_episode_length = MAX_EPISODE_LENGTH
        if self.check_play():
            self.max_episode_length *= 2

        self.robot_inc_steps = deque(maxlen=20)
        self.latest_action = None
        self.robot_dist_threshold = 3.5*self.robot_inc_steps.maxlen

    def set_info(self, info):
        self.info = info

    def step(self, action):
        # Perform action
        try:
            self.omnirobot_proxy.setSpeedBase(float(action[2]), float(action[1]), float(action[0]))
        except Exception as e:
            print('error sending command', e)
        # Step in the simulator
        try:
            self.simulator_proxy.step(TIME_STEP)
        except Ice.ConnectionRefusedException:
            print('The simulator seems to be off! We will:')
            print(' - wait a few seconds')
            print(' - return the last observation-reward, stating that the episode ended')
            print(' - hope that the simulator will be back after the wait')
            time.sleep(15)
            return self.just_in_case__observation, True, {'info': 'none yet'}

        # Compute the observation
        observation = self.generate_observation(action)
        self.latest_action = observation[0]['command']

        # print(self.latest_action)
        inc_pose = abs(self.latest_action[0])
        self.robot_inc_steps.append(inc_pose)


        observation_graph = SocNavDataset(observation, '8', 'test', prev_graph=None, verbose=False, debug=True)

        
        self.steps += 1

        is_done = self.check_state(use_l2=False)

        # info must have the min, max values of the spawning entities
        self.just_in_case__observation = observation
        return observation_graph, is_done, {'info': 'none yet'}


    def check_state(self, use_l2=True):
        # check for collision
        collision = self.collision_check()

        # Check if we're done
        distance_to_goal = math.sqrt((self.goal_coordinates[0])**2 + (self.goal_coordinates[1])**2)
        if distance_to_goal > 10:
            distance_to_goal = 10
            collision = True

        is_done = False

        if distance_to_goal <= self.threshold_distance:
            print("Reached the goal position. Simulating new episode.")
            is_done = True
        elif collision:
            print("Collision! Simulating New Episode")
            is_done = True
        elif self.steps > self.max_episode_length:
            print("Max steps! Simulating New Episode")
            is_done = True
        else:
            is_done = False

        return is_done

    def speeds_to_wheels(self, forward_speed, rotation_speed, r, l):
        MAX_SPEED = 6.28
        left_wheel = (forward_speed - (-rotation_speed*l)/2.) / r
        right_wheel = (forward_speed + (-rotation_speed*l)/2.) / r

        # we want the network to predict permissible speed values for each wheel
        return max(min(left_wheel,MAX_SPEED),-MAX_SPEED), max(min(right_wheel,MAX_SPEED),-MAX_SPEED)


    def set_people_data(self, people):
        self.people_data = people
        # if self.people_mutex.locked():
        #     self.people_mutex.release()

    def set_object_data(self, objects):
        self.object_data = objects

    def set_interactions_data(self, interactions):
        self.interactions_data = interactions

    def set_walls_data(self, walls):
        self.walls_data = walls

    def set_goal_data(self, goal):
        self.goal_data = goal
        self.goal_coordinates = [goal.x, goal.y]
        if self.ALLDATA_mutex.locked():
            self.ALLDATA_mutex.release()

    def generate_observation(self, command):
        ret = self.ALLDATA_mutex.acquire(timeout=2)

        current_observation = self.get_current_observation(command)

        self.observations.insert(0, current_observation)
        current_time = self.observations[0]["timestamp"]
        for o in self.observations[::-1]:
            if current_time - o["timestamp"] > self.MAX_OBSERVATION_TIME:
                self.observations.remove(o)
            else:
                break

        return self.observations


    def reset(self):
        done_reset = False

        while not done_reset:
            try:
                self.steps = 0
                self.initial_positions = dict()
                self.initial_positions_objects = dict()
                self.observations = []

                nhumans_max = 10  # self.ui.nhumans_max.value()
                nhumans_lambda = 5.0  # self.ui.nhumans_lambda.value()
                nwandhumans_max = 10  # self.ui.nwandhumans_max.value()
                nwandhumans_lambda = 5.0  # self.ui.nwandhumans_lambda.value()
                nplants_max = 5  # self.ui.nplants_max.value()
                nplants_lambda = 5.0  # self.ui.nplants_lambda.value()
                ntables_max = 5  # self.ui.ntables_max.value()
                ntables_lambda = 5.0  # self.ui.ntables_lambda.value()
                nrelations_max = 5  # self.ui.nrelations_max.value()
                nrelations_lambda = 5.0  # self.ui.nrelations_lambda.value()

                include_walls = 0
                # if self.ui.include_walls.isChecked():
                    # include_walls = 1

                dynamic_humans = int(np.random.exponential(scale=1./nwandhumans_lambda))
                dynamic_humans = min(dynamic_humans, nwandhumans_max)

                static_humans = int(np.random.exponential(scale=1./nhumans_lambda))
                static_humans = min(static_humans, nhumans_max)

                plants = int(np.random.exponential(scale=1./nplants_lambda))
                plants = min(plants, nplants_max)  # random.randint(nplants_lambda, nplants_max)

                tables = int(np.random.exponential(scale=1./ntables_lambda))
                tables = min(tables, ntables_max)  # random.randint(ntables_lambda, ntables_max)

                nrelations = int(np.random.exponential(scale=1./nrelations_lambda))
                nrelations = min(nrelations, nrelations_max)  # random.randint(nrelations_lambda, nrelations_max)

                string_to_send = str(static_humans)+" "+str(dynamic_humans)+" "+str(plants)+" "+str(tables)+" "+str(nrelations)+" "+str(include_walls)
                self.last_scene_string = string_to_send
                self.simulator_proxy.regenerate(string_to_send)

                done_reset = True        
            except Exception as e:
                print('something went bad regenerating', e)
                time.sleep(30)
        return self.observations

    def collision_check(self):
        # collision penalty is -1
        collision = False
        # collision_pen = 0

        for object_ in self.object_data:
            if object_.id == -2:
                collision = object_.collision

        return collision 

    def get_current_observation(self, command):
        temp_obj_list = []
        temp_inter_list = []
        temp_people_list = []
        temp_goal_list = []
        temp_wall_list = []


        for inter in self.interactions_data:
            temp_inter_dic = {}
            temp_inter_dic['src'] = inter.idSrc
            temp_inter_dic['relation'] = inter.type
            temp_inter_dic['dst'] = inter.idDst
            temp_inter_list.append(temp_inter_dic)


        for pos, obj in enumerate(self.object_data):
            if obj.id != -2:
                temp_obj_dic = {}
                temp_obj_dic["id"] = obj.id
                temp_obj_dic["x"] = obj.x
                temp_obj_dic["y"] = obj.y
                temp_obj_dic["a"] = obj.angle
                temp_obj_dic["size_x"] = abs(obj.bbx2 - obj.bbx1)
                temp_obj_dic["size_y"] = abs(obj.bby2 - obj.bby1)
                try:
                    temp_obj_dic["vx"] = self.initial_positions_objects[pos][0] - obj.x
                    temp_obj_dic["vy"] = self.initial_positions_objects[pos][1] - obj.y
                    temp_obj_dic["va"] = self.initial_positions_objects[pos][2] - obj.angle
                except:
                    temp_obj_dic["vx"] = temp_obj_dic["vy"] = temp_obj_dic["va"] = 0.
                self.initial_positions_objects[pos] = [ obj.x, obj.y, obj.angle ]
                temp_obj_list.append(temp_obj_dic)


        for pos,person in enumerate(self.people_data):
            temp_person_dic = {}
            temp_person_dic["id"] = person.id
            temp_person_dic["x"] = person.x
            temp_person_dic["y"] = person.y
            temp_person_dic["a"] = person.angle
            try:
                temp_person_dic["vx"] = self.initial_positions[pos][0] - person.x
                temp_person_dic["vy"] = self.initial_positions[pos][1] - person.y
                temp_person_dic["va"] = self.initial_positions[pos][2] - person.angle
            except:
                temp_person_dic["vx"] = temp_person_dic["vy"] = temp_person_dic["va"] = 0
            self.initial_positions[pos] = [ person.x, person.y, person.angle ]
            temp_people_list.append(temp_person_dic)

        for wall in self.walls_data:
            temp_wall_dic = {}
            temp_wall_dic["x1"] = wall.x1
            temp_wall_dic["y1"] = wall.y1
            temp_wall_dic["x2"] = wall.x2
            temp_wall_dic["y2"] = wall.y2
            temp_wall_list.append(temp_wall_dic)

        try:
            temp_goal_dic = {"x": self.goal_data.x, "y": self.goal_data.y}
        except:
            print(self.goal_data)

        temp_goal_list.append(temp_goal_dic)

        command[0] = command[0]*3.5/0.45
        command[2] = command[2]*4./0.45

        data = {"ID": 0,
                "timestamp":self.goal_data.timestamp,
                "step_fraction": self.steps/MAX_EPISODE_LENGTH,
                "objects":temp_obj_list,
                "people":temp_people_list,
                "walls":temp_wall_list,
                "goal":temp_goal_list,
                "command": command,
                "interaction":temp_inter_list}
        return data

    def check_play(self):
        for x in sys.argv:
            if x.startswith('--play='):
                return x.split('=')[1]
        return None
