#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2020 by YOUR NAME HERE
#
#    This file is part of RoboComp
#
#    RoboComp is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    RoboComp is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with RoboComp.  If not, see <http://www.gnu.org/licenses/>.
#

import sys
import threading
import time
import ast
import random
import numpy as np
import json
import cv2
import signal
import numpy
from math import pi, sin, cos
from pyrep.objects.object import Object

import _pickle as pickle

sys.path.append('../python')

from sonata import SODA

from genericworker import *

import signal

class SpecificWorker(GenericWorker):
    def __init__(self, proxy_map):
        super(SpecificWorker, self).__init__(proxy_map)

        self.data = {
        'walls': [],
        'goal': [],
        'humans': None,
        'robot_position': None,
        'robot_orientation': None,
        'simulator_mutex': threading.RLock()
        }


        self.soda = SODA(proxy_map, self.data)
        self.min_max_data = {"min_humans":0, "max_humans":4,
                             "min_wandering_humans":0, "max_wandering_humans":4,
                             "min_tables":0, "max_tables":4,
                             "min_plants":0, "max_plants":4,
                             "min_relations":0, "max_relations":4}

        self.data, self.wandering_humans = self.soda.room_setup(self.min_max_data['min_humans'],
                                                                self.min_max_data['min_wandering_humans'],
                                                                self.min_max_data['min_plants'],
                                                                self.min_max_data['min_tables'],
                                                                self.min_max_data['min_relations'],
                                                                self.min_max_data['max_humans'],
                                                                self.min_max_data['max_wandering_humans'],
                                                                self.min_max_data['max_plants'],
                                                                self.min_max_data['max_tables'],
                                                                self.min_max_data['max_relations'] )

        #Loop
        self.adv = self.rot = 0.
        self.last_ten = time.time()-10
        self.last_point_one = time.time()-10
        self.end_simulation = False
        self.vision_sensor = Object.get_object('Vision_sensor')

    def __del__(self):
        print('SpecificWorker destructor')
        del self.soda

    def stop_simulation(self):
        self.end_simulation = True

    def setParams(self, params):
        return True

    def compute(self):
        time_to_sleep = 0.
        while time.sleep(time_to_sleep) is None and not self.end_simulation:
            time_zero = time.time()
            # print('SpecificWorker.compute...', time_to_sleep)

            # Initialise people/object/wall's structure
            people = []
            objects = []
            interactions = []
            walls = []
            goal = []

            #
            # MUTEX ACQUIRE
            #
            self.data['simulator_mutex'].acquire()

            # Simulation step
            self.data, people, objects, interactions, walls, goal = self.soda.soda_compute(people, objects, walls, goal)
            self.data['simulator_mutex'].release()
            #
            # MUTEX RELEASE
            #


            # THIS IS JUST TO LET YOU KNOW HOW TO PUBLISH THE STRUCTURES!!
            # This is for the goal
            # This is for the images
            image = self.vision_sensor.capture_rgb()
            image = (image * 255.).round().astype(np.uint8)
            byte_stream = pickle.dumps(image)
            self.bytesequencepublisher_proxy.newsequence(byte_stream)

            self.peopledetector_proxy.gotpeople(people)
            self.objectdetector_proxy.gotobjects(objects)
            self.interactiondetector_proxy.gotinteractions(interactions)
            self.walldetector_proxy.gotwalls(walls)
            self.goalpublisher_proxy.goalupdated(goal)


            # ta = time.time()
            # vision.capture_rgb()
            # print (time.time()-ta)
            time_spent = time.time() - time_zero
            time_to_sleep = self.soda.get_simulation_timestep() - time_spent
            if time_to_sleep < 0:
                time_to_sleep = 0
        print("main simulator loop ended")
        return True

    # =============== Methods for Component Implements ==================
    # ===================================================================

    #
    # correctOdometer
    #
    def OmniRobot_correctOdometer(self, x, z, alpha):
        #
        # implementCODE
        #
        pass


    #
    # getBasePose
    #
    def OmniRobot_getBasePose(self):
        #
        # implementCODE
        #
        self.soda.data['simulator_mutex'].acquire()
        x = int(0)
        z = int(0)
        alpha = float(0)
        self.soda.data['simulator_mutex'].release()
        return [x, z, alpha]


    #
    # getBaseState
    #
    def OmniRobot_getBaseState(self):
        #
        # implementCODE
        #
        state = RoboCompGenericBase.TBaseState()
        self.soda.data['simulator_mutex'].acquire()
        state.x = int(0)
        state.z = int(0)
        state.alpha = float(0)
        self.soda.data['simulator_mutex'].release()
        return state


    #
    # resetOdometer
    #
    def OmniRobot_resetOdometer(self):
        #
        # implementCODE
        #
        pass


    #
    # setOdometer
    #
    def OmniRobot_setOdometer(self, state):
        #
        # implementCODE
        #
        pass


    #
    # setOdometerPose
    #
    def OmniRobot_setOdometerPose(self, x, z, alpha):
        #
        # implementCODE
        #
        pass


    #
    # setSpeedBase
    #
    def OmniRobot_setSpeedBase(self, advx, advz, rot):
        self.soda.data['simulator_mutex'].acquire()
        radius = 0.0475 # youbot's weel radius
        rotation_to_linear_ratio = 2. * pi * radius
        #print(rotation_to_linear_ratio)
        self.soda.robot.set_base_angular_velocites([advx/rotation_to_linear_ratio, advz/rotation_to_linear_ratio, rot])
        self.soda.data['simulator_mutex'].release()


    #
    # stopBase
    #
    def OmniRobot_stopBase(self):
        self.OmniRobot_setSpeedBase(0., 0., 0.)


    #
    # Reset simulation
    #
    def Simulator_regenerate(self, scene):
        print('We should reinitialise the simulation')
        print("scene", scene)
        self.min_max_data = ast.literal_eval(scene)
        print(self.min_max_data)
        self.soda.data['simulator_mutex'].acquire()
        self.data, self.wandering_humans = self.soda.room_setup(self.min_max_data['min_humans'],
                                                                self.min_max_data['min_wandering_humans'],
                                                                self.min_max_data['min_plants'],
                                                                self.min_max_data['min_tables'],
                                                                self.min_max_data['min_relations'],
                                                                self.min_max_data['max_humans'],
                                                                self.min_max_data['max_wandering_humans'],
                                                                self.min_max_data['max_plants'],
                                                                self.min_max_data['max_tables'],
                                                                self.min_max_data['max_relations'])
        self.soda.data['simulator_mutex'].release()



    # ===================================================================
    # ===================================================================
