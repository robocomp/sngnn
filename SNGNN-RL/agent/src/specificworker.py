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

from genericworker import *

import os
import json
import subprocess
import time
import signal
import inspect
import numpy as np
import threading
import copy
import sys

import datetime
import math
from PySide2 import QtGui, QtCore

import PySide2.QtGui as QtGui
from PySide2.QtCore import QSettings

import _pickle as pickle

from rl_environment import RLEnvironment

import importlib

import time

from rl_agent_dueling_dqn import RLAgent

import json


from shapely.geometry import Point, Polygon


class SpecificWorker(GenericWorker):
    def __init__(self, proxy_map, fullscreen, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map)
        self.timer.timeout.connect(self.compute)

        self.relations_dic = {}

        # self.ui.quitButton.clicked.connect(self.quit)

        self.environment = RLEnvironment(self.simulator_proxy, self.omnirobot_proxy, None)  #self.ui)
        self.agent = RLAgent(self.environment)

        # from hanging_threads import start_monitoring
        # monitoring_thread = start_monitoring(seconds_frozen=10, test_interval=100)

        self.app = None

        self.environment.reset()
        self.timer.start(0)

        self.lastTimeClose = time.time()
        self.closeRequests = 0


    @QtCore.Slot()
    def quit(self):
        self.close()

    @QtCore.Slot()
    def contributor_changed(self):
        self.settings.setValue("contributor", self.form.val)

    @QtCore.Slot()
    def keyPressEvent(self, event):
        if event.type() == QtCore.QEvent.KeyPress:
            if event.key() == QtCore.Qt.Key_R:
                pass  # Not used here

    def closeEvent(self, event):
        if time.time() - self.lastTimeClose < 1:
            self.closeRequests += 1
        else:
            self.closeRequests = 1

        self.lastTimeClose = time.time()

        if self.closeRequests > 4:
            print('Exiting')
            sys.exit(1)
        else:
            event.ignore()

    def __del__(self):
        self.quit_slot()

    def setParams(self, params):
        return True

    @QtCore.Slot()
    def quit_slot(self):
        pass

    @QtCore.Slot()
    def compute(self):
        self.agent.compute()
        # self.close()

    def set_app(self, app):
        self.app = app
        self.agent.set_app(app)

    # SUBSCRIPTION to gotobjects method from ObjectDetector interface
    #
    def ObjectDetector_gotobjects(self, lst):
        self.environment.set_object_data(lst)

    #
    # SUBSCRIPTION to gotinteractions method from InteractionDetector interface
    #
    def InteractionDetector_gotinteractions(self, lst):
        self.environment.set_interactions_data(lst)

    #
    # SUBSCRIPTION to gotpeople method from PeopleDetector interface
    #
    def PeopleDetector_gotpeople(self, lst):
        self.environment.set_people_data(lst)

    #
    # SUBSCRIPTION to gotwalls method from WallDetector interface
    #
    def WallDetector_gotwalls(self, lst):
        self.environment.set_walls_data(lst)

    #
    # SUBSCRIPTION to newsequence method from ByteSequencePublisher interface
    #
    def ByteSequencePublisher_newsequence(self, bs):
        # self.img_mutex.acquire()
        self.img = pickle.loads(bs)
        # self.img_mutex.release()

    #
    # SUBSCRIPTION to goalupdated method from GoalPublisher interface
    #
    def GoalPublisher_goalupdated(self, goal):
        self.environment.set_goal_data(goal)

    def JoystickAdapter_sendData(self, data):
        pass

    def info(self):
        dic = {"lambda_humans":0, "max_humans":0,
                   "lambda_wandering_humans":0, "max_wandering_humans":0,
                   "plants_lambda":0, "max_plants":0,
                   "lambda_tables":0, "max_tables":0, 
                   "lambda_relations":0, "max_relations":0}
        # dic = {"lambda_humans":self.ui.nhumans_lambda.value(), "max_humans":self.ui.nhumans_max.value(),
        #            "lambda_wandering_humans":self.ui.nwandhumans_lambda.value(), "max_wandering_humans":self.ui.nwandhumans_max.value(),
        #            "plants_lambda":self.ui.nplants_lambda.value(), "max_plants":self.ui.nplants_max.value(),
        #            "lambda_tables":self.ui.ntables_lambda.value(), "max_tables":self.ui.ntables_max.value(), 
        #            "lambda_relations":self.ui.nrelations_lambda.value(), "max_relations":self.ui.nrelations_max.value()}
        return dic
