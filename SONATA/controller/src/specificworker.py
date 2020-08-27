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
from PySide2 import QtGui, QtCore, QtWidgets

import PySide2.QtGui as QtGui
from PySide2.QtGui import QPixmap, QImage
from PySide2.QtCore import QSettings

import _pickle as pickle
# If RoboComp was compiled with Python bindings you can use InnerModel in Python
# sys.path.append('/opt/robocomp/lib')
# import librobocomp_qmat
# import librobocomp_osgviewer
# import librobocomp_innermodel
import json
from PySide2.QtWidgets import (QLabel, QLineEdit, QPushButton, QApplication,
    QVBoxLayout, QDialog, QWidget)
from ui_configuration import *
from contributor_GUI import *

sys.path.append(os.path.join(os.path.dirname(__file__),'../../usecase'))
from socnavAPI import *
from socnavData import *

class SpecificWorker(GenericWorker):
    def __init__(self, proxy_map, fullscreen, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map)
        os.system('bash ../joystick.sh &')
        os.system('bash ../simulator.sh &')
        self.timer.timeout.connect(self.compute)
        self.TEST_MODE = False
        self.i_frame = 0
        if self.TEST_MODE:
            self.model = SocNavAPI('.', None)

        self.img_mutex = threading.RLock()
        self.img = None
        self.threshold = 0.4
        self.robot_coordinates = [0,0]
        ###
        self.window = QWidget()
        self.button = QPushButton("enter")
        self.layout = QVBoxLayout()
        ###
        self.goal_coordinates = [100,100]
        self.Period = 100
        self.timer.start(self.Period)
        self.data = []
        self.goal_data = []
        self.wall_data = []
        self.interaction_data = []
        self.people_data = []
        self.object_data = []
        self.updates_list = []
        self.initial_positions = []
        self.initial_positions_objects = []
        self.initial_flag = 0
        self.data_json = []
        self.speed_command = [0., 0., 0.]
        self.relations_dic = {}

        self.configGui = Ui_confDialog()
        self.configWindow = QDialog()
        self.configGui.setupUi(self.configWindow)
        if fullscreen != "True":
            self.WINDOW_WIDTH = 1042
            self.WINDOW_HEIGHT = 819
            self.setFixedSize(self.WINDOW_WIDTH,self.WINDOW_HEIGHT)
        self.save_interval = 0.1 #
        self.last_saved_time = time.time()

        self.new_goal = False
        self.on_a_mission = True
        self.started_mission = False
        self.ui.configuration.clicked.connect(self.configuration_slot)
        self.ui.regenerate.clicked.connect(self.regenerate_slot)
        self.ui.quit.clicked.connect(self.quit_slot)

        self.configGui.nhumans_min.valueChanged.connect(self.min_nhumans_changed)
        self.configGui.nhumans_max.valueChanged.connect(self.max_nhumans_changed)
        self.configGui.nwandhumans_min.valueChanged.connect(self.min_nwandhumans_changed)
        self.configGui.nwandhumans_max.valueChanged.connect(self.max_nwandhumans_changed)
        self.configGui.ntables_min.valueChanged.connect(self.min_ntables_changed)
        self.configGui.ntables_max.valueChanged.connect(self.max_ntables_changed)
        self.configGui.nplants_min.valueChanged.connect(self.min_nplants_changed)
        self.configGui.nplants_max.valueChanged.connect(self.max_nplants_changed)
        self.configGui.nrelations_min.valueChanged.connect(self.min_nrelations_changed)
        self.configGui.nrelations_max.valueChanged.connect(self.max_nrelations_changed)
        
        self.form = CForm()
        self.form.show()
        self.form.exec_()
        self.cont_name = self.form.val

        self.settings = QSettings("dataset", "xxx")
        contributor = self.settings.value("contributor", self.cont_name)
        self.ui.contributor.setText(contributor)
        self.ui.contributor.textChanged.connect(self.contributor_changed)

        if fullscreen == 'True':
            self.showFullScreen()

    @QtCore.Slot()
    def contributor_changed(self):
        self.settings.setValue("contributor", self.form.val)

    @QtCore.Slot()
    def configuration_slot(self, active):
        if active:
            self.configWindow.show()
        else:
            self.configWindow.hide()

    @QtCore.Slot()
    def regenerate_slot(self):

        print('regenerate')
        self.omnirobot_proxy.setSpeedBase(0, 0, 0)
        self.i_frame = 0
        self.updates_list = []
        self.initial_positions = []
        self.initial_positions_objects = []
        self.initial_flag=0
        self.info()
        
        self.simulator_proxy.regenerate(scene=self.relations_dic)
        self.new_goal = False
        self.on_a_mission = True
        self.started_mission = False


    @QtCore.Slot()
    def quit_slot(self):
        os.system('pwd')
        os.system('bash ../kill.sh &')


    @QtCore.Slot()
    def keyPressEvent(self, event):
        if event.type() == QtCore.QEvent.KeyPress:
            if event.key() == QtCore.Qt.Key_Q:
                self.quit_slot()
            elif event.key() == QtCore.Qt.Key_R:
                self.regenerate_slot()


    @QtCore.Slot()
    def min_nhumans_changed(self, val):
        if val>self.configGui.nhumans_max.value():
            self.configGui.nhumans_max.setValue(val)

    @QtCore.Slot()
    def max_nhumans_changed(self, val):
        if val<self.configGui.nhumans_min.value():
            self.configGui.nhumans_min.setValue(val)

    @QtCore.Slot()
    def min_nwandhumans_changed(self, val):
        if val>self.configGui.nwandhumans_max.value():
            self.configGui.nwandhumans_max.setValue(val)

    @QtCore.Slot()
    def max_nwandhumans_changed(self, val):
        if val<self.configGui.nwandhumans_min.value():
            self.configGui.nwandhumans_min.setValue(val)

    @QtCore.Slot()
    def min_ntables_changed(self, val):
        if val>self.configGui.ntables_max.value():
            self.configGui.ntables_max.setValue(val)

    @QtCore.Slot()
    def max_ntables_changed(self, val):
        if val<self.configGui.ntables_min.value():
            self.configGui.ntables_min.setValue(val)

    @QtCore.Slot()
    def min_nplants_changed(self, val):
        if val>self.configGui.nplants_max.value():
            self.configGui.nplants_max.setValue(val)

    @QtCore.Slot()
    def max_nplants_changed(self, val):
        if val<self.configGui.nplants_min.value():
            self.configGui.nplants_min.setValue(val)

    @QtCore.Slot()
    def min_nrelations_changed(self, val):
        if val>self.configGui.nrelations_max.value():
            self.configGui.nrelations_max.setValue(val)

    @QtCore.Slot()
    def max_nrelations_changed(self, val):
        if val<self.configGui.nrelations_min.value():
            self.configGui.nrelations_min.setValue(val)


    def __del__(self):
        os.system('kill -15 `ps ax |grep simulator | grep config | awk \'{print $1}\'`')
        os.system('kill -15 `ps ax |grep joystick | grep config | awk \'{print $1}\'`')
        print('SpecificWorker destructor')
        self.quit_slot()

    def setParams(self, params):
        return True

    @QtCore.Slot()
    def compute(self):
        self.show_image()

        if self.TEST_MODE:
            self.predict_from_network()
        else:
            self.collect_data()


    def predict_from_network(self):
        if not self.on_a_mission or not self.new_goal:
            return

        proximity = math.sqrt(sum([(a - b) ** 2 for a, b in zip(self.robot_coordinates, self.goal_coordinates)]))
        if proximity <= self.threshold:
            self.regenerate_slot()
            return


        if time.time()-self.last_saved_time > self.save_interval:
            self.last_saved_time = time.time()
            
            temp_obj_list, temp_inter_list, temp_people_list, temp_goal_list, temp_wall_list = self.get_scene()
            data = {"timestamp":time.time(),
                    "objects":temp_obj_list,
                    "people":temp_people_list,
                    "walls":temp_wall_list,
                    "goal":temp_goal_list,
                    "command": self.speed_command,
                    "interaction":temp_inter_list}

            self.updates_list.insert(0, data)

            i = -1
            new_data = [data]
            for i in range(1, len(self.updates_list)):
                if new_data[-1]['timestamp']-self.updates_list[i]['timestamp'] >= FRAMES_INTERVAL:
                    new_data.append(self.updates_list[i])
                    if len(new_data) == N_INTERVALS:
                        break

            if len(new_data)>=1:
                graph = GenerateDataset(new_data, 'run', '1', i_frame = self.i_frame)
                results = self.model.predictOneGraph(graph)[0]
                adv = results[0].item()*3.5
                rot = results[2].item()*4.
                self.omnirobot_proxy.setSpeedBase(adv, 0, rot)
                if len(new_data)==N_INTERVALS and i+1 < len(self.updates_list):
                    for r in range(i+1, len(self.updates_list)):
                        self.updates_list.pop(i+1)
                print(results)

            self.i_frame += 1

    def collect_data(self):

        if not self.on_a_mission or not self.new_goal or not self.started_mission:
            return

        self.omnirobot_proxy.setSpeedBase(*self.speed_command)

        
        control_flag = 0

        proximity = math.sqrt(sum([(a - b) ** 2 for a, b in zip(self.robot_coordinates, self.goal_coordinates)]))


        if time.time()-self.last_saved_time > self.save_interval:
            self.last_saved_time = time.time()
            
            temp_obj_list, temp_inter_list, temp_people_list, temp_goal_list, temp_wall_list = self.get_scene()
            if  len(temp_obj_list)!=0 or len(temp_people_list)!=0 or len(temp_wall_list)!=0:
                self.updates_list.append({"timestamp":time.time(),
                                          "objects":temp_obj_list,
                                          "people":temp_people_list,
                                          "walls":temp_wall_list,
                                          "goal":temp_goal_list,
                                          "command": self.speed_command,
                                          "interaction":temp_inter_list})
                self.data_json = self.updates_list


        if proximity <= self.threshold:
            msgBox = QtWidgets.QMessageBox()
            msgBox.setText("Robot found the Goal")
            msgBox.setInformativeText("Do you want to save the data?")
            msgBox.setStandardButtons(QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Discard)
            msgBox.setDefaultButton(QtWidgets.QMessageBox.Save)
            self.new_goal = False
            self.on_a_mission = False
            self.show_image()
            ret = msgBox.exec_()
            if ret == QMessageBox.Save:
                time_str = '{0:%Y-%m-%d__%H:%M:%S}'.format(datetime.datetime.now())
                with open(f'../{self.ui.contributor.text()}__{time_str}.json', 'w') as fp:
                    json.dump(self.data_json, fp, indent=4, sort_keys=True)
                    self.updates_list = []
                    self.initial_positions = []
                    self.initial_positions_objects = []
                    self.initial_flag=0

            self.regenerate_slot()

    def get_scene(self):
        temp_obj_list = []
        temp_inter_list = []
        temp_people_list = []
        temp_goal_list = []
        temp_wall_list = []

        for inter in self.interaction_data:
            temp_inter_dic = {}
            temp_inter_dic['src'] = inter.idSrc
            temp_inter_dic['relation'] = inter.type
            temp_inter_dic['dst'] = inter.idDst
            temp_inter_list.append(temp_inter_dic)
                
        if self.initial_flag==0:
            for _, obj in enumerate(self.object_data):
                self.initial_positions_objects.append([obj.x, obj.y, obj.angle])
            for _,person in enumerate(self.people_data):
                self.initial_positions.append([person.x, person.y, person.angle])

        #print("INITIAL POSITIONS >>>>",self.initial_positions)
        self.initial_flag=1

            
        for pos, obj in enumerate(self.object_data):
            if obj.id != -2:
                temp_obj_dic = {}
                temp_obj_dic["id"] = obj.id
                temp_obj_dic["x"] = obj.x
                temp_obj_dic["y"] = obj.y
                temp_obj_dic["a"] = obj.angle
                temp_obj_dic["vx"] = self.initial_positions_objects[pos][0] - obj.x
                temp_obj_dic["vy"] = self.initial_positions_objects[pos][1] - obj.y
                temp_obj_dic["va"] = self.initial_positions_objects[pos][2] - obj.angle
                temp_obj_dic["size_x"] = abs(obj.bbx2 - obj.bbx1)
                temp_obj_dic["size_y"] = abs(obj.bby2 - obj.bby1)
                self.initial_positions_objects[pos][0] = obj.x
                self.initial_positions_objects[pos][1] = obj.y
                self.initial_positions_objects[pos][2] = obj.angle
                temp_obj_list.append(temp_obj_dic)


        for pos,person in enumerate(self.people_data):
            temp_person_dic = {}
            temp_person_dic["id"] = person.id
            temp_person_dic["x"] = person.x
            temp_person_dic["y"] = person.y
            temp_person_dic["a"] = person.angle
            temp_person_dic["vx"] = self.initial_positions[pos][0] - person.x
            temp_person_dic["vy"] = self.initial_positions[pos][1] - person.y
            temp_person_dic["va"] = self.initial_positions[pos][2] - person.angle
            self.initial_positions[pos][0] = person.x
            self.initial_positions[pos][1] = person.y
            self.initial_positions[pos][2] = person.angle
            temp_people_list.append(temp_person_dic)
          
        for wall in self.wall_data:
            temp_wall_dic = {}
            temp_wall_dic["x1"] = wall.x1
            temp_wall_dic["y1"] = wall.y1
            temp_wall_dic["x2"] = wall.x2
            temp_wall_dic["y2"] = wall.y2
            temp_wall_list.append(temp_wall_dic)

            
        temp_goal_dic = {"x":self.goal_coordinates[0],  "y":self.goal_coordinates[1]}
        temp_goal_list.append(temp_goal_dic)

        return temp_obj_list, temp_inter_list, temp_people_list, temp_goal_list, temp_wall_list



    def show_image(self):
        self.img_mutex.acquire()
        img = copy.deepcopy(self.img)
        self.img_mutex.release()
        if img is None:
            return

        if not self.TEST_MODE:
            if not self.on_a_mission:
                img[:,:,1] = img[:,:,2] // 3
                img[:,:,2] = img[:,:,1] // 3
            elif self.on_a_mission and not self.started_mission:
                img[:,:,0] = img[:,:,0] // 3
                img[:,:,2] = img[:,:,2] // 3
        self.ui.label.setPixmap(QPixmap(QImage(img.data, img.shape[1], img.shape[0], QtGui.QImage.Format_RGB888)))


    #
    # SUBSCRIPTION to gotobjects method from ObjectDetector interface
    #
    def ObjectDetector_gotobjects(self, lst):
        #
        #get robots position
        self.object_data = lst
        for obj in lst:
            if obj.id == -2:
                self.robot_coordinates =  [obj.x, obj.y]
        #
        #pass

    #
    # SUBSCRIPTION to gotinteractions method from InteractionDetector interface
    #
    def InteractionDetector_gotinteractions(self, lst):
        self.interaction_data = lst


    #
    # SUBSCRIPTION to gotpeople method from PeopleDetector interface
    #
    def PeopleDetector_gotpeople(self, lst):
        self.people_data = lst


    #
    # SUBSCRIPTION to gotwalls method from WallDetector interface
    #
    def WallDetector_gotwalls(self, lst):
        self.wall_data = lst


    #
    # SUBSCRIPTION to newsequence method from ByteSequencePublisher interface
    #
    def ByteSequencePublisher_newsequence(self, bs):
        # print('GOT NEW BYTE SEQUENCE', bs)
        # bs = np.load(bs, allow_pickle=True)
        self.img_mutex.acquire()
        self.img = pickle.loads(bs)
        self.img_mutex.release()


    #
    # SUBSCRIPTION to goalupdated method from GoalPublisher interface
    #
    def GoalPublisher_goalupdated(self, goal):
        if self.on_a_mission:
            self.new_goal = True
        self.goal_data = goal
        self.goal_coordinates = [goal.x,  goal.y]
        #pass


    def JoystickAdapter_sendData(self, data):
        if self.on_a_mission:
            self.started_mission = True

        adv = -3.5*data.axes[1].value # m/s
        rot = 4.0*data.axes[0].value  # rad/s
        self.speed_command = [adv, 0, rot]
        self.omnirobot_proxy.setSpeedBase(adv, 0., rot)



    def info(self):
	    dic = {"min_humans":self.configGui.nhumans_min.value(), "max_humans":self.configGui.nhumans_max.value(),
                   "min_wandering_humans":self.configGui.nwandhumans_min.value(), "max_wandering_humans":self.configGui.nwandhumans_max.value()
                   ,"min_plants":self.configGui.nplants_min.value(), "max_plants":self.configGui.nplants_max.value(),
                   "min_tables":self.configGui.ntables_min.value(), "max_tables":self.configGui.ntables_max.value(), 
                   "min_relations":self.configGui.nrelations_min.value(), "max_relations":self.configGui.nrelations_max.value()}
	    self.relations_dic= json.dumps(dic)



