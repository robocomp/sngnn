#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
#    Copyright (C) 2020 by YOUR NAME HERE
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

# \mainpage RoboComp::controller
#
# \section intro_sec Introduction
#
# Some information about the component...
#
# \section interface_sec Interface
#
# Descroption of the interface provided...
#
# \section install_sec Installation
#
# \subsection install1_ssec Software depencences
# Software dependences....
#
# \subsection install2_ssec Compile and install
# How to compile/install the component...
#
# \section guide_sec User guide
#
# \subsection config_ssec Configuration file
#
# <p>
# The configuration file...
# </p>
#
# \subsection execution_ssec Execution
#
# Just: "${PATH_TO_BINARY}/controller --Ice.Config=${PATH_TO_CONFIG_FILE}"
#
# \subsection running_ssec Once running
#
#
#

import sys
import traceback
import IceStorm
import time
import os
import copy
import argparse
from termcolor import colored
# Ctrl+c handling
import signal

from PySide2 import QtCore
from PySide2 import QtWidgets

from specificworker import *


class CommonBehaviorI(RoboCompCommonBehavior.CommonBehavior):
    def __init__(self, _handler):
        self.handler = _handler
    def getFreq(self, current = None):
        self.handler.getFreq()
    def setFreq(self, freq, current = None):
        self.handler.setFreq()
    def timeAwake(self, current = None):
        try:
            return self.handler.timeAwake()
        except:
            print('Problem getting timeAwake')
    def killYourSelf(self, current = None):
        self.handler.killYourSelf()
    def getAttrList(self, current = None):
        try:
            return self.handler.getAttrList()
        except:
            print('Problem getting getAttrList')
            traceback.print_exc()
            status = 1
            return

#SIGNALS handler
def sigint_handler(*args):
    QtCore.QCoreApplication.quit()
    
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    parser = argparse.ArgumentParser()
    parser.add_argument('iceconfigfile', nargs='?', type=str, default='etc/config')
    parser.add_argument('--startup-check', action='store_true')

    args = parser.parse_args()

    ic = Ice.initialize(args.iceconfigfile)
    status = 0
    mprx = {}
    parameters = {}
    for i in ic.getProperties():
        parameters[str(i)] = str(ic.getProperties().getProperty(i))
    # Topic Manager
    proxy = ic.getProperties().getProperty("TopicManager.Proxy")
    obj = ic.stringToProxy(proxy)
    try:
        topicManager = IceStorm.TopicManagerPrx.checkedCast(obj)
    except Ice.ConnectionRefusedException as e:
        print(colored('Cannot connect to rcnode! This must be running to use pub/sub.', 'red'))
        exit(1)

    # Remote object connection for OmniRobot
    try:
        proxyString = ic.getProperties().getProperty('OmniRobotProxy')
        try:
            basePrx = ic.stringToProxy(proxyString)
            omnirobot_proxy = RoboCompOmniRobot.OmniRobotPrx.uncheckedCast(basePrx)
            mprx["OmniRobotProxy"] = omnirobot_proxy
        except Ice.Exception:
            print('Cannot connect to the remote object (OmniRobot)', proxyString)
            #traceback.print_exc()
            status = 1
    except Ice.Exception as e:
        print(e)
        print('Cannot get OmniRobotProxy property.')
        status = 1

    fullscreen = ic.getProperties().getProperty("FULL_SCREEN")

    # Remote object connection for Simulator
    try:
        proxyString = ic.getProperties().getProperty('SimulatorProxy')
        try:
            basePrx = ic.stringToProxy(proxyString)
            simulator_proxy = RoboCompSimulator.SimulatorPrx.uncheckedCast(basePrx)
            mprx["SimulatorProxy"] = simulator_proxy
        except Ice.Exception:
            print('Cannot connect to the remote object (Simulator)', proxyString)
            #traceback.print_exc()
            status = 1
    except Ice.Exception as e:
        print(e)
        print('Cannot get SimulatorProxy property.')
        status = 1

    if status == 0:
        worker = SpecificWorker(mprx, fullscreen, args.startup_check)
        worker.setParams(parameters)
    else:
        print("Error getting required connections, check config file")
        sys.exit(-1)


    ByteSequencePublisher_adapter = ic.createObjectAdapter("ByteSequencePublisherTopic")
    bytesequencepublisherI_ = bytesequencepublisherI.ByteSequencePublisherI(worker)
    bytesequencepublisher_proxy = ByteSequencePublisher_adapter.addWithUUID(bytesequencepublisherI_).ice_oneway()

    subscribeDone = False
    while not subscribeDone:
        try:
            bytesequencepublisher_topic = topicManager.retrieve("ByteSequencePublisher")
            subscribeDone = True
        except Ice.Exception as e:
            print("Error. Topic does not exist (creating)")
            time.sleep(1)
            try:
                bytesequencepublisher_topic = topicManager.create("ByteSequencePublisher")
                subscribeDone = True
            except:
                print("Error. Topic could not be created. Exiting")
                status = 0
    qos = {}
    bytesequencepublisher_topic.subscribeAndGetPublisher(qos, bytesequencepublisher_proxy)
    ByteSequencePublisher_adapter.activate()


    GoalPublisher_adapter = ic.createObjectAdapter("GoalPublisherTopic")
    goalpublisherI_ = goalpublisherI.GoalPublisherI(worker)
    goalpublisher_proxy = GoalPublisher_adapter.addWithUUID(goalpublisherI_).ice_oneway()

    subscribeDone = False
    while not subscribeDone:
        try:
            goalpublisher_topic = topicManager.retrieve("GoalPublisher")
            subscribeDone = True
        except Ice.Exception as e:
            print("Error. Topic does not exist (creating)")
            time.sleep(1)
            try:
                goalpublisher_topic = topicManager.create("GoalPublisher")
                subscribeDone = True
            except:
                print("Error. Topic could not be created. Exiting")
                status = 0
    qos = {}
    goalpublisher_topic.subscribeAndGetPublisher(qos, goalpublisher_proxy)
    GoalPublisher_adapter.activate()


    InteractionDetector_adapter = ic.createObjectAdapter("InteractionDetectorTopic")
    interactiondetectorI_ = interactiondetectorI.InteractionDetectorI(worker)
    interactiondetector_proxy = InteractionDetector_adapter.addWithUUID(interactiondetectorI_).ice_oneway()

    subscribeDone = False
    while not subscribeDone:
        try:
            interactiondetector_topic = topicManager.retrieve("InteractionDetector")
            subscribeDone = True
        except Ice.Exception as e:
            print("Error. Topic does not exist (creating)")
            time.sleep(1)
            try:
                interactiondetector_topic = topicManager.create("InteractionDetector")
                subscribeDone = True
            except:
                print("Error. Topic could not be created. Exiting")
                status = 0
    qos = {}
    interactiondetector_topic.subscribeAndGetPublisher(qos, interactiondetector_proxy)
    InteractionDetector_adapter.activate()


    JoystickAdapter_adapter = ic.createObjectAdapter("JoystickAdapterTopic")
    joystickadapterI_ = joystickadapterI.JoystickAdapterI(worker)
    joystickadapter_proxy = JoystickAdapter_adapter.addWithUUID(joystickadapterI_).ice_oneway()

    subscribeDone = False
    while not subscribeDone:
        try:
            joystickadapter_topic = topicManager.retrieve("JoystickAdapter")
            subscribeDone = True
        except Ice.Exception as e:
            print("Error. Topic does not exist (creating)")
            time.sleep(1)
            try:
                joystickadapter_topic = topicManager.create("JoystickAdapter")
                subscribeDone = True
            except:
                print("Error. Topic could not be created. Exiting")
                status = 0
    qos = {}
    joystickadapter_topic.subscribeAndGetPublisher(qos, joystickadapter_proxy)
    JoystickAdapter_adapter.activate()


    ObjectDetector_adapter = ic.createObjectAdapter("ObjectDetectorTopic")
    objectdetectorI_ = objectdetectorI.ObjectDetectorI(worker)
    objectdetector_proxy = ObjectDetector_adapter.addWithUUID(objectdetectorI_).ice_oneway()

    subscribeDone = False
    while not subscribeDone:
        try:
            objectdetector_topic = topicManager.retrieve("ObjectDetector")
            subscribeDone = True
        except Ice.Exception as e:
            print("Error. Topic does not exist (creating)")
            time.sleep(1)
            try:
                objectdetector_topic = topicManager.create("ObjectDetector")
                subscribeDone = True
            except:
                print("Error. Topic could not be created. Exiting")
                status = 0
    qos = {}
    objectdetector_topic.subscribeAndGetPublisher(qos, objectdetector_proxy)
    ObjectDetector_adapter.activate()


    PeopleDetector_adapter = ic.createObjectAdapter("PeopleDetectorTopic")
    peopledetectorI_ = peopledetectorI.PeopleDetectorI(worker)
    peopledetector_proxy = PeopleDetector_adapter.addWithUUID(peopledetectorI_).ice_oneway()

    subscribeDone = False
    while not subscribeDone:
        try:
            peopledetector_topic = topicManager.retrieve("PeopleDetector")
            subscribeDone = True
        except Ice.Exception as e:
            print("Error. Topic does not exist (creating)")
            time.sleep(1)
            try:
                peopledetector_topic = topicManager.create("PeopleDetector")
                subscribeDone = True
            except:
                print("Error. Topic could not be created. Exiting")
                status = 0
    qos = {}
    peopledetector_topic.subscribeAndGetPublisher(qos, peopledetector_proxy)
    PeopleDetector_adapter.activate()


    WallDetector_adapter = ic.createObjectAdapter("WallDetectorTopic")
    walldetectorI_ = walldetectorI.WallDetectorI(worker)
    walldetector_proxy = WallDetector_adapter.addWithUUID(walldetectorI_).ice_oneway()

    subscribeDone = False
    while not subscribeDone:
        try:
            walldetector_topic = topicManager.retrieve("WallDetector")
            subscribeDone = True
        except Ice.Exception as e:
            print("Error. Topic does not exist (creating)")
            time.sleep(1)
            try:
                walldetector_topic = topicManager.create("WallDetector")
                subscribeDone = True
            except:
                print("Error. Topic could not be created. Exiting")
                status = 0
    qos = {}
    walldetector_topic.subscribeAndGetPublisher(qos, walldetector_proxy)
    WallDetector_adapter.activate()

    signal.signal(signal.SIGINT, sigint_handler)
    signal.signal(signal.SIGTERM, sigint_handler)
    app.exec_()

    if ic:
        # try:
        ic.destroy()
        # except:
        #     traceback.print_exc()
        #     status = 1
