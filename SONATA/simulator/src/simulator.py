#!/usr/bin/env python3
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

# \mainpage RoboComp::simulator
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
# Just: "${PATH_TO_BINARY}/simulator --Ice.Config=${PATH_TO_CONFIG_FILE}"
#
# \subsection running_ssec Once running
#
#
#

import sys, traceback, IceStorm, time, os, copy

# Ctrl+c handling
import signal

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
    global worker
    worker.stop_simulation()
    del worker

if __name__ == '__main__':
    params = copy.deepcopy(sys.argv)
    if len(params) > 1:
        if not params[1].startswith('--Ice.Config='):
            params[1] = '--Ice.Config=' + params[1]
    elif len(params) == 1:
        params.append('--Ice.Config=config')
    ic = Ice.initialize(params)
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
        print('Cannot connect to IceStorm! ('+proxy+')')
        status = 1

    # Create a proxy to publish a PeopleDetector topic
    topic = False
    try:
        topic = topicManager.retrieve("PeopleDetector")
    except:
        pass
    while not topic:
        try:
            topic = topicManager.retrieve("PeopleDetector")
        except IceStorm.NoSuchTopic:
            try:
                topic = topicManager.create("PeopleDetector")
            except:
                print('Another client created the PeopleDetector topic? ...')
    pub = topic.getPublisher().ice_oneway()
    peopledetectorTopic = PeopleDetectorPrx.uncheckedCast(pub)
    mprx["PeopleDetectorPub"] = peopledetectorTopic

    # Create a proxy to publish a ObjectDetector topic
    topic = False
    try:
        topic = topicManager.retrieve("ObjectDetector")
    except:
        pass
    while not topic:
        try:
            topic = topicManager.retrieve("ObjectDetector")
        except IceStorm.NoSuchTopic:
            try:
                topic = topicManager.create("ObjectDetector")
            except:
                print('Another client created the ObjectDetector topic? ...')
    pub = topic.getPublisher().ice_oneway()
    objectdetectorTopic = ObjectDetectorPrx.uncheckedCast(pub)
    mprx["ObjectDetectorPub"] = objectdetectorTopic


    # Create a proxy to publish a InteractionDetector topic
    topic = False
    try:
        topic = topicManager.retrieve("InteractionDetector")
    except:
        pass
    while not topic:
        try:
            topic = topicManager.retrieve("InteractionDetector")
        except IceStorm.NoSuchTopic:
            try:
                topic = topicManager.create("InterationDetector")
            except:
                print('Another client created the InteractionDetector topic? ...')
    pub = topic.getPublisher().ice_oneway()
    interactiondetectorTopic = InteractionDetectorPrx.uncheckedCast(pub)
    mprx["InteractionDetectorPub"] = interactiondetectorTopic


    # Create a proxy to publish a WallDetector topic
    topic = False
    try:
        topic = topicManager.retrieve("WallDetector")
    except:
        pass
    while not topic:
        try:
            topic = topicManager.retrieve("WallDetector")
        except IceStorm.NoSuchTopic:
            try:
                topic = topicManager.create("WallDetector")
            except:
                print('Another client created the WallDetector topic? ...')
    pub = topic.getPublisher().ice_oneway()
    walldetectorTopic = WallDetectorPrx.uncheckedCast(pub)
    mprx["WallDetectorPub"] = walldetectorTopic


    # Create a proxy to publish a ByteSequencePublisher topic
    topic = False
    try:
        topic = topicManager.retrieve("ByteSequencePublisher")
    except:
        pass
    while not topic:
        try:
            topic = topicManager.retrieve("ByteSequencePublisher")
        except IceStorm.NoSuchTopic:
            try:
                topic = topicManager.create("ByteSequencePublisher")
            except:
                print('Another client created the ByteSequencePublisher topic? ...')
    pub = topic.getPublisher().ice_oneway()
    bytesequencepublisherTopic = ByteSequencePublisherPrx.uncheckedCast(pub)
    mprx["ByteSequencePublisherPub"] = bytesequencepublisherTopic


    # Create a proxy to publish a GoalPublisher topic
    topic = False
    try:
        topic = topicManager.retrieve("GoalPublisher")
    except:
        pass
    while not topic:
        try:
            topic = topicManager.retrieve("GoalPublisher")
        except IceStorm.NoSuchTopic:
            try:
                topic = topicManager.create("GoalPublisher")
            except:
                print('Another client created the GoalPublisher topic? ...')
    pub = topic.getPublisher().ice_oneway()
    goalpublisherTopic = GoalPublisherPrx.uncheckedCast(pub)
    mprx["GoalPublisherPub"] = goalpublisherTopic


    global worker
    if status == 0:
        worker = SpecificWorker(mprx)
        worker.setParams(parameters)
    else:
        print("Error getting required connections, check config file")
        sys.exit(-1)

    adapter = ic.createObjectAdapter('OmniRobot')
    adapter.add(OmniRobotI(worker), ic.stringToIdentity('omnirobot'))
    adapter.activate()

    adapter2 = ic.createObjectAdapter('Simulator')
    adapter2.add(SimulatorI(worker), ic.stringToIdentity('simulator'))
    adapter2.activate()

    signal.signal(signal.SIGINT, sigint_handler)
    signal.signal(signal.SIGTERM, sigint_handler)

    worker.compute()

    if ic:
        try:
            ic.destroy()
        except:
            traceback.print_exc()
            status = 1

    sys.exit(0)
