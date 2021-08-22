#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import traceback
import IceStorm
import time
import os
import copy
from termcolor import colored
import torch
from PySide2 import QtCore

# Ctrl+c handling
import signal

sys.path.append('src/')
from specificworker import *

#SIGNALS handler
def sigint_handler(*args):
    QtCore.QCoreApplication.quit()
    exit()

    
if __name__ == '__main__':
    app = QtCore.QCoreApplication(sys.argv)

    print('Using "etc/config" as configuration file.')
    ic = Ice.initialize('etc/config')
    status = 0
    mprx = {}
    parameters = {}
    for i in ic.getProperties():
        parameters[str(i)] = str(ic.getProperties().getProperty(i))
    # Topic Manager
    proxy = ic.getProperties().getProperty("TopicManager.Proxy")
    obj = ic.stringToProxy(proxy)
    max_attempts = 10
    for i in range(max_attempts):
        try:
            topicManager = IceStorm.TopicManagerPrx.checkedCast(obj)
        except Ice.ConnectionRefusedException as e:
            i1 = i+1
            w = 0.015*i1*i1
            if i1 == max_attempts:
                print(colored('Cannot connect to icebox! This must be running to use pub/sub.', 'red'))
                print('We tried using proxy:', str(proxy))
                exit(1)
            else:
                print(colored('Cannot connect to icebox! This must be running to use pub/sub. Waiting '+str(w)+' seconds', 'red'))
                print(colored('Make sure you started the simulation first.', 'red'))

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
        worker = SpecificWorker(mprx, fullscreen, False)
        worker.setParams(parameters)
    else:
        print("Error getting required connections, check config file")
        sys.exit(-1)


    ByteSequencePublisher_adapter = ic.createObjectAdapter("ByteSequencePublisherTopic")
    bytesequencepublisherI_ = bytesequencepublisherI.ByteSequencePublisherI(worker)
    bytesequencepublisher_proxy = ByteSequencePublisher_adapter.addWithUUID(bytesequencepublisherI_).ice_oneway()

    try:
        bytesequencepublisher_topic = topicManager.retrieve("ByteSequencePublisher")
    except Ice.Exception as e:
        print("Creating topic ByteSequencePublisher")
        time.sleep(1)
        try:
            bytesequencepublisher_topic = topicManager.create("ByteSequencePublisher")
        except:
            print("Error. Topic could not be created. Exiting")
            status = 0
    qos = {}
    bytesequencepublisher_topic.subscribeAndGetPublisher(qos, bytesequencepublisher_proxy)
    ByteSequencePublisher_adapter.activate()


    GoalPublisher_adapter = ic.createObjectAdapter("GoalPublisherTopic")
    goalpublisherI_ = goalpublisherI.GoalPublisherI(worker)
    goalpublisher_proxy = GoalPublisher_adapter.addWithUUID(goalpublisherI_).ice_oneway()

    try:
        goalpublisher_topic = topicManager.retrieve("GoalPublisher")
    except Ice.Exception as e:
        print("Creating topic GoalPublisher")
        time.sleep(1)
        try:
            goalpublisher_topic = topicManager.create("GoalPublisher")
        except:
            print("Error. Topic could not be created. Exiting")
            status = 0
    qos = {}
    goalpublisher_topic.subscribeAndGetPublisher(qos, goalpublisher_proxy)
    GoalPublisher_adapter.activate()


    InteractionDetector_adapter = ic.createObjectAdapter("InteractionDetectorTopic")
    interactiondetectorI_ = interactiondetectorI.InteractionDetectorI(worker)
    interactiondetector_proxy = InteractionDetector_adapter.addWithUUID(interactiondetectorI_).ice_oneway()

    try:
        interactiondetector_topic = topicManager.retrieve("InteractionDetector")
    except Ice.Exception as e:
        print("Creating topic InteractionDetector")
        time.sleep(1)
        try:
            interactiondetector_topic = topicManager.create("InteractionDetector")
        except:
            print("Error. Topic could not be created. Exiting")
            status = 0
    qos = {}
    interactiondetector_topic.subscribeAndGetPublisher(qos, interactiondetector_proxy)
    InteractionDetector_adapter.activate()

    JoystickAdapter_adapter = ic.createObjectAdapter("JoystickAdapterTopic")
    joystickadapterI_ = joystickadapterI.JoystickAdapterI(worker)
    joystickadapter_proxy = JoystickAdapter_adapter.addWithUUID(joystickadapterI_).ice_oneway()

    try:
        joystickadapter_topic = topicManager.retrieve("JoystickAdapter")
    except Ice.Exception as e:
        print("Creating topic JoystickAdapter")
        time.sleep(1)
        try:
            joystickadapter_topic = topicManager.create("JoystickAdapter")
        except:
            print("Error. Topic could not be created. Exiting")
            status = 0
    qos = {}
    joystickadapter_topic.subscribeAndGetPublisher(qos, joystickadapter_proxy)
    JoystickAdapter_adapter.activate()


    ObjectDetector_adapter = ic.createObjectAdapter("ObjectDetectorTopic")
    objectdetectorI_ = objectdetectorI.ObjectDetectorI(worker)
    objectdetector_proxy = ObjectDetector_adapter.addWithUUID(objectdetectorI_).ice_oneway()

    try:
        objectdetector_topic = topicManager.retrieve("ObjectDetector")
    except Ice.Exception as e:
        print("Creating topic ObjectDetector")
        time.sleep(1)
        try:
            objectdetector_topic = topicManager.create("ObjectDetector")
        except:
            print("Error. Topic could not be created. Exiting")
            status = 0
    qos = {}
    objectdetector_topic.subscribeAndGetPublisher(qos, objectdetector_proxy)
    ObjectDetector_adapter.activate()

    PeopleDetector_adapter = ic.createObjectAdapter("PeopleDetectorTopic")
    peopledetectorI_ = peopledetectorI.PeopleDetectorI(worker)
    peopledetector_proxy = PeopleDetector_adapter.addWithUUID(peopledetectorI_).ice_oneway()

    try:
        peopledetector_topic = topicManager.retrieve("PeopleDetector")
    except Ice.Exception as e:
        print("Creating topic PeopleDetector")
        time.sleep(1)
        try:
            peopledetector_topic = topicManager.create("PeopleDetector")
        except:
            print("Error. Topic could not be created. Exiting")
            status = 0
    qos = {}
    peopledetector_topic.subscribeAndGetPublisher(qos, peopledetector_proxy)
    PeopleDetector_adapter.activate()


    WallDetector_adapter = ic.createObjectAdapter("WallDetectorTopic")
    walldetectorI_ = walldetectorI.WallDetectorI(worker)
    walldetector_proxy = WallDetector_adapter.addWithUUID(walldetectorI_).ice_oneway()

    try:
        walldetector_topic = topicManager.retrieve("WallDetector")
    except Ice.Exception as e:
        print("Creating topic WallDetector")
        time.sleep(1)
        try:
            walldetector_topic = topicManager.create("WallDetector")
        except:
            print("Error. Topic could not be created. Exiting")
            status = 0
    qos = {}
    walldetector_topic.subscribeAndGetPublisher(qos, walldetector_proxy)
    WallDetector_adapter.activate()

    signal.signal(signal.SIGINT, sigint_handler)
    signal.signal(signal.SIGTERM, sigint_handler)
    
    worker.set_app(app)

    if 'profile' in sys.argv:
        print("RUNNING IN PROFILING MODE!!\n"*10)
        import cProfile
        #cProfile.run("app.exec_()", filename="pyprof", sort=-1)
        #pyprof2calltree -i pyprof -k
    else:
        app.exec_()

    if ic:
        # try:
        ic.destroy()
        # except:
        #     traceback.print_exc()
        #     status = 1
