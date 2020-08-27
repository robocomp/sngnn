#!/usr/bin/python3
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

import sys, Ice, os
from PySide2 import QtWidgets, QtCore

ROBOCOMP = ''
try:
    ROBOCOMP = os.environ['ROBOCOMP']
except KeyError:
    print('$ROBOCOMP environment variable not set, using the default value /opt/robocomp')
    ROBOCOMP = '/opt/robocomp'

Ice.loadSlice("-I ./src/ --all ./src/CommonBehavior.ice")
import RoboCompCommonBehavior

Ice.loadSlice("-I ./src/ --all ./src/ByteSequencePublisher.ice")
import RoboCompByteSequencePublisher
Ice.loadSlice("-I ./src/ --all ./src/GenericBase.ice")
import RoboCompGenericBase
Ice.loadSlice("-I ./src/ --all ./src/GoalPublisher.ice")
import RoboCompGoalPublisher
Ice.loadSlice("-I ./src/ --all ./src/InteractionDetector.ice")
import RoboCompInteractionDetector
Ice.loadSlice("-I ./src/ --all ./src/JoystickAdapter.ice")
import RoboCompJoystickAdapter
Ice.loadSlice("-I ./src/ --all ./src/ObjectDetector.ice")
import RoboCompObjectDetector
Ice.loadSlice("-I ./src/ --all ./src/OmniRobot.ice")
import RoboCompOmniRobot
Ice.loadSlice("-I ./src/ --all ./src/PeopleDetector.ice")
import RoboCompPeopleDetector
Ice.loadSlice("-I ./src/ --all ./src/Simulator.ice")
import RoboCompSimulator
Ice.loadSlice("-I ./src/ --all ./src/WallDetector.ice")
import RoboCompWallDetector

class bytesequence(list):
    def __init__(self, iterable=list()):
        super(bytesequence, self).__init__(iterable)

    def append(self, item):
        assert isinstance(item, byte)
        super(bytesequence, self).append(item)

    def extend(self, iterable):
        for item in iterable:
            assert isinstance(item, byte)
        super(bytesequence, self).extend(iterable)

    def insert(self, index, item):
        assert isinstance(item, byte)
        super(bytesequence, self).insert(index, item)

setattr(RoboCompByteSequencePublisher, "bytesequence", bytesequence)

class InteractionList(list):
    def __init__(self, iterable=list()):
        super(InteractionList, self).__init__(iterable)

    def append(self, item):
        assert isinstance(item, RoboCompInteractionDetector.InteractionT)
        super(InteractionList, self).append(item)

    def extend(self, iterable):
        for item in iterable:
            assert isinstance(item, RoboCompInteractionDetector.InteractionT)
        super(InteractionList, self).extend(iterable)

    def insert(self, index, item):
        assert isinstance(item, RoboCompInteractionDetector.InteractionT)
        super(InteractionList, self).insert(index, item)

setattr(RoboCompInteractionDetector, "InteractionList", InteractionList)

class AxisList(list):
    def __init__(self, iterable=list()):
        super(AxisList, self).__init__(iterable)

    def append(self, item):
        assert isinstance(item, RoboCompJoystickAdapter.AxisParams)
        super(AxisList, self).append(item)

    def extend(self, iterable):
        for item in iterable:
            assert isinstance(item, RoboCompJoystickAdapter.AxisParams)
        super(AxisList, self).extend(iterable)

    def insert(self, index, item):
        assert isinstance(item, RoboCompJoystickAdapter.AxisParams)
        super(AxisList, self).insert(index, item)

setattr(RoboCompJoystickAdapter, "AxisList", AxisList)

class ButtonsList(list):
    def __init__(self, iterable=list()):
        super(ButtonsList, self).__init__(iterable)

    def append(self, item):
        assert isinstance(item, RoboCompJoystickAdapter.ButtonParams)
        super(ButtonsList, self).append(item)

    def extend(self, iterable):
        for item in iterable:
            assert isinstance(item, RoboCompJoystickAdapter.ButtonParams)
        super(ButtonsList, self).extend(iterable)

    def insert(self, index, item):
        assert isinstance(item, RoboCompJoystickAdapter.ButtonParams)
        super(ButtonsList, self).insert(index, item)

setattr(RoboCompJoystickAdapter, "ButtonsList", ButtonsList)

class ObjectList(list):
    def __init__(self, iterable=list()):
        super(ObjectList, self).__init__(iterable)

    def append(self, item):
        assert isinstance(item, RoboCompObjectDetector.ObjectT)
        super(ObjectList, self).append(item)

    def extend(self, iterable):
        for item in iterable:
            assert isinstance(item, RoboCompObjectDetector.ObjectT)
        super(ObjectList, self).extend(iterable)

    def insert(self, index, item):
        assert isinstance(item, RoboCompObjectDetector.ObjectT)
        super(ObjectList, self).insert(index, item)

setattr(RoboCompObjectDetector, "ObjectList", ObjectList)

class PeopleList(list):
    def __init__(self, iterable=list()):
        super(PeopleList, self).__init__(iterable)

    def append(self, item):
        assert isinstance(item, RoboCompPeopleDetector.Person)
        super(PeopleList, self).append(item)

    def extend(self, iterable):
        for item in iterable:
            assert isinstance(item, RoboCompPeopleDetector.Person)
        super(PeopleList, self).extend(iterable)

    def insert(self, index, item):
        assert isinstance(item, RoboCompPeopleDetector.Person)
        super(PeopleList, self).insert(index, item)

setattr(RoboCompPeopleDetector, "PeopleList", PeopleList)

class WallList(list):
    def __init__(self, iterable=list()):
        super(WallList, self).__init__(iterable)

    def append(self, item):
        assert isinstance(item, RoboCompWallDetector.WallT)
        super(WallList, self).append(item)

    def extend(self, iterable):
        for item in iterable:
            assert isinstance(item, RoboCompWallDetector.WallT)
        super(WallList, self).extend(iterable)

    def insert(self, index, item):
        assert isinstance(item, RoboCompWallDetector.WallT)
        super(WallList, self).insert(index, item)

setattr(RoboCompWallDetector, "WallList", WallList)


import bytesequencepublisherI
import goalpublisherI
import interactiondetectorI
import joystickadapterI
import objectdetectorI
import peopledetectorI
import walldetectorI


try:
    from ui_mainUI import *
except:
    print("Can't import UI file. Did you run 'make'?")
    sys.exit(-1)



class GenericWorker(QtWidgets.QWidget):

    kill = QtCore.Signal()

    def __init__(self, mprx):
        super(GenericWorker, self).__init__()

        self.omnirobot_proxy = mprx["OmniRobotProxy"]
        self.simulator_proxy = mprx["SimulatorProxy"]

        self.ui = Ui_guiDlg()
        self.ui.setupUi(self)
        self.show()

        self.mutex = QtCore.QMutex(QtCore.QMutex.Recursive)
        self.Period = 30
        self.timer = QtCore.QTimer(self)


    @QtCore.Slot()
    def killYourSelf(self):
        rDebug("Killing myself")
        self.kill.emit()

    # \brief Change compute period
    # @param per Period in ms
    @QtCore.Slot(int)
    def setPeriod(self, p):
        print("Period changed", p)
        self.Period = p
        self.timer.start(self.Period)
