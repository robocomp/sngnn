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

import sys, Ice, os

ROBOCOMP = ''
try:
	ROBOCOMP = os.environ['ROBOCOMP']
except KeyError:
	print('$ROBOCOMP environment variable not set, using the default value /opt/robocomp')
	ROBOCOMP = '/opt/robocomp'

preStr = "-I/opt/robocomp/interfaces/ -I"+ROBOCOMP+"/interfaces/ --all /opt/robocomp/interfaces/"
Ice.loadSlice(preStr+"CommonBehavior.ice")
import RoboCompCommonBehavior

additionalPathStr = ''
icePaths = [ '/opt/robocomp/interfaces' ]
try:
	SLICE_PATH = os.environ['SLICE_PATH'].split(':')
	for p in SLICE_PATH:
		icePaths.append(p)
		additionalPathStr += ' -I' + p + ' '
	icePaths.append('/opt/robocomp/interfaces')
except:
	print('SLICE_PATH environment variable was not exported. Using only the default paths')
	pass

ice_OmniRobot = False
for p in icePaths:
	if os.path.isfile(p+'/OmniRobot.ice'):
		preStr = "-I/opt/robocomp/interfaces/ -I"+ROBOCOMP+"/interfaces/ " + additionalPathStr + " --all "+p+'/'
		wholeStr = preStr+"OmniRobot.ice"
		Ice.loadSlice(wholeStr)
		ice_OmniRobot = True
		break
if not ice_OmniRobot:
	print('Couln\'t load OmniRobot')
	sys.exit(-1)
from RoboCompOmniRobot import *
ice_Simulator = False
for p in icePaths:
	if os.path.isfile(p+'/Simulator.ice'):
		preStr = "-I/opt/robocomp/interfaces/ -I"+ROBOCOMP+"/interfaces/ " + additionalPathStr + " --all "+p+'/'
		wholeStr = preStr+"Simulator.ice"
		Ice.loadSlice(wholeStr)
		ice_Simulator = True
		break
if not ice_Simulator:
	print('Couln\'t load Simulator')
	sys.exit(-1)
from RoboCompSimulator import *
ice_PeopleDetector = False
for p in icePaths:
	if os.path.isfile(p+'/PeopleDetector.ice'):
		preStr = "-I/opt/robocomp/interfaces/ -I"+ROBOCOMP+"/interfaces/ " + additionalPathStr + " --all "+p+'/'
		wholeStr = preStr+"PeopleDetector.ice"
		Ice.loadSlice(wholeStr)
		ice_PeopleDetector = True
		break
if not ice_PeopleDetector:
	print('Couln\'t load PeopleDetector')
	sys.exit(-1)
from RoboCompPeopleDetector import *


ice_ObjectDetector = False
for p in icePaths:
	if os.path.isfile(p+'/ObjectDetector.ice'):
		preStr = "-I/opt/robocomp/interfaces/ -I"+ROBOCOMP+"/interfaces/ " + additionalPathStr + " --all "+p+'/'
		wholeStr = preStr+"ObjectDetector.ice"
		Ice.loadSlice(wholeStr)
		ice_ObjectDetector = True
		break
if not ice_ObjectDetector:
	print('Couln\'t load ObjectDetector')
	sys.exit(-1)
from RoboCompObjectDetector import *


ice_InteractionDetector = False
for p in icePaths:
	if os.path.isfile(p+'/InteractionDetector.ice'):
		preStr = "-I/opt/robocomp/interfaces/ -I"+ROBOCOMP+"/interfaces/ " + additionalPathStr + " --all "+p+'/'
		wholeStr = preStr+"InteractionDetector.ice"
		Ice.loadSlice(wholeStr)
		ice_InteractionDetector = True
		break
if not ice_InteractionDetector:
	print('Couln\'t load InteractionDetector')
	sys.exit(-1)
from RoboCompInteractionDetector import *


ice_WallDetector = False
for p in icePaths:
	if os.path.isfile(p+'/WallDetector.ice'):
		preStr = "-I/opt/robocomp/interfaces/ -I"+ROBOCOMP+"/interfaces/ " + additionalPathStr + " --all "+p+'/'
		wholeStr = preStr+"WallDetector.ice"
		Ice.loadSlice(wholeStr)
		ice_WallDetector = True
		break
if not ice_WallDetector:
	print('Couln\'t load WallDetector')
	sys.exit(-1)
from RoboCompWallDetector import *




ice_ByteSequencePublisher = False
for p in icePaths:
	if os.path.isfile(p+'/ByteSequencePublisher.ice'):
		preStr = "-I/opt/robocomp/interfaces/ -I"+ROBOCOMP+"/interfaces/ " + additionalPathStr + " --all "+p+'/'
		wholeStr = preStr+"ByteSequencePublisher.ice"
		Ice.loadSlice(wholeStr)
		ice_ByteSequencePublisher = True
		break
if not ice_ByteSequencePublisher:
	print('Couln\'t load ByteSequencePublisher')
	sys.exit(-1)
from RoboCompByteSequencePublisher import *

ice_GoalPublisher = False
for p in icePaths:
	if os.path.isfile(p+'/GoalPublisher.ice'):
		preStr = "-I/opt/robocomp/interfaces/ -I"+ROBOCOMP+"/interfaces/ " + additionalPathStr + " --all "+p+'/'
		wholeStr = preStr+"GoalPublisher.ice"
		Ice.loadSlice(wholeStr)
		ice_GoalPublisher = True
		break
if not ice_GoalPublisher:
	print('Couln\'t load GoalPublisher')
	sys.exit(-1)
from RoboCompGoalPublisher import *




ice_GenericBase = False
for p in icePaths:
	if os.path.isfile(p+'/GenericBase.ice'):
		preStr = "-I/opt/robocomp/interfaces/ -I"+ROBOCOMP+"/interfaces/ " + additionalPathStr + " --all "+p+'/'
		wholeStr = preStr+"GenericBase.ice"
		Ice.loadSlice(wholeStr)
		ice_GenericBase = True
		break
if not ice_GenericBase:
	print('Couln\'t load GenericBase')
	sys.exit(-1)
from RoboCompGenericBase import *


from omnirobotI import *
from simulatorI import *


class GenericWorker(object):

	def __init__(self, mprx):
		super(GenericWorker, self).__init__()


		self.peopledetector_proxy = mprx["PeopleDetectorPub"]
		self.objectdetector_proxy = mprx["ObjectDetectorPub"]
		self.interactiondetector_proxy = mprx["InteractionDetectorPub"]
		self.walldetector_proxy = mprx["WallDetectorPub"]
		self.bytesequencepublisher_proxy = mprx["ByteSequencePublisherPub"]
		self.goalpublisher_proxy = mprx["GoalPublisherPub"]



	def killYourSelf(self):
		rDebug("Killing myself")
		self.kill.emit()

	# \brief Change compute period
	# @param per Period in ms
	def setPeriod(self, p):
		print("Period changed", p)
		self.Period = p
		self.timer.start(self.Period)
