#
# Copyright (C) 2019 by YOUR NAME HERE
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

import sys, os, Ice

ROBOCOMP = ''
try:
	ROBOCOMP = os.environ['ROBOCOMP']
except:
	print '$ROBOCOMP environment variable not set, using the default value /opt/robocomp'
	ROBOCOMP = '/opt/robocomp'
if len(ROBOCOMP)<1:
	print 'ROBOCOMP environment variable not set! Exiting.'
	sys.exit()

additionalPathStr = ''
icePaths = []
try:
	icePaths.append('/opt/robocomp/interfaces')
	SLICE_PATH = os.environ['SLICE_PATH'].split(':')
	for p in SLICE_PATH:
		icePaths.append(p)
		additionalPathStr += ' -I' + p + ' '
except:
	print 'SLICE_PATH environment variable was not exported. Using only the default paths'
	pass

ice_AGMExecutive = False
for p in icePaths:
	print 'Trying', p, 'to load AGMExecutive.ice'
	if os.path.isfile(p+'/AGMExecutive.ice'):
		print 'Using', p, 'to load AGMExecutive.ice'
		preStr = "-I/opt/robocomp/interfaces/ -I"+ROBOCOMP+"/interfaces/ " + additionalPathStr + " --all "+p+'/'
		wholeStr = preStr+"AGMExecutive.ice"
		Ice.loadSlice(wholeStr)
		ice_AGMExecutive = True
		break
if not ice_AGMExecutive:
	print 'Couldn\'t load AGMExecutive'
	sys.exit(-1)
from RoboCompAGMExecutive import *
ice_AGMCommonBehavior = False
for p in icePaths:
	print 'Trying', p, 'to load AGMCommonBehavior.ice'
	if os.path.isfile(p+'/AGMCommonBehavior.ice'):
		print 'Using', p, 'to load AGMCommonBehavior.ice'
		preStr = "-I/opt/robocomp/interfaces/ -I"+ROBOCOMP+"/interfaces/ " + additionalPathStr + " --all "+p+'/'
		wholeStr = preStr+"AGMCommonBehavior.ice"
		Ice.loadSlice(wholeStr)
		ice_AGMCommonBehavior = True
		break
if not ice_AGMCommonBehavior:
	print 'Couldn\'t load AGMCommonBehavior'
	sys.exit(-1)
from RoboCompAGMCommonBehavior import *
ice_AGMWorldModel = False
for p in icePaths:
	print 'Trying', p, 'to load AGMWorldModel.ice'
	if os.path.isfile(p+'/AGMWorldModel.ice'):
		print 'Using', p, 'to load AGMWorldModel.ice'
		preStr = "-I/opt/robocomp/interfaces/ -I"+ROBOCOMP+"/interfaces/ " + additionalPathStr + " --all "+p+'/'
		wholeStr = preStr+"AGMWorldModel.ice"
		Ice.loadSlice(wholeStr)
		ice_AGMWorldModel = True
		break
if not ice_AGMWorldModel:
	print 'Couldn\'t load AGMWorldModel'
	sys.exit(-1)
from RoboCompAGMWorldModel import *

class AGMExecutiveTopicI(AGMExecutiveTopic):
	def __init__(self, worker):
		self.worker = worker

	def structuralChange(self, w, c):
		return self.worker.structuralChange(w)
	def edgesUpdated(self, modifications, c):
		return self.worker.edgesUpdated(modifications)
	def edgeUpdated(self, modification, c):
		return self.worker.edgeUpdated(modification)
	def symbolUpdated(self, modification, c):
		return self.worker.symbolUpdated(modification)
	def symbolsUpdated(self, modifications, c):
		return self.worker.symbolsUpdated(modifications)
