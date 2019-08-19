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

import sys, os, traceback, time

from PySide import QtGui, QtCore
from genericworker import *

# If RoboComp was compiled with Python bindings you can use InnerModel in Python
# sys.path.append('/opt/robocomp/lib')
# import librobocomp_qmat
# import librobocomp_osgviewer
# import librobocomp_innermodel

class SpecificWorker(GenericWorker):
	def __init__(self, proxy_map):
		super(SpecificWorker, self).__init__(proxy_map)
		self.timer.timeout.connect(self.compute)
		self.Period = 100
		self.timer.start(self.Period)

	def setParams(self, params):
		self.model = self.agmexecutive_proxy.getModel()
		return True

	@QtCore.Slot()
	def compute(self):
		print("Starting transportation")
		move_speed = 2
		self.move(move_speed)
		print("Reducing y coordinates")
		self.updateWorld()
		return True


	def updateWorld(self):
		self.agmexecutive_proxy.structuralChangeProposal(self.model, "room_handler", "updated_room")

	def move(self,move_speed):
		for node in self.model.nodes:
			if node.nodeType == 'Room':
				for i in range(int(len(self.model.nodes[1].attributes)/2)):
					self.model.nodes[1].attributes['y'+str(i)] = str(int(self.model.nodes[1].attributes['y'+str(i)])+move_speed)
			elif node.nodeType not in ['Robot','robot']:
				node.attributes['yPos'] = str(float(node.attributes['yPos'])+move_speed)

	#
	# structuralChange
	#
	def structuralChange(self, w):
		self.model = self.agmexecutive_proxy.getModel()
		#
		#subscribesToCODE
		#
		pass


	#
	# edgesUpdated
	#
	def edgesUpdated(self, modifications):
		#
		#subscribesToCODE
		#
		pass


	#
	# edgeUpdated
	#
	def edgeUpdated(self, modification):
		#
		#subscribesToCODE
		#
		pass


	#
	# symbolUpdated
	#
	def symbolUpdated(self, modification):
		#
		#subscribesToCODE
		#
		pass


	#
	# symbolsUpdated
	#
	def symbolsUpdated(self, modifications):
		#
		#subscribesToCODE
		#
		pass


	#
	# reloadConfigAgent
	#
	def reloadConfigAgent(self):
		ret = bool()
		#
		#implementCODE
		#
		return ret


	#
	# activateAgent
	#
	def activateAgent(self, prs):
		ret = bool()
		#
		#implementCODE
		#
		return ret


	#
	# setAgentParameters
	#
	def setAgentParameters(self, prs):
		ret = bool()
		#
		#implementCODE
		#
		return ret


	#
	# getAgentParameters
	#
	def getAgentParameters(self):
		ret = ParameterMap()
		#
		#implementCODE
		#
		return ret


	#
	# killAgent
	#
	def killAgent(self):
		#
		#implementCODE
		#
		pass


	#
	# uptimeAgent
	#
	def uptimeAgent(self):
		ret = int()
		#
		#implementCODE
		#
		return ret


	#
	# deactivateAgent
	#
	def deactivateAgent(self):
		ret = bool()
		#
		#implementCODE
		#
		return ret


	#
	# getAgentState
	#
	def getAgentState(self):
		ret = StateStruct()
		#
		#implementCODE
		#
		return ret
