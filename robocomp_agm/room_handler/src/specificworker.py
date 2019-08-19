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
#from PySide import QtCore, QtGui
import copy
import random


# If RoboComp was compiled with Python bindings you can use InnerModel in Python
sys.path.append('../etc/')
from polygonmisc import translatePolygon, movePolygon

# import librobocomp_qmat
# import librobocomp_osgviewer
# import librobocomp_innermodel

# Structure of the AGM World Model  :
#
#
#
#
#
#
#

class SpecificWorker(GenericWorker):
	def __init__(self, proxy_map):
		super(SpecificWorker, self).__init__(proxy_map)
		self.timer.timeout.connect(self.compute)
		self.Period = 10000
		self.timer.start(self.Period)
		self.count = 0

	def setParams(self, params):
		try:
			self.model = copy.deepcopy(self.agmexecutive_proxy.getModel())
			self.node_dup = copy.deepcopy(self.model.nodes[0])
			#print(self.model)
			self.poly = QtGui.QPolygon()
		except:
			traceback.print_exc()
			print "Error reading config params"
		return True

	@QtCore.Slot()
	def compute(self):
		#print 'SpecificWorker.compute...'
		self.clearWorld()
		self.generateRoom()
		#self.display()
		#self.update_model = self.agmexecutive_proxy.getModel()
		self.updateWorld()
		#print(self.model)
		#print(self.agmexecutive_proxy.getModel())
		#self.agmexecutive_proxy.structuralChangeProposal(self.update_model, "s", "updated")
		#print(self.agmexecutive_proxy.getNode())
		return True

	def clearWorld(self):
		# Clear the room to contain only the robot and the room
		self.model.nodes = self.model.nodes[0:1]
		self.model.edges = self.model.edges[0:1]
		#print(self.model)


	def getRectRoom(self):
	    w = min(10, abs(random.normalvariate(1.5, 2.5))+1.5)/2 * 100
	    h = min(10, abs(random.normalvariate(3.0, 2.5))+4.0)/2 * 100
	    polygon = QtGui.QPolygon()
	    polygon.append( QtCore.QPoint(-w, -h) )
	    polygon.append( QtCore.QPoint(-w, +h) )
	    polygon.append( QtCore.QPoint(+w, +h) )
	    polygon.append( QtCore.QPoint(+w, -h) )
	    return polygon


	def getRobotPolygon(self):
		w = h = 25
		polygon = QtGui.QPolygon()
		polygon.append( QtCore.QPoint(-w, -h) )
		polygon.append( QtCore.QPoint(-w, +h) )
		polygon.append( QtCore.QPoint(+w, +h) )
		polygon.append( QtCore.QPoint(+w, -h) )
		return polygon

	def generateRandomCoordinates(self):
	  # Generate a room so that the robot will be inside the room, not colliding with the walls
  		self.robot = self.getRobotPolygon()
		self.poly = None
		while self.poly is None:
	        # Generate three randomised polygons
			p1 = translatePolygon(self.getRectRoom())
			p2 = translatePolygon(self.getRectRoom())
			p3 = translatePolygon(self.getRectRoom())
			# The room is generated as: (p1 - p2) + p3
			pRes = p1
			if random.randint(0,1) == 0:
				pRes = pRes.subtracted(p2)
			if random.randint(0,1) == 0:
				pRes = pRes.united(p3)
	        # Perform an additional check to verify that
	        # the polygon is not a degenerate one
			error_found = False
			l = pRes.toList()[:-1]
			for e in l:
				if l.count(e) > 1:
					error_found = True
	        # If the polygon has passed all the checks, go ahead
			if error_found == False:
				pRes = movePolygon(pRes)
				pRobot = pRes.united(self.robot)
				if len(pRes) == len(pRobot):
					self.poly = pRes

	def getRoomCoordinates(self):
		# TODO: Generate random coordinates
		coordinates = {}
		self.generateRandomCoordinates()
		count = 0
		for i in self.poly:
			coordinates['x'+str(count)] = str(i.x())
			coordinates['y'+str(count)] = str(i.y())
			count = count + 1

		return coordinates

	def getRobotCoordinates(self):
		coordinates = {}
		self.generateRandomCoordinates()
		count = 0
		for i in self.robot:
			coordinates['x'+str(count)] = str(i.x())
			coordinates['y'+str(count)] = str(i.y())
			count = count + 1

		return coordinates


	def generateRoom(self):
		self.model.nodes[0] = self.createNode("Robot",{}, 0)
		if len(self.model.nodes) < 2:
			#print("Printing the number of nodes")
			#print(len(self.model.nodes))
			#print("Appending")
			self.model.nodes.append(self.createNode("Room",self.getRoomCoordinates(),1))

		else:
			#print("Overwriting it now")
			self.model.nodes[1] = self.createNode("Room",self.getRoomCoordinates(),1)
		#print("Generating the room")
		#print(self.model)


	def createNode(self, nodeType, attributes, nodeIdentifier):
		updatedNode = copy.deepcopy(self.node_dup)
		updatedNode.attributes = attributes
		updatedNode.nodeType = nodeType
		updatedNode.nodeIdentifier = nodeIdentifier
		#print("Printing the World below")
		#print(	updatedNode)
		return updatedNode

	def display(self):
		for node in self.model.nodes:
			print("Node Identifier"+str(node.nodeIdentifier))
			print("Node Attributes:" + str(node.attributes))
			print("Node Type : " + str(node.nodeType))
			self.count = self.count + 1



	def updateWorld(self):
		self.agmexecutive_proxy.structuralChangeProposal(self.model, "room_handler", "updated_room")

	#
	# structuralChange
	#
	def structuralChange(self, w):
		self.model = self.agmexecutive_proxy.getModel()
		#print(self.model)
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
		print(ret)
		#
		#implementCODE
		#
		return ret


	#
	# getAgentState
	#
	def getAgentState(self):
		ret = StateStruct()
		print(ret)
		#
		#implementCODE
		#
		return ret
