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
import random
import copy
import math
sys.path.append('../etc/')
from interaction import Interaction
from human import Human
from room import Room
from regularobject import RegularObject

# If RoboComp was compiled with Python bindings you can use InnerModel in Python
# sys.path.append('/opt/robocomp/lib')
# import librobocomp_qmat
# import librobocomp_osgviewer
# import librobocomp_innermodel

class SpecificWorker(GenericWorker):

	def __init__(self, proxy_map):
		super(SpecificWorker, self).__init__(proxy_map)
		self.timer.timeout.connect(self.compute)
		self.Period = 6000
		self.timer.start(self.Period)
		self.MAX_GENERATION_WAIT = 1
	def setParams(self, params):
		try:
			self.model = self.agmexecutive_proxy.getModel()
			self.sampleNode = copy.deepcopy(self.model.nodes[0])
			self.sampleEdge = copy.deepcopy(self.model.edges[0])
		except:
			traceback.print_exc()
			print "Error reading config params"
		return True

	@QtCore.Slot()
	def compute(self):
		#print 'SpecificWorker.compute...'
		if self.check():
			return True
		self.generation_time = time.time()
		self.generateObject()
		self.addObjectToWorld()
		self.updateWorld()
		return True

	def check(self):
		if len(self.model.nodes) < 3:
			return True

	def addObjectToWorld(self):
		for object in self.objects:
			# print("Object Xpos : " + str(object.xPos))
			# print("Object yPos : "+str(object.yPos))
			# print("Object Angle : "+str(object.angle))
			# print("Object ID : " + str(object.id))
			node = copy.deepcopy(self.sampleNode)
			node.attributes['xPos'] = str(object.xPos)
			node.attributes['yPos'] = str(object.yPos)
			node.attributes['angle'] = str(object.angle)
			node.nodeIdentifier = object.id
			node.nodeType = "Object"
			self.model.nodes.append(node)
			edge = copy.deepcopy(self.sampleEdge)
			edge.edgeType = "O-R"
			edge.a = 1
			edge.b = object.id
			self.model.edges.append(edge)
		#print("Interactions")
		for interaction in self.interactions:
			# print("Source :" + str(interaction.a.id))
			# print("Destination : " + str(interaction.b.id))
			# print("Human-Object")
			edge = copy.deepcopy(self.sampleEdge)
			edge.edgeType = "H-O"
			edge.a = interaction.a.id
			edge.b = interaction.b.id
			self.model.edges.append(edge)



	def updateWorld(self):
		self.agmexecutive_proxy.structuralChangeProposal(self.model, "ObjectHandler", "updated objects and interaction")

	def getHuman(self):
		self.humans = []
		count = 0
		#print("Start appending")
		for node in self.model.nodes:
			if node.nodeType == 'Human':
				count = count + 1
				human = Human(node.nodeIdentifier, float(node.attributes['xPos']), float(node.attributes['yPos']), float(node.attributes['angle']))
				self.humans.append(human)
		#print("End Appending")
		print("Added "+ str(count) + "Humans")

	def generateComplementaryObject(self, human, availableId):
		a = math.pi*human.angle/180.
		dist = float(QtCore.qrand()%250+50)
		obj = None
		while obj is None:
			if time.time() - self.generation_time > self.MAX_GENERATION_WAIT:
				print("Time")
				raise RuntimeError('MAX_GENERATION_ATTEMPTS')
			xPos = human.xPos+dist*math.sin(a)
			yPos = human.yPos-dist*math.cos(a)
			obj = RegularObject(availableId, xPos, yPos, (human.angle+180)%360)
			#print(obj)
			if not self.room.containsPolygon(obj.polygon()):
				dist -= 5
				if dist <= 5:
					obj.setAngle(human.angle+180)
					a = math.pi*human.angle/180.
					dist = float(QtCore.qrand()%300+50)
				obj = None
		return obj

	def generateObject(self):
		self.interactions = []
		self.objects = []
		obj = None
		for human in self.humans:
			if human is not None:
				if QtCore.qrand()%3 == 0: #The constant can be updated acoording to need for more or less number of objects in the world
					try:
						obj = self.generateComplementaryObject(human, self.availableId)
					except RuntimeError:
						#Handling the time generation error
						obj = None
						pass
						print("Took more time to create a random object")
						#Returning so that too much time is not wasted in creating a random object which is inside the room
						return
					if obj is not None:
						self.availableId += 1
						print("Object")
						print(obj)
						try:
							interaction = Interaction(human, obj)
							self.interactions.append(copy.deepcopy(interaction))
						except Exception:
							print(Exception)
							pass
						if obj not in self.objects:
							self.objects.append(copy.deepcopy(obj))
						else:
							print("Repeated object which will fail")
	def createRoom(self):
		roomGraph = self.model.nodes[1]
		room = QtGui.QPolygon()
		for i in range(len(roomGraph.attributes)/2):
			room.append( QtCore.QPoint(int(roomGraph.attributes['x'+str(i)]),int(roomGraph.attributes['y'+str(i)])))
		return room
	#
	# structuralChange
	#
	def structuralChange(self, w):
		#Update all the dependent variables
		#Update the current world
		self.model  = self.agmexecutive_proxy.getModel()
		#Update the room whenever a change is made
		self.room = Room(self.createRoom())
		#Update the availableId variable to the number of nodes currently in the model as id starts from 0
		self.availableId = len(self.model.nodes)
		#Update the humans in the updated model
		self.getHuman()

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
