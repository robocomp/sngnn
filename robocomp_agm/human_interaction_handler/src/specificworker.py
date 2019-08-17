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
import copy
import random
import math
sys.path.append('../etc/')
from interaction import Interaction
from human import Human
from room import Room

# from interaction import Interaction

# If RoboComp was compiled with Python bindings you can use InnerModel in Python
# sys.path.append('/opt/robocomp/lib')
# import librobocomp_qmat
# import librobocomp_osgviewer
# import librobocomp_innermodel
MAX_GENERATION_WAIT = 1
class SpecificWorker(GenericWorker):
	def __init__(self, proxy_map):
		super(SpecificWorker, self).__init__(proxy_map)
		self.timer.timeout.connect(self.compute)
		self.Period = 10
		self.timer.start(self.Period)

	def setParams(self, params):
		try:
			self.model = self.agmexecutive_proxy.getModel()
			self.sampleNode = copy.deepcopy(self.model.nodes[0])
			self.sampleEdge = copy.deepcopy(self.model.edges[0])
			self.room = Room(self.createRoom())
		except:
			traceback.print_exc()
			print "Error reading config params"
		return True

	@QtCore.Slot()
	def compute(self):
		self.generation_time = time.time()
		if self.check():
			return True
		self.generate()
		self.addHumanAndInteractions()
		self.updateWorld()
		#print(self.model)
		return True

	def check(self):
		if len(self.model.nodes) > 2 :
			return True

	def updateWorld(self):
		self.agmexecutive_proxy.structuralChangeProposal(self.model, "humanAndInteractionHandler", "updated human and interactions")

	def addHumanAndInteractions(self):
		print("Human")
		for human in self.humans:
			print("Human Xpos : " + str(human.xPos))
			print("Human yPos : "+str(human.yPos))
			print("Human Angle : "+str(human.angle))
			print("Human ID : " + str(human.id))
			node = copy.deepcopy(self.sampleNode)
			node.attributes['xPos'] = str(human.xPos)
			node.attributes['yPos'] = str(human.yPos)
			node.attributes['angle'] = str(human.angle)
			node.nodeIdentifier = human.id
			node.nodeType = "Human"
			self.model.nodes.append(node)
			edge = copy.deepcopy(self.sampleEdge)
			edge.edgeType = "H-R"
			edge.a = 1
			edge.b = human.id
			self.model.edges.append(edge)


		print("Interactions")
		for interaction in self.interactions:
			print("Source :" + str(interaction.a.id))
			print("Destination : " + str(interaction.b.id))
			print("Human-Human")
			edge = copy.deepcopy(self.sampleEdge)
			edge.edgeType = "H-H"
			edge.a = interaction.a.id
			edge.b = interaction.b.id
			self.model.edges.append(edge)


	def createRoom(self):
		roomGraph = self.agmexecutive_proxy.getModel().nodes[1]
		#print(roomGraph.attributes)
		room = QtGui.QPolygon()
		for i in range(len(roomGraph.attributes)/2):
			room.append( QtCore.QPoint(int(roomGraph.attributes['x'+str(i)]),int(roomGraph.attributes['y'+str(i)])))
		return room

	def generateHuman(self, availableId):
		human = None
		while human is None:
			if time.time() - self.generation_time > MAX_GENERATION_WAIT:
				raise RuntimeError('MAX_GENERATION_ATTEMPTS')
			if QtCore.qrand() % 3 == 0:
				xx = int(random.normalvariate(0, 150))
				yy = int(random.normalvariate(0, 150))
			else:
				xx = QtCore.qrand()%800-400
				yy = QtCore.qrand()%800-400
				human = Human(availableId, xx, yy, (QtCore.qrand()%360)-180)

			if human is not None:
				if not self.room.containsPolygon(human.polygon()):
					human = None
		return human

	def generateComplementaryHuman(self, human, availableId):
		a = math.pi*human.angle/180.
		dist = float(QtCore.qrand()%300+50)
		human2 = None
		while human2 is None:
			if time.time() - self.generation_time > MAX_GENERATION_WAIT:
				raise RuntimeError('MAX_GENERATION_ATTEMPTS')
			xPos = human.xPos+dist*math.sin(a)
			yPos = human.yPos-dist*math.cos(a)
			human2 = Human(availableId, xPos, yPos, human.angle+180)
			if not self.room.containsPolygon(human2.polygon()):
				dist -= 5
				if dist < 20:
					human.setAngle(human.angle+180)
					a = math.pi*human.angle/180.
					dist = float(QtCore.qrand()%300+50)
				human2 = None
		return human2

	def generate(self):
		regenerateScene = True
		while regenerateScene:
			availableId = 2
			regenerateScene = False
			self.humans = []
            #self.objects = []
			self.interactions = []
            # We generate a number of humans using the absolute of a normal
            # variate with mean 1, sigma 4, capped to 15. If it's 15 we get
            # the remainder of /15
			humanCount = 1 + int(abs(random.normalvariate(1, 4))) % 10
			if humanCount == 0:
				humanCount = QtCore.qrand() % 3

			for i in range(humanCount):
				human = self.generateHuman(availableId)
				availableId += 1
                #self.addItem(human)
				self.humans.append(human)
				human2 = None
				if QtCore.qrand()%3 == 0:
					try:
						human2 = self.generateComplementaryHuman(human, availableId)
					except Exception:
						human2 = None
						pass
					print("Complementary human ")
					print(human2)
					if human2 is not None:
						availableId += 1
						interaction = Interaction(human, human2)
						self.interactions.append(copy.deepcopy(interaction))
                    	#self.addItem(interaction)
						self.humans.append(copy.deepcopy(human2))
                # elif QtCore.qrand()%2 == 0:
                #     obj = self.generateComplementaryObject(human, availableId)
                #     availableId += 1
                #     interaction = Interaction(human, obj)
                #     self.interactions.append(interaction)
                #     self.addItem(interaction)
                #     self.addItem(obj)
                #     self.objects.append(obj)

            # self.robot = Robot()
            # self.robot.setPos(0, 0)
            # self.addItem(self.robot)






	#
	# structuralChange
	#
	def structuralChange(self, w):
		self.model  = self.agmexecutive_proxy.getModel()
		self.room = Room(self.createRoom())
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
