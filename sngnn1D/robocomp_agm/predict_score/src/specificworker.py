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
import json
import pickle
from PySide2 import QtGui, QtCore
from genericworker import *
import math
import copy
def collate(sample):
	graphs, feats, labels = map(list, zip(*sample))
	graph = dgl.batch(graphs)
	feats = torch.from_numpy(np.concatenate(feats))
	labels = torch.from_numpy(np.concatenate(labels))
	return graph, feats, labels
import torch.nn.functional as F
import pickle
import sys
import json
import torch
from ui_mainUI import Ui_guiDlg
sys.path.append('../../models')
import gat
import gcn
import rgcn
import pg_gcn
import pg_rgcn
import pg_ggn
import pg_gat
import pg_rgcn_gat
from torch.utils.data import DataLoader
import dgl
sys.path.append('.')
from sndgAPI import *
sys.path.append('../..')
import socnav
import numpy as np
from torch_geometric.data import Data
sys.path.append('etc/qt')
sys.path.append('../etc')
from WorldGenerator import WorldGenerator

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
		self.params = pickle.load(open('../../'+'SNGNN_PARAMETERS'+'.prms', 'rb'))
		self.NNmodel = pg_rgcn_gat.PRGAT(self.params[5],
                    self.params[7],
                    5,
                    self.params[14],
                    self.params[14], #num_rels?   # TODO: Add variable
                    142,
                    int(6/2),
                    int(6/2),
                    0.001,
                    F.relu,
                    0.12,
                    bias=True
                    )
		self.NNmodel.load_state_dict(torch.load('../../'+'SNGNN_MODEL'+'.tch', map_location='cpu'))
		self.NNmodel.eval()
		self.makeJson()
		self.sngnn = SNGNN()
		return True

	@QtCore.Slot(int)
	def on_slider_valueChanged(self, value):
		value = float(len(self.labels)-1) * float(value)/float(self.ui.slider.maximum()-self.ui.slider.minimum())
		for i, label in zip(range(len(self.labels)), self.labels):
			v = 1.-math.fabs(i-value)
			if v < 0: v = 0
			label.setStyleSheet('color: rgba(0, 0, 0, {});'.format(v))

	def eventFilter(self, receiver, event):
		if event.type() is not QtCore.QEvent.Type.KeyRelease:
			return super(SpecificWorker, self).eventFilter(receiver, event)
		event.accept()
		return True

	@QtCore.Slot()
	def populateWorld(self):
		print("Populating the world")
		self.world = WorldGenerator(json.dumps(self.jsonmodel))
		self.ui.graphicsView.setScene(self.world)
		score  = self.predictBestScore()
		self.ui.slider.setValue(score)
		self.ui.graphicsView.show()
		time2  = time.time()

	@QtCore.Slot()
	def compute(self):
		#print("compute")
		self.makeJson()
		self.populateWorld()
		return True

	def check(self):
		#Sanity check if any
		pass


	def makeScenario(self,sn):
		self.jsonmodel = {}
		self.jsonmodel['identifier'] = "123231 B"
		#Adding Robot
		self.jsonmodel['robot'] = {'id':0}
		#Adding Room
		room = {}
		for i in range(int(len(self.model.nodes[1].attributes)/2)):
			room['x'+str(i)] = int(self.model.nodes[1].attributes['x'+str(i)])
			room['y'+str(i)] = int(self.model.nodes[1].attributes['y'+str(i)])
		#Adding humans and objects
		sn.add_room(room)
		for node in self.model.nodes:
			if node.nodeType == 'Human':
				human = {}
				#print(node)
				human['id'] = int(node.nodeIdentifier)-1
				human['xPos'] = float(node.attributes['xPos'])
				human['yPos'] = float(node.attributes['yPos'])
				human['orientation'] = float(node.attributes['angle'])
				sn.add_human(Human(human['id'], human['xPos'], human['yPos'], human['orientation']))
			elif node.nodeType == 'Object':
				_Object = {}
				_Object['id'] = int(node.nodeIdentifier)-1
				_Object['xPos'] = float(node.attributes['xPos'])
				_Object['yPos'] = float(node.attributes['yPos'])
				_Object['orientation'] = float(node.attributes['angle'])
				sn.add_object(Object(_Object['id'], _Object['xPos'], _Object['yPos'], _Object['orientation']))
		#Adding links
		for edge in self.model.edges:
			if edge.edgeType == 'H-H' or edge.edgeType == 'H-O':
				link = []
				link.append(int(edge.a)-1)
				link.append(int(edge.b)-1)
				sn.add_interaction(link)
		#print(self.jsonmodel)
		return sn
		pass
	def makeJson(self):
		self.jsonmodel = {}
		self.jsonmodel['identifier'] = "123231 B"
		#Adding Robot
		self.jsonmodel['robot'] = {'id':0}
		#Adding Room
		self.jsonmodel['room'] = []
		for i in range(int(len(self.model.nodes[1].attributes)/2)):
			self.jsonmodel['room'].append([int(self.model.nodes[1].attributes['x'+str(i)]),int(self.model.nodes[1].attributes['y'+str(i)])])
		#Adding humans and objects
		self.jsonmodel['humans'] = []
		self.jsonmodel['objects'] = []
		for node in self.model.nodes:
			if node.nodeType == 'Human':
				human = {}
				#print(node)
				human['id'] = int(node.nodeIdentifier)-1
				human['xPos'] = float(node.attributes['xPos'])
				human['yPos'] = float(node.attributes['yPos'])
				human['orientation'] = float(node.attributes['angle'])
				self.jsonmodel['humans'].append(human)
			elif node.nodeType == 'Object':
				Object = {}
				Object['id'] = int(node.nodeIdentifier)-1
				Object['xPos'] = float(node.attributes['xPos'])
				Object['yPos'] = float(node.attributes['yPos'])
				Object['orientation'] = float(node.attributes['angle'])
				self.jsonmodel['objects'].append(Object)
		#Adding links
		self.jsonmodel['links'] = []
		for edge in self.model.edges:
			if edge.edgeType == 'H-H':
				link = []
				link.append(int(edge.a)-1)
				link.append(int(edge.b)-1)
				link.append('interact')
				self.jsonmodel['links'].append(link)
			elif edge.edgeType == 'H-O':
				link = []
				link.append(int(edge.a)-1)
				link.append(int(edge.b)-1)
				link.append('interact')
				self.jsonmodel['links'].append(link)
		self.jsonmodel['score'] = 0
		#print(self.jsonmodel)
		pass

	def predictBestScore(self):
		print("Predicting  Best Score...")
		sn = SNScenario()
		sn = self.makeScenario(sn)
		return int(self.sngnn.predict(sn))
	#
	#
	# def predictScore(self):
	# 	print("Predicting Score...")
	# 	NNmodel = None
	# 	fws = ['dgl', 'pg']
	# 	nets = ['GCN', 'GAT' ,'GGN', 'RGCN']
	# 	for fw in fws:
	# 		for net in nets:
	# 			if NNmodel is None:
	# 				if fw == 'dgl' and (net == 'GGN' or net == 'RGAT'):
	# 					continue
	# 				params = pickle.load(open('../../'+fw+net.lower()+'.prms', 'rb'))
	# 				graph_type = params[1]
	# 			device = torch.device("cpu")
	# 			train_dataset = socnav.SocNavDataset(self.jsonmodel, mode='train', alt=graph_type)
	# 			train_dataloader = DataLoader(train_dataset, batch_size=1, collate_fn=collate)
	# 			for batch, data in enumerate(train_dataloader):
	# 				subgraph, feats, labels = data
	# 				feats = feats.to(device)
	# 				#print(subgraph)
	# 				if fw == 'dgl':
	# 					if net == 'GAT':
	# 						#print(params[4:14])
	# 						NNmodel = gat.GAT(subgraph,*params[4:14],freeze=params[15])
	# 					elif net == 'GCN':
	# 						NNmodel = gcn.GCN(subgraph,params[5],params[6],params[7],params[4],params[9],params[10])
	# 					elif net == 'RGCN':
	# 						NNmodel = rgcn.RGCN(subgraph,
	# 								in_dim=params[5],
	# 								h_dim=params[6],
	# 								out_dim=params[7],
	# 								num_rels=params[14],
	# 								feat_drop=params[10],
	# 								num_hidden_layers=params[4],
	# 								freeze=params[15])
	# 					else:
	# 						continue
	# 				else:
	# 					if net == 'GAT':
	# 					    NNmodel = pg_gat.PGAT(params[5],
	# 	                                params[7],
	# 	                                params[8][0],
	# 	                                params[10],
	# 	                                params[6],
	# 	                                params[4],
	# 	                                F.relu,
	# 	                                concat=True,
	# 	                                neg_slope=params[12],
	# 	                                bias=True)
	# 					elif net == 'GCN':
	# 						NNmodel = pg_gcn.PGCN(params[5],
	# 	                                params[7],
	# 	                                params[6],
	# 	                                params[4],
	# 	                                params[10],
	# 	                                F.relu,
	# 	                                improved=True,#Compute A-hat as A + 2I
	# 	                                bias=True)
	# 					elif net == 'RGCN':
	# 						NNmodel = pg_rgcn.PRGCN(params[5],
	# 	                                params[7],
	# 	                                params[14],
	# 	                                params[14], #num_rels?   # TODO: Add variable
	# 	                                params[6],
	# 	                                params[4],
	# 	                                params[10],
	# 	                                F.relu,
	# 	                                bias=True)
	# 					elif net == 'GGN':
	# 						NNmodel = pg_ggn.GGN(params[5],
	# 	                                params[4],
	# 	                                aggr='mean',
	# 	                                bias=True)
	# 					else:
	# 						NNmodel = pg_rgcn_gat.PRGAT(params[5],
	# 	                                params[7],
	# 	                                params[8][0],
	# 	                                params[14],
	# 	                                params[14], #num_rels?   # TODO: Add variable
	# 	                                params[6],
	# 	                                params[4],
	# 	                                params[4],
	# 	                                params[10],
	# 	                                F.relu,
	# 	                                params[12],
	# 	                                bias=True
	# 	                                )
	# 				NNmodel.load_state_dict(torch.load('../../'+fw+net.lower()+'.tch', map_location='cpu'))
	# 				NNmodel.eval()
	# 				NNmodel.g = subgraph
	# 		        # for layer in model.layers:
	# 				if fw == 'dgl':
	# 					for layer in NNmodel.layers:
	# 						layer.NNg = subgraph
	# 					logits = NNmodel(feats.float())[0].detach().numpy()[0]
	# 				else:
	# 					if net in ['GCN', 'GAT', 'GGN']:
	# 							data = Data(x=feats.float(), edge_index=torch.stack(subgraph.edges()).to(device))
	# 					else:
	# 						data = Data(x=feats.float(), edge_index=torch.stack(subgraph.edges()).to(device), edge_type=subgraph.edata['rel_type'].squeeze().to(device))
	# 					logits = NNmodel(data)[0].detach().numpy()[0]
	# 				translate = logits*100
	# 				if translate < 0: translate = 0
	# 				if translate > 100: translate = 100
	# 				print("Score : " + str(translate) + " for "+ fw + " " + net)
	# 				#self.ui.slider.setValue(int(translate))
	# 				NNmodel = None
	#
	# structuralChange
	#
	def structuralChange(self, w):
		#print(self.mutex.try_lock())
		self.model = self.agmexecutive_proxy.getModel()
		# self.makeJson()
		# self.populateWorld()
		# event = QtCore.QEvent(QtCore.QEvent.LocaleChange)
		# QtCore.QCoreApplication.sendEvent(self,event)
		#self.mutex.unlock()
		#self.populateWorld()
		#print(self.model)
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
