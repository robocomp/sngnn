'''sngnn
Copyright (C) 2019  RoboComp

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

from torch_geometric.data import Data
from torch.utils.data import DataLoader
import dgl
import torch
import numpy as np
import sys, os
import socnav
sys.path.append(os.path.join(os.path.dirname(__file__),'models'))
import pg_rgcn_gat
import pickle
import torch.nn.functional as F

def collate(sample):
    graphs, feats, labels = map(list, zip(*sample))
    graph = dgl.batch(graphs)
    feats = torch.from_numpy(np.concatenate(feats))
    labels = torch.from_numpy(np.concatenate(labels))
    return graph, feats, labels


class SNGNN():
    def __init__(self, device_='cpu'):
        self.device = torch.device(device_)
        self.params = pickle.load(open('SNGNN_PARAMETERS.prms', 'rb'), fix_imports=True)
        self.GNNmodel = pg_rgcn_gat.PRGAT(self.params[5],
                    self.params[7],
                    self.params[8][0],
                    self.params[14],
                    self.params[14],
                    self.params[6],
                    int(self.params[4]/2),
                    int(self.params[4]/2),
                    self.params[10],
                    self.params[9],
                    self.params[12],
                    bias=True
                    )
        self.GNNmodel.load_state_dict(torch.load('SNGNN_MODEL.tch', map_location=device_))
        self.GNNmodel.eval()

    def makeJson(self,sn_scenario):
        #Make Json from sn_scenario
        self.jsonmodel = {}
        self.jsonmodel['identifier'] = "000000 A"
		#Adding Robot
        self.jsonmodel['robot'] = {'id':0}
		#Adding Room
        self.jsonmodel['room'] = []
        for i in range(int(len(sn_scenario.room.keys())/2)):
            self.jsonmodel['room'].append([sn_scenario.room['x'+str(i)],sn_scenario.room['y'+str(i)]])
		#Adding humans and objects
        self.jsonmodel['humans'] = []
        self.jsonmodel['objects'] = []
        for _human in sn_scenario.humans:
            human = {}
			#print(node)
            human['id'] = int(_human.id)
            human['xPos'] = float(_human.xPos)
            human['yPos'] = float(_human.yPos)
            human['orientation'] = float(_human.angle)
            self.jsonmodel['humans'].append(human)
        for object in sn_scenario.objects:
            Object = {}
            Object['id'] = int(object.id)
            Object['xPos'] = float(object.xPos)
            Object['yPos'] = float(object.yPos)
            Object['orientation'] = float(object.angle)
            self.jsonmodel['objects'].append(Object)
	    #Adding links
        self.jsonmodel['links'] = []
        for interaction in sn_scenario.interactions:
            link = []
            link.append(int(interaction[0]))
            link.append(int(interaction[1]))
            link.append('interact')
            self.jsonmodel['links'].append(link)

        self.jsonmodel['score'] = 0
        return self.jsonmodel

    def predict(self, sn_scenario):
        jsonmodel = self.makeJson(sn_scenario)
        graph_type = 'relational'
        train_dataset = socnav.SocNavDataset(jsonmodel, mode='train', alt=graph_type)
        train_dataloader = DataLoader(train_dataset, batch_size=1, collate_fn=collate)
        for batch, data in enumerate(train_dataloader):
            subgraph, feats, labels = data
            feats = feats.to(self.device)
            data = Data(x=feats.float(),edge_index=torch.stack(subgraph.edges()).to(self.device),edge_type=subgraph.edata['rel_type'].squeeze().to(self.device))
            logits = self.GNNmodel(data)[0].detach().numpy()[0]
            score = logits*100
            if score > 100:
                score = 100
            elif score < 0:
                score = 0
        return score

class Human():
    def __init__(self, id, xPos, yPos, angle):
        self.id = id
        self.xPos = xPos
        self.yPos = yPos
        self.angle = angle

class Object():
    def __init__(self, id, xPos, yPos, angle):
        self.id = id
        self.xPos = xPos
        self.yPos = yPos
        self.angle = angle


class SNScenario():
    def __init__(self):
        self.room = None
        self.humans = []
        self.objects = []
        self.interactions = []

    def add_room(self, sn_room):
        self.room  = sn_room

    def add_human(self, sn_human):
        self.humans.append(sn_human)

    def add_object(self, sn_object):
        self.objects.append(sn_object)

    def add_interaction(self, sn_interactions):
        h_l = [human.id for human in self.humans]
        o_l = [object.id for object in self.objects]
        s = sn_interactions[0]
        d = sn_interactions[1]
        if (s in h_l and d in o_l) or (s in h_l and d in h_l):
            self.interactions.append(sn_interactions)
        else:
            raise ValueError('Invalid Interaction. Allowed Interactions: Human-Human and Human-Object')
