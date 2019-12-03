# SNGNN

SNGNN: A graph neural network for social navigation conventions

__A brief description of the motivations and some results (including several videos) can be found in [https://ljmanso.com/sngnn](https://ljmanso.com/sngnn).__

[![VIDEO](https://raw.githubusercontent.com/robocomp/sngnn/master/video.png)](https://www.youtube.com/embed/QVvuywgomTE "Everything Is AWESOME")

## Introduction

The document describes how to use SNGNN, a graph neural network trained to estimate the compliance of social navigation scenarios.


## Software requirements
1. PyTorch [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
2. Dgl [https://www.dgl.ai/pages/start.html](https://www.dgl.ai/pages/start.html)
3. Rdflib (`pip install rdflib`)
4. PyTorch Geometric [https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)


## Integrating the network in your projects

Support is provided only for Python, as pytorch-geometric only supports Python.


### Python




1. Import the model and some dependencies

```python
import model_filename
from torch_geometric.data import Data
from torch.utils.data import DataLoader
import dgl
import feature_extractor_filename
import numpy as np
```


2. Load the parameters that were used to train the model
```python
params = pickle.load('SNGNN_PARAMETERS.prms', 'rb'))
```





3. Load the models state dict

```python
NNmodel = model_filename.ModelName(params)
NNmodel.load_state_dict(torch.load('SNGNN_MODEL.tch', map_location='cpu'))
NNmodel.eval()
```


4. Use a dataset loader

This is a helpful link. [https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#dataset-class](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#dataset-class)

Here socnav is the dataset class file
```python
    device = torch.device("cpu")
    train_dataset = socnav.SocNavDataset(jsonmodel, mode='train', alt=graph_type)
    train_dataloader = DataLoader(train_dataset, batch_size=1, collate_fn=collate)
```


5. Get the data and convert it to the data structure the framework needs
```python
for batch, data in enumerate(train_dataloader):
    subgraph, feats, labels = data
    feats = feats.to(device)
    data = Data(x=feats.float(), edge_index=torch.stack(subgraph.edges()).to(device), edge_type=subgraph.edata['rel_type'].squeeze().to(device))
```




6. Pass the data to the model obtain the logits. Obtain score after modifying the logits

```python
logits = self.NNmodel(data)[0].detach().numpy()[0]
score = logits*100
```




#### Python user API




```python
class SNGNN():
	def predict(self, sn_scenario):
      '''
      This function takes an instance of Scenario and returns the score
      '''
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
    	    self.interactions.append(sn_interactions)
```


### Tutorial of how to use the API:
```python
from sndgAPI import *
'''
Usage of this API
Add room using a dictionary of the x and y coordinates starting with index 0.
Add humans with instance of Human class(id,x-coordinate,y-coordinate,orientation).
Add objects with instance of Object class(id,x-coordinate,y-coordinate,orientation).
Add interactions with a list of source_index and destination_index as [src_index,dst_index].
Pass scenario to sngnn and then call predict method and you will obtain the score.
'''

sn = SNScenario()
sn.add_room({'x0':100,'y0':100,'x1':-100,'y1':100,'x2':-100,'y2':-100,'x3':100,'y3':-100})
sn.add_human(Human(1,0,0,10))
sn.add_object(Object(2,0,10,10))
sn.add_interaction([1,2])
sngnn = SNGNN()
print(sngnn.predict(sn))
```
