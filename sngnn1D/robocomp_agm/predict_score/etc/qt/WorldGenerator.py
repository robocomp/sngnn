import random
import math
import json
import time

from PySide2 import QtCore, QtGui, QtWidgets

from human import Human
from robot import Robot
from regularobject import RegularObject
from room import Room
from interaction import Interaction

MAX_GENERATION_WAIT = 1.

class WorldGenerator(QtWidgets.QGraphicsScene):
    available_identifier = 0

    def __init__(self, data=None):
        super(WorldGenerator, self).__init__()
        self.setSceneRect(-400, -400, 800 - 2, 800 - 2)
        self.setItemIndexMethod(QtWidgets.QGraphicsScene.NoIndex)

        if data is None:
            self.generateRandomWorld()
        else:
            self.generateFromData(data)

    def generateFromData(self, raw_data):
        data = json.loads(raw_data)
        idMap = dict()
        self.clear()

        self.ds_identifier = int(data['identifier'].split()[0])

        self.room = Room(data['room'])
        self.addItem(self.room)

        self.humans = []
        for raw_human in data['humans']:
            human = Human.from_json(raw_human)
            self.addItem(human)
            self.humans.append(human)
            idMap[raw_human['id']] = human

        self.objects = []
        for raw_object in data['objects']:
            obj = RegularObject.from_json(raw_object)
            self.addItem(obj)
            self.objects.append(obj)
            idMap[raw_object['id']] = obj

        self.interactions = []
        interactions_done = []
        for interaction_raw in data['links']:
            if not [interaction_raw[1], interaction_raw[0], interaction_raw[2]] in interactions_done:
                interactions_done.append(interaction_raw)
                human = idMap[interaction_raw[0]]
                other = idMap[interaction_raw[1]]
                interaction = Interaction(human, other)
                self.interactions.append(interaction)
                self.addItem(interaction)


        self.robot = Robot()
        self.robot.setPos(0, 0)
        self.addItem(self.robot)

    def generateRandomWorld(self):
        done = False
        self.ds_identifier = WorldGenerator.available_identifier
        WorldGenerator.available_identifier += 1
        while not done:
            try:
                self.generation_time = time.time()
                self.generate()
                done = True
                # print(time.time()-self.generation_time)
            except RuntimeError:
                pass

    @staticmethod
    def distanceTo(something):
        return int(math.sqrt(something.xPos*something.xPos + something.yPos*something.yPos))

    @staticmethod
    def angleTo(something):
        angle = int(int(180.*math.atan2(something.yPos, something.xPos)/math.pi)+90.)
        if angle > 180.: angle = -360.+angle
        return angle

    def serialize(self, score=-1):
        structure = dict()
        structure['identifier'] = str(self.ds_identifier).zfill(5) + ' A'
        structure['score'] = 0
        if score > 0:
            structure['score'] = score
        structure['robot'] = {'id': 0}

        humansList = []
        for human in self.humans:
            h = dict()
            h['id'] = human.id
            h['xPos'] = +human.xPos
            h['yPos'] = +human.yPos
            h['orientation'] = +human.angle
            humansList.append(h)
        structure['humans'] = humansList

        objectsList = []
        for object in self.objects:
            o = dict()
            o['id'] = object.id
            o['xPos'] = +object.xPos
            o['yPos'] = +object.yPos
            o['orientation'] = +object.angle
            objectsList.append(o)
        structure['objects'] = objectsList

        structure['links'] = []
        for interaction in self.interactions:
            structure['links'].append( [interaction.a.id, interaction.b.id, 'interact'] )
            if type(interaction.b) is Human:
                structure['links'].append( [interaction.b.id, interaction.a.id, 'interact'] )

        structure['room'] = [ [+point.x(), point.y()] for point in self.room.poly ]

        if score >= 0:
            print(json.dumps(structure))

        return structure

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

    def generateComplementaryObject(self, human, availableId):
        a = math.pi*human.angle/180.
        dist = float(QtCore.qrand()%250+50)
        obj = None
        while obj is None:
            if time.time() - self.generation_time > MAX_GENERATION_WAIT:
                raise RuntimeError('MAX_GENERATION_ATTEMPTS')
            xPos = human.xPos+dist*math.sin(a)
            yPos = human.yPos-dist*math.cos(a)
            obj = RegularObject(availableId, xPos, yPos, (human.angle+180)%360)
            if not self.room.containsPolygon(obj.polygon()):
                dist -= 5
                if dist <= 5:
                    obj.setAngle(human.angle+180)
                    a = math.pi*human.angle/180.
                    dist = float(QtCore.qrand()%300+50)
                obj = None
        return obj

    def generate(self):
        regenerateScene = True
        while regenerateScene:

            availableId = 1
            regenerateScene = False
            self.clear()
            self.humans = []
            self.objects = []
            self.interactions = []


            self.room = Room()
            self.addItem(self.room)

            # We generate a number of humans using the absolute of a normal
            # variate with mean 1, sigma 4, capped to 15. If it's 15 we get
            # the remainder of /15
            humanCount = int(abs(random.normalvariate(1, 4))) % 15
            if humanCount == 0:
                humanCount = QtCore.qrand() % 3

            for i in range(humanCount):
                human = self.generateHuman(availableId)
                availableId += 1
                self.addItem(human)
                self.humans.append(human)
                if QtCore.qrand()%3 == 0:
                    human2 = self.generateComplementaryHuman(human, availableId)
                    availableId += 1
                    self.addItem(human2)
                    interaction = Interaction(human, human2)
                    self.interactions.append(interaction)
                    self.addItem(interaction)
                    self.humans.append(human2)
                elif QtCore.qrand()%2 == 0:
                    obj = self.generateComplementaryObject(human, availableId)
                    availableId += 1
                    interaction = Interaction(human, obj)
                    self.interactions.append(interaction)
                    self.addItem(interaction)
                    self.addItem(obj)
                    self.objects.append(obj)

            self.robot = Robot()
            self.robot.setPos(0, 0)
            self.addItem(self.robot)



        self.text = 'Humans:' + str(len(self.humans)) + ' ' + 'Objects:' + str(len(self.objects))



