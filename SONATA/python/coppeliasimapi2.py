#!/usr/bin/env python3
import os
import sys
import numpy as np
from math import cos, sin, atan2
from os import path
import time
from typing import List, Tuple, Sequence
import threading 

from pyrep import PyRep
from pyrep.objects.shape import Shape
from pyrep.objects.object import Object
from pyrep.const import PrimitiveShape
# from pyrep.robots.mobiles.youbot import YouBot
from youbot import YouBot

from shapely.geometry import Polygon

#
# Wall
#  
class Wall(object):
    def __init__(self, p1: Sequence[float], p2: Sequence[float]):
        super(Wall, self).__init__()
        self.p1, self.p2 = p1, p2
        # pre
        x, y = 0.5*(p1[0] + p2[0]), 0.5*(p1[1] + p2[1])
        angle = atan2(p2[1]-p1[1], p2[0]-p1[0])
        self.length = np.linalg.norm(np.array(p2)-np.array(p1))
        ss = Shape.create(type=PrimitiveShape.CUBOID, 
                      color=[1,1,1], size=[self.length, 0.1, 1.],
                      position=[x, y, 0.5])
        ss.set_color([1,1,1])
        ss.set_position([x, y, 0.5])
        ss.set_dynamic(False)
        ss.rotate([0., 0., angle])
        self.handle = ss
        self.handle.set_model(True)

    def get_length(self):
        return self.length

    def get_position(self, relative_to=None):
        return self.handle.get_position(relative_to=relative_to)

    def get_orientation(self, relative_to=None):
        return self.handle.get_orientation(relative_to=relative_to)

    def get_handle(self):
        return self.handle._handle

    def remove(self):
        self.handle.remove()

    def check_collision(self, obj):
        return self.handle.check_collision(obj)

    def get_model_bounding_box(self):
        return self.handle.get_model_bounding_box()



class Goal(object):
    def __init__(self, x, y):
        super(Goal, self).__init__()
        ss1 = Shape.create(type=PrimitiveShape.CONE, 
                              color=[1,0,0], size=[0.4, 0.4, 0.75],
                              position=[x, y, 2.5],orientation=[3.14,0,3.14])
        ss1.set_color([1, 0, 0])
        ss1.set_position([x, y, 1.5])
        ss1.set_orientation([3.14,0,3.14])        
        ss1.set_dynamic(False)

        self.handle_add = ss1

        ss2 = Shape.create(type=PrimitiveShape.CONE, 
                              color=[0,1,0], size=[0.75, 0.75, 0.0015],
                              position=[x, y, 0.000],orientation=[3.14,0,3.14])
        ss2.set_color([0, 1, 0])
        ss2.set_position([x, y, 0.000])
        ss2.set_orientation([3.14,0,3.14])        
        ss2.set_dynamic(False)
        self.handle = ss2
        self.handle.set_model(True)

    def get_position(self, relative_to=None):
        return self.handle.get_position(relative_to=relative_to)

    def get_orientation(self, relative_to=None):
        return self.handle.get_orientation(relative_to=relative_to)

    def get_handle(self):
        return self.handle._handle

    def remove(self):
        self.handle.remove()
        self.handle_add.remove()

    def check_collision(self, obj):
        return self.handle.check_collision(obj)

    def get_model_bounding_box(self):
        return self.handle.get_model_bounding_box()


class RelationObject(object):
    def __init__(self,x,y,z,pitch,yaw,roll,length):
        super(RelationObject,self).__init__()
        ss = Shape.create(type=PrimitiveShape.CYLINDER, 
                              color=[1,0,0], size=[0.01, 0.02, length],
                              position=[x, y, 2.5],orientation=[3.14,0,3.14])

        ss.set_color([0, 0, 1])
        ss.set_position([x, y, 0.0])
        ss.set_orientation([pitch,yaw,roll])  
        #ss.set_orientation([1.57,1.57,1.57])        
        ss.set_dynamic(False)
        self.handle = ss

    def get_position(self, relative_to=None):
        return self.handle.get_position(relative_to=relative_to)

    def get_orientation(self, relative_to=None):
        return self.handle.get_orientation(relative_to=relative_to)

    def get_handle(self):
        return self.handle._handle

    def remove(self):
        self.handle.remove()

    def check_collision(self, obj):
        return self.handle.check_collision(obj)

    def get_model_bounding_box(self):
        return self.handle.get_model_bounding_box()

    def move(self, pose_x,pose_y,pose_z,pitch,yaw,roll,length):
        self.handle.set_position([pose_x,pose_y,pose_z])
        self.handle.set_orientation([pitch,yaw,roll])




class Human(object):
    def __init__(self, handle: Object):
        super(Human, self).__init__()
        self.handle = handle
        handle.value = 0
        children = handle.get_objects_in_tree(handle)
        for child in children:
            name = child.get_name() 
            if 'Bill_goalPosCylinder' in name:
                self.dummy_handle = child.get_parent()
                # self.dummy_handle._set_property(prop_type: int, value: bool) -> None:

    def set_position(self, position, relative_to=None):
        self.handle.set_position(position, relative_to)
        self.move(position, relative_to)

    def set_orientation(self, position, relative_to=None):
        self.handle.set_orientation(position, relative_to)

    def move(self, position, relative_to=None ):
        self.dummy_handle.set_position(position, relative_to)

    def get_position(self, relative_to=None):
        return self.handle.get_position(relative_to=relative_to)

    def get_orientation(self, relative_to=None):
        return self.handle.get_orientation(relative_to=relative_to)

    def get_handle(self):
        return self.handle._handle

    def remove(self):
        self.dummy_handle.remove()
        self.handle.remove()

    def check_collision(self, obj):
        if type(obj)==type(self):
            return self.handle.check_collision(obj.handle)
        else:
            return self.handle.check_collision(obj)

    def get_model_bounding_box(self):
        return self.dummy_handle.get_model_bounding_box()

class HumanOnPath(object):
    def __init__(self, handle: Object):
        super(HumanOnPath, self).__init__()
        self.handle = handle
        handle.value = 0
        children = self.handle.get_objects_in_tree()
        for child in children:
            name = child.get_name() 
            if 'Bill_base' in name:
                self.human_handle = child

    def set_position(self, position):
        self.handle.set_position(position)

    def set_orientation(self, position):
        self.handle.set_orientation(position)

    def get_position(self, relative_to=None):
        return self.human_handle.get_position(relative_to=relative_to)

    def get_orientation(self, relative_to=None):
        return self.human_handle.get_orientation(relative_to=relative_to)

# CoppeliaSimAPI
class CoppeliaSimAPI(PyRep):
    def __init__(self, paths: Sequence[str]=[]):
        super(CoppeliaSimAPI, self).__init__()
        self.coppelia_paths = paths + ['./', os.environ['COPPELIASIM_ROOT']+'/']

    def load_scene(self, scene_path: str, headless: bool=False):
        for source in self.coppelia_paths:
            full_path = source + '/' + scene_path
            if path.exists(full_path):
                return self.launch(os.path.abspath(full_path), headless)

    def close(self):
        self.shutdown()

    def create_wall(self, p1: Sequence[float], p2: Sequence[float]):
        return Wall(p1, p2)

    def create_goal(self, p1: Sequence[float], p2: Sequence[float]):
        return Goal(p1, p2)

    def create_relation(self, p1: Sequence[float], p2: Sequence[float], p3: Sequence[float], pitch: Sequence[float], yaw: Sequence[float], roll: Sequence[float], length: Sequence[float]):
        return RelationObject(p1, p2, p3, pitch, yaw, roll, length)


    def get_object(self, name: str):
        return Object.get_object(name)


    def set_object_parent(self, obj, parent, keep_in_place=True):
        obj = self.convert_to_valid_handle(obj)
        parent = self.convert_to_valid_handle(parent)
        code = f'sim.setObjectParent({obj}, {parent}, {keep_in_place})'
        ret = self.run_script(code)
        return ret

    def create_human(self):
        model = 'models/people/path planning Bill.ttm'
        human_handle = self.load_model(model)
        return Human(human_handle)

    def create_human2(self):
        model = 'Bill_on_simple_path.ttm'
        human_handle = self.load_model(model)
        return HumanOnPath(human_handle)


    def load_model(self, model):
        for source in self.coppelia_paths:
            full_path = source + '/' + model
            if path.exists(full_path):
                ret = self.import_model(os.path.abspath(full_path))
        return ret

    def remove_objects(self, humans_list,tables_list,laptops_list,plants_list,goal,walls_list,relations_list,moving_relations_list):
        for i in range(len(walls_list)):
            walls_list[i].remove()
        for i in range(len(humans_list)):
            humans_list[i].remove()
        for i in range(len(tables_list)):
            tables_list[i].remove()
        for i in range(len(laptops_list)):
            laptops_list[i].remove()
        for i in range(len(plants_list)):
            plants_list[i].remove()

        if goal is not None:
            goal.remove()

        for relation in relations_list:
            relation.remove()

        for relation in moving_relations_list:
            relation.remove()

        # robot.remove()


    def remove_object(self, object_):
        object_.remove()



    # NOT INCLUDED IN THE DOCUMENTATION YET
    def get_youbot(self) -> YouBot:
        children = self.get_objects_children('sim.handle_scene', children_type='sim.object_shape_type', filter_children=1+2)
        for h in children:
            name = self.get_object_name(h)
            if name == 'youBot':
                return YouBot(self, h)

    def create_youbot(self, x: float, y: float, z: float) -> YouBot:
        ix, iy, iz = YouBot.get_position_offsets()
        ret = self.create_model('models/robots/mobile/KUKA YouBot.ttm', x+ix, y+iy, z+iz, 0.)
        self.set_object_orientation(ret, *YouBot.get_orientation_offsets())
        return YouBot(self, ret)


    def set_joint_target_velocity(self, handle, target, asynch=False):
        call = self.get_call_object(asynch)
        return self.client.simxSetJointTargetVelocity(handle, target, call.get())

    def pause(self):
        call = self.get_call_object(asynch)
        self.client.simxPauseSimulation(call.get())

    def check_collision(self, obj1, obj2, asynch=False):
        poly1 = self.getobject_polygon(obj1)
        poly2 = self.getobject_polygon(obj2)
        return poly1.intersects(poly2)

    def getobject_polygon(self, obj):
        bb = obj.get_model_bounding_box()
        pos = obj.get_position()
        poly = []
        poly.append((bb[0]+pos[0], bb[2]+pos[1]))
        poly.append((bb[0]+pos[0], bb[3]+pos[1]))
        poly.append((bb[1]+pos[0], bb[3]+pos[1]))        
        poly.append((bb[1]+pos[0], bb[2]+pos[1]))
        return Polygon(poly)
         



    def set_collidable(self, obj, asynch=False):
        handle = self.convert_to_valid_handle(obj)
        return self.run_script(f'sim.setObjectSpecialProperty({handle},sim.objectspecialproperty_collidable+'
                               f'sim.objectspecialproperty_measurable+sim.objectspecialproperty_detectable_all'
                               f'+sim.objectspecialproperty_renderable)', asynch)

    @staticmethod
    def get_transform_matrix(x: float, y: float, z: float, angle: float):
        rotate_matrix = np.matrix([[cos(angle), -sin(angle), 0., 0.],
                                [sin(angle),  cos(angle), 0., 0.],
                                [        0.,          0., 1., 0.],
                                [        0.,          0., 0., 1.]])
        translate_matrix = np.matrix([[ 1., 0., 0., x ],
                                    [ 0., 1., 0., y ],
                                    [ 0., 0., 1., z ],
                                    [ 0., 0., 0., 1.]])
        return (translate_matrix @ rotate_matrix).flatten().tolist()[0]

    @staticmethod
    def get_transformation_matrix(x: float, y: float, angle: float):
        M = np.zeros( (3,3) )
        M[0][0], M[0][1], M[0][2] = +cos(angle), -sin(angle), x
        M[1][0], M[1][1], M[1][2] = +sin(angle), +cos(angle), y
        M[2][0], M[2][1], M[2][2] =          0.,          0., 1.
        return M

