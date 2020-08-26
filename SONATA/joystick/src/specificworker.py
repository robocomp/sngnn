#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
#    Copyright (C) 2020 by YOUR NAME HERE
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
import sys
import signal
import os, sys, threading, numpy
import time
import contextlib
with contextlib.redirect_stdout(None):
    import pygame


from genericworker import *


class SpecificWorker(GenericWorker):
    def __init__(self, proxy_map, control_type, MouseControl):
        super(SpecificWorker, self).__init__(proxy_map)
        self.timer.timeout.connect(self.compute)
        self.Period = 1
        self.timer.start(self.Period)
        self.type = control_type
        self.MouseControl = MouseControl
        self.last_joystick_data_sent = None
        self.WINDOW_HEIGHT = 400
        self.WINDOW_WIDTH = 400
        pygame.init()
        joystick_count = pygame.joystick.get_count()
        print(f'joystick count:{joystick_count}')

        if joystick_count == 0:
            self.type = 'mouse'

        if self.type == 'joystick':
            print('joystick!')
            pygame.joystick.init()
            pygame.joystick.Joystick(0).init()

        else:
            print('mouse!')
            pygame.display.init()
            display = pygame.display.set_mode((self.WINDOW_WIDTH,self.WINDOW_HEIGHT))
            surface = pygame.display.get_surface()
            pygame.display.set_caption('Mouse controller') 
            font = pygame.font.SysFont("ubuntu", 20)
            pygame.draw.line(surface,(255,0,0),(self.WINDOW_WIDTH/2,0),(self.WINDOW_WIDTH/2,self.WINDOW_HEIGHT))
            pygame.draw.line(surface,(255,0,0),(0,self.WINDOW_HEIGHT/2),(self.WINDOW_WIDTH,self.WINDOW_HEIGHT/2))
            text = font.render("Hold the button down and move the mouse cursor", True, (255, 255, 255))
            _, text_height = text.get_size()
            textRect = text.get_rect()
            display.blit(text, textRect) 
            # text = font.render("Press barspace to stop the robot", True, (255, 255, 255))
            # display.blit(text, (0, text_height+10)) 
            # text = font.render("Close this window to finish the test", True, (255, 255, 255))
            # display.blit(text, (0, 2*text_height+20)) 
            pygame.display.update()

        self.prev_adv = self.prev_rot = 0.

    def __del__(self):
        print('SpecificWorker destructor')

    def setParams(self, params):
        return True

    @QtCore.Slot()
    def compute(self):

        if self.type == 'joystick':
            while True:
                time.sleep(0.1)
                pygame.event.pump()
                
                joystick_data = RoboCompJoystickAdapter.TData()
                joystick_data.id = 'joystick0'
                joystick_data.buttons = []
                joystick_data.axes = [ RoboCompJoystickAdapter.AxisParams(str(axis), pygame.joystick.Joystick(0).get_axis(axis)) for axis in range(2)]

                if self.last_joystick_data_sent != joystick_data:
                    if self.last_joystick_data_sent is not None:
                        self.joystickadapter_proxy.sendData(joystick_data)
                    self.last_joystick_data_sent = joystick_data

        else:
            flag = True
            prev_x = 0.
            prev_y = 0.
            move_robot = False
            self.axis = [0, 0]
            self.button_pressed = False
            while True:
                if pygame.mouse.get_focused() and flag:
                    flag = False
                    pygame.mouse.set_pos([self.WINDOW_WIDTH/2,self.WINDOW_HEIGHT/2])
                elif pygame.mouse.get_focused() == 0:
                    flag = True
                    time.sleep(0.5)

                time.sleep(0.05)

                events = pygame.event.get()

                for event in events:
                    if self.MouseControl == "True":
                        if event.type == pygame.MOUSEBUTTONDOWN:
#                            b1,b2,b3 = pygame.mouse.get_pressed()
                            self.button_pressed = True
                        if event.type == pygame.MOUSEBUTTONUP:
                            self.button_pressed = False



                if self.button_pressed:
                    x,y = pygame.mouse.get_pos()
                    if prev_x == x and prev_y == y:
                        # print("Continuing as the values are same")
                        continue
                    else:
                        self.axis[0] = (x-self.WINDOW_WIDTH/2)/(self.WINDOW_WIDTH/2)
                        self.axis[1] = (y-self.WINDOW_WIDTH/2)/(self.WINDOW_WIDTH/2)
                        prev_x = x
                        prev_y = y
                        move_robot = True
                else:
                    if self.axis[0]!=0 or self.axis[1]!=0:
                        self.axis = [0, 0]
                        prev_x = 0
                        prev_y = 0
                        move_robot = True
                    else:
                        move_robot = False


                    if event.type == pygame.QUIT : 
                        pygame.quit() 
                        os._exit(0)


                if move_robot:
                    move_robot = False
                    try:
                        joystick_data = RoboCompJoystickAdapter.TData()
                        joystick_data.id = 'Mouse'
                        joystick_data.buttons = []
                        joystick_data.axes = [RoboCompJoystickAdapter.AxisParams(str(axis), self.axis[axis]) for axis in range(2)]
                        self.joystickadapter_proxy.sendData(joystick_data)

                    except Exception as e:
                        print(e)
        
        return True

