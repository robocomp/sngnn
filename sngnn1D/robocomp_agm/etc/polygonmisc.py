import math
import numpy as np
import random

from PySide import QtCore

def movePolygon(p):
    return translatePolygon(rotatePolygon(p))

def rotatePolygon(p, theta=None):
    if theta is None:
        theta = random.uniform(-math.pi, math.pi)
    c, s = np.cos(theta), np.sin(theta)
    m = np.array([[c,-s], [s, c]])

    for i in range(len(p)):
        pp = np.array([[p[i].x()], [p[i].y()]])
        ppp = np.dot (m, pp)
        p[i] = QtCore.QPoint(ppp[0], ppp[1])
    return p

def translatePolygon(p, tx=None, ty=None):
    if tx is None:
        tx = random.uniform(-150, +150)
    if ty is None:
        ty = random.uniform(-250, +150)
    for i in range(len(p)):
        pp = np.array([[p[i].x()], [p[i].y()]])
        p[i] = QtCore.QPoint(pp[0]+tx, pp[1]+ty)
    return p
