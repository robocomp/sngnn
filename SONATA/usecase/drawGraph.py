import sys
import json
from PySide2 import QtWidgets, QtCore, QtGui
from PySide2.QtCore import QRect
import subprocess
from socnavData import GenerateDataset
import math

import ui_drawgraph


class MainClass(QtWidgets.QWidget):
    def __init__(self, scenarios, start):
        super().__init__()
        self.scenarios = scenarios
        self.ui = ui_drawgraph.Ui_SocNavWidget()
        self.ui.setupUi(self)
        self.next_index = start
        self.view = None
        self.show()
        self.installEventFilter(self)
        self.load_next()
        self.ui.tableWidget.setRowCount(self.view.graph.features.shape[1]+1)
        self.ui.tableWidget.setColumnCount(1)
        self.ui.tableWidget.setColumnWidth(0, 200)
        self.ui.tableWidget.show()

        # Initialize table
        self.ui.tableWidget.setHorizontalHeaderItem(0, QtWidgets.QTableWidgetItem('value'))
        self.ui.tableWidget.horizontalHeader().hide()
        self.ui.tableWidget.setVerticalHeaderItem(0, QtWidgets.QTableWidgetItem('type'))
        self.ui.tableWidget.setItem(0, 0, QtWidgets.QTableWidgetItem('0'))
        features_aux = self.view.graph.features[1]
        for idx, feature in enumerate(features_aux, 1):
            self.ui.tableWidget.setVerticalHeaderItem(idx, QtWidgets.QTableWidgetItem(self.view.graph.all_features[idx-1]))
            self.ui.tableWidget.setItem(idx, 0, QtWidgets.QTableWidgetItem('0'))

    def load_next(self):
        if self.view:
            self.view.close()
            del self.view
        if self.next_index >= len(self.scenarios):
            print("All graphs shown")
            sys.exit(0)
        self.view = MyView(self.scenarios[self.next_index], self.ui.tableWidget)
        self.next_index += 1
        self.view.setParent(self.ui.widget)
        self.view.show()
        self.ui.widget.setFixedSize(self.view.width(), self.view.height())
        self.show()

        # Initialize table with zeros
        for idx in range(self.view.graph.features.shape[1]+2):
            self.ui.tableWidget.setItem(idx, 0, QtWidgets.QTableWidgetItem('0'))

    def eventFilter(self, widget, event):
        if event.type() == QtCore.QEvent.KeyPress:
            key = event.key()
            if key == QtCore.Qt.Key_Escape:
                sys.exit(0)
            else:
                if key == QtCore.Qt.Key_Return:
                    self.load_next()
                elif key == QtCore.Qt.Key_Enter:
                    self.close()
                return True
        return False


class MyView(QtWidgets.QGraphicsView):
    def __init__(self, graph, table):
        super().__init__()
        self.table = table
        self.graph = graph
        self.scene = QtWidgets.QGraphicsScene(self)
        self.nodeItems = dict()
        self.setFixedSize(1002, 1002)
        self.create_scene()
        self.installEventFilter(self)

    def create_scene(self):

        self.scene.setSceneRect(QtCore.QRectF(-500, -500, 1000, 1000))

        # Draw nodes and print labels
        for time_n, time in enumerate(self.graph.typeMap):
            for idx, n_type in time.items():
                # p = person
                # r = room
                # o = object
                # w = wall
                # g = goal
                index = idx + (time_n*len(time))
                if n_type == 'p':
                    colour = QtCore.Qt.blue
                    node_radius = 15
                    x = self.graph.features[index][self.graph.all_features.index('hum_x_pos')] * 800
                    y = self.graph.features[index][self.graph.all_features.index('hum_y_pos')] * 800
                elif n_type == 'o':
                    colour = QtCore.Qt.green
                    node_radius = 10
                    x = self.graph.features[index][self.graph.all_features.index('obj_x_pos')] * 800
                    y = self.graph.features[index][self.graph.all_features.index('obj_y_pos')] * 800
                elif n_type == 'r':
                    colour = QtCore.Qt.black
                    node_radius = 7
                    x = 0
                    y = 0
                elif n_type == 'w':
                    colour = QtCore.Qt.cyan
                    node_radius = 10
                    x = self.graph.features[index][self.graph.all_features.index('wall_x_pos')] * 800
                    y = self.graph.features[index][self.graph.all_features.index('wall_y_pos')] * 800
                elif n_type == 'g':
                    colour = QtCore.Qt.darkRed
                    node_radius = 10
                    x = self.graph.features[index][self.graph.all_features.index('goal_x_pos')] * 800
                    y = self.graph.features[index][self.graph.all_features.index('goal_y_pos')] * 800
                else:
                    colour = None
                    node_radius = None
                    x = None
                    y = None

                if x is not None:
                    shift = time_n * (node_radius + 20)
                    x += shift
                    item = self.scene.addEllipse(x - node_radius, y - node_radius, node_radius*2,
                                                 node_radius*2, brush=colour)
                else:
                    print(n_type)
                    print("Invalid node")
                    sys.exit(0)

                c = (x, y)
                self.nodeItems[index] = (item, c, n_type)

                # Print labels of the nodes
                text = self.scene.addText(n_type)
                text.setDefaultTextColor(QtCore.Qt.magenta)
                text.setPos(*c)

        self.setScene(self.scene)

        # Draw edges
        edges = self.graph.edges()

        for e_id in range(len(edges[0])):
            edge_a = edges[0][e_id].item()
            edge_b = edges[1][e_id].item()

            type_a = self.nodeItems[edge_a][2]
            type_b = self.nodeItems[edge_b][2]

            if edge_a == edge_b:  # No self edges printed
                continue

            ax, ay = self.nodeItems[edge_a][1]
            bx, by = self.nodeItems[edge_b][1]
            pen = QtGui.QPen()
            rel_type = self.graph.edata['rel_type'][e_id].item()
            colour, width = self.type_to_colour_width(rel_type, type_a, type_b)

            if colour is None or width is None:
                print("Error for link between these two types:")
                print(type_a)
                print(type_b)
                sys.exit(0)

            pen.setColor(colour)
            pen.setWidth(width)
            self.scene.addLine(ax, ay, bx, by, pen=pen)

    @staticmethod
    def type_to_colour_width(rel_type, type1, type2):
        if type1 == type2:
            if rel_type in [11, 12, 13, 14]:
                colour = QtCore.Qt.red
                width = 1
            else:
                colour = QtCore.Qt.black
                width = 5
        elif (type1 == 'p' and type2 == 'o') or (type1 == 'o' and type2 == 'p'):
            colour = QtCore.Qt.black
            width = 5
        elif (type1 == 'r' or type2 == 'r') and rel_type not in [11, 12, 13, 14]:
            colour = QtCore.Qt.lightGray
            width = 1
        else:
            colour = None
            width = None

        return colour, width

    def closest_node_view(self, event_x, event_y):
        WINDOW_SIZE = 496
        closest_node = -1
        closest_node_type = -1
        x_mouse = (event_x - WINDOW_SIZE)
        y_mouse = (event_y - WINDOW_SIZE)
        old_dist = WINDOW_SIZE * 2

        for idx, node in self.nodeItems.items():
            x = node[1][0]
            y = node[1][1]
            dist = abs(x - x_mouse) + abs(y - y_mouse)
            if dist < old_dist:
                old_dist = dist
                closest_node = idx
                closest_node_type = node[2]

        return closest_node, closest_node_type

    # def set_pixmap(self, pixmap_path):
    #     pixmap = QtGui.QPixmap(pixmap_path)
    #     pixmap_item = QtWidgets.QGraphicsPixmapItem(pixmap)
    #     pixmap_item.setPos(-455, -500)
    #     pixmap_item.setScale(1.)
    #     self.scene.addItem(pixmap_item)
    #     #  self.scene.addRect(-30, -30, 60, 60, pen=QtGui.QPen(QtCore.Qt.white), brush=QtGui.QBrush(QtCore.Qt.white))

    def eventFilter(self, widget, event):
        if event.type() == QtCore.QEvent.MouseButtonPress:
            closest_node_id, n_type = self.closest_node_view(event.x()-7, event.y()-7)
            if n_type == -1:
                print('Not valid label')
            self.table.setItem(0, 0, QtWidgets.QTableWidgetItem(n_type))
            features = self.graph.features[closest_node_id]

            one_hot_nodes = [str(int(x)) for x in features[0:5]]
            one_hot_times = [str(int(x)) for x in features[5:9]]
            rest = ['{:1.3f}'.format(x) for x in features[9:len(features)]]
            features_format = one_hot_nodes + one_hot_times + rest

            for idx, feature in enumerate(features_format):
                self.table.setItem(idx, 1, QtWidgets.QTableWidgetItem(feature))

            return True
        return False


if __name__ == '__main__':

    dataset = GenerateDataset(sys.argv[1], mode='run', alt='1', debug=True)
    scenarios = dataset.data

    app = QtWidgets.QApplication(sys.argv)
    if len(sys.argv) > 2:
        start = int(sys.argv[2])
    else:
        start = 0

    view = MainClass(scenarios, start)

    exit_code = app.exec_()
    sys.exit(exit_code)
