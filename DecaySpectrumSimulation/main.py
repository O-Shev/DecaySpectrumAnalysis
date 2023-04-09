import sys
from PyQt5.QtCore import Qt
from PyQt5 import QtGui
from PyQt5.QtWidgets import QDialog, QApplication, QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QTableWidget, QTableWidgetItem, QLineEdit
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

import matplotlib.pyplot as plt
import numpy as np
import random
import operator

from plot_figure import plot_ax as class_of_plot

class Window(QDialog):

    # constructor
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)

        self.ploti = class_of_plot()
        self.ploti.update_graf()

        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        hl = QHBoxLayout()
        vll = QVBoxLayout()
        vlr = QVBoxLayout()

###label table #self.table_peak line
        if(True):
            vlt = QVBoxLayout()

            label_peak = QLabel('peaks', self)
            label_peak.setFont(QtGui.QFont('SansSerif', 10))
            label_peak.setAlignment(Qt.AlignCenter)
            label_peak.setFixedHeight(20)
            vlt.addWidget(label_peak)

            self.table_peak = QTableWidget(self)
            self.table_peak.setColumnCount(3) 
            self.table_peak.setRowCount(1)
            self.table_peak.setHorizontalHeaderLabels(["E, (Kev)", "I", "FWHM, (Kev)"])
            self.table_peak.resizeColumnsToContents()
            self.table_peak.setFixedSize(270, 200)
            

            for iter in range (len(self.ploti.line_Energy_characteristic)):
                self.table_peak.setItem(iter, 0, QTableWidgetItem(str(self.ploti.line_Energy_characteristic[iter]['E'])))
                self.table_peak.setItem(iter, 1, QTableWidgetItem(str(self.ploti.line_Energy_characteristic[iter]['I'])))
                self.table_peak.setItem(iter, 2, QTableWidgetItem(str(self.ploti.line_Energy_characteristic[iter]['FWHM'])))
                rowPosition = self.table_peak.rowCount() 
                self.table_peak.insertRow(rowPosition)
            self.table_peak.resizeColumnsToContents()

            
            

            vlt.addWidget(self.table_peak)
            vlr.addLayout(vlt)

###########background
        if(True):
            vlt = QVBoxLayout()

            label_back = QLabel('background', self)
            label_back.setFont(QtGui.QFont('SansSerif', 10))
            label_back.setAlignment(Qt.AlignCenter)
            label_back.setFixedHeight(20)
            vlt.addWidget(label_back)

        #####линейной (y=a*x+b)
            thl = QHBoxLayout()

            label_background_line = QLabel('linear background (y=a*x+b):       ', self)
            label_background_line.setFont(QtGui.QFont('SansSerif', 10))
            label_background_line.setFixedHeight(20)
            thl.addWidget(label_background_line)

            #x
            x = QLabel('x=', self)
            x.setFont(QtGui.QFont('SansSerif', 10))
            x.setAlignment(Qt.AlignRight)

            self.lxEdit = QLineEdit()
            self.lxEdit.setMaxLength(10)
            self.lxEdit.setMaximumWidth(40)
            self.lxEdit.setText(str(self.ploti.back_function_dict['la']))
        
            thl.addWidget(x)
            thl.addWidget(self.lxEdit)
            ####
            #y
            y = QLabel('y=', self)
            y.setFont(QtGui.QFont('SansSerif', 10))
            y.setAlignment(Qt.AlignRight)

            self.lyEdit = QLineEdit()
            self.lyEdit.setMaxLength(10)
            self.lyEdit.setMaximumWidth(40)
            self.lyEdit.setText(str(self.ploti.back_function_dict['lb']))

            thl.addWidget(y)
            thl.addWidget(self.lyEdit)
            #####

            vlt.addLayout(thl)
        ########

        ###кспоненциальной (y=a*exp(-b*x))
            thl = QHBoxLayout()

            label_background_line = QLabel('exp background (y=a*exp(-b*x):   ', self)
            label_background_line.setFont(QtGui.QFont('SansSerif', 10))
            label_background_line.setFixedHeight(20)
            thl.addWidget(label_background_line)

            #x
            x = QLabel('x=', self)
            x.setFont(QtGui.QFont('SansSerif', 10))
            x.setAlignment(Qt.AlignRight)

            self.exEdit = QLineEdit()
            self.exEdit.setMaxLength(10)
            self.exEdit.setMaximumWidth(40)
            self.exEdit.setText(str(self.ploti.back_function_dict['ea']))

            thl.addWidget(x)
            thl.addWidget(self.exEdit)
            ####
            #y
            y = QLabel('y=', self)
            y.setFont(QtGui.QFont('SansSerif', 10))
            y.setAlignment(Qt.AlignRight)

            self.eyEdit = QLineEdit()
            self.eyEdit.setMaxLength(10)
            self.eyEdit.setMaximumWidth(40)
            self.eyEdit.setText(str(self.ploti.back_function_dict['eb']))

            thl.addWidget(y)
            thl.addWidget(self.eyEdit)
            #####

            vlt.addLayout(thl)
        ########
            vlr.addLayout(vlt)
#########range Energy
        if(True):
            vlt = QVBoxLayout()
            

            label_line = QLabel('range Energy', self)
            label_line.setFont(QtGui.QFont('SansSerif', 10))
            label_line.setFixedHeight(20)
            label_line.setAlignment(Qt.AlignCenter)
            vlt.addWidget(label_line)

            thl = QHBoxLayout()
            label_line = QLabel('Emax(Kev)      =  ', self)
            label_line.setFont(QtGui.QFont('SansSerif', 10))
            label_line.setFixedHeight(20)
            self.Emax_Edit = QLineEdit()
            self.Emax_Edit.setMaxLength(10)
            self.Emax_Edit.setText(str(self.ploti.range_Energy['Emax']))


            thl.addWidget(label_line)
            thl.addWidget(self.Emax_Edit)
            vlt.addLayout(thl)


            thl = QHBoxLayout()
            label_line = QLabel('num channels  =  ', self)
            label_line.setFont(QtGui.QFont('SansSerif', 10))
            label_line.setFixedHeight(20)
            self.nc_Edit = QLineEdit()
            self.nc_Edit.setMaxLength(10)
            self.nc_Edit.setText(str(self.ploti.range_Energy['numCh']))
            thl.addWidget(label_line)
            thl.addWidget(self.nc_Edit)
            vlt.addLayout(thl)


            thl = QHBoxLayout()
            label_line = QLabel('bias(Kev)        =  ', self)
            label_line.setFont(QtGui.QFont('SansSerif', 10))
            label_line.setFixedHeight(20)
            self.b_Edit = QLineEdit()
            self.b_Edit.setMaxLength(10)
            self.b_Edit.setText(str(self.ploti.range_Energy['bias']))
            thl.addWidget(label_line)
            thl.addWidget(self.b_Edit)
            vlt.addLayout(thl)


            vlr.addLayout(vlt)

####################
        
        #####button
        butlay = QHBoxLayout()
        btn_update = QPushButton("update", self)
        btn_update.clicked.connect(self.update)
        butlay.addWidget(btn_update)

        butlay2 = QHBoxLayout()

        btn_update1 = QPushButton("fig 1", self)
        btn_update1.clicked.connect(self.initial1)
        butlay2.addWidget(btn_update1)

        btn_update2 = QPushButton("fig 2", self)
        btn_update2.clicked.connect(self.initial2)
        butlay2.addWidget(btn_update2)

        btn_update3 = QPushButton("fig 3", self)
        btn_update3.clicked.connect(self.initial3)
        butlay2.addWidget(btn_update3)

        btn_update4 = QPushButton("fig 4", self)
        btn_update4.clicked.connect(self.initial4)
        butlay2.addWidget(btn_update4)
        ########

        vlr.addLayout(butlay2)
        vlr.addLayout(butlay)
        hl.addLayout(vlr)
        vll.addWidget(self.toolbar)
        vll.addWidget(self.canvas)
        hl.addLayout(vll)

        self.setLayout(hl)

        self.plot(self.ploti)
        
    def initial1(self):
        # create an axis
        ploti_i = self.update()

        figure = plt.figure()

        ax = figure.add_subplot(111)
        
        

        ######dimension
        ax.set_xlim([ploti_i.range_Energy['bias']*ploti_i.range_Energy['numCh']/ploti_i.range_Energy['Emax'], ploti_i.range_Energy['numCh']])


        # plot data
        ax.plot(ploti_i.graf1[0], ploti_i.graf1[1], ',-')

        # refresh canvas
        figure.show()

    def initial2(self):
        # create an axis
        ploti_i = self.update()

        figure = plt.figure()

        ax = figure.add_subplot(111)
        
        

        ######dimension
        ax.set_xlim([ploti_i.range_Energy['bias']*ploti_i.range_Energy['numCh']/ploti_i.range_Energy['Emax'], ploti_i.range_Energy['numCh']])


        # plot data
        ax.plot(ploti_i.graf2[0], ploti_i.graf2[1], ',-')

        # refresh canvas
        figure.show()

    def initial3(self):
        # create an axis
        ploti_i = self.update()

        figure = plt.figure()

        ax = figure.add_subplot(111)
        
        

        ######dimension
        ax.set_xlim([ploti_i.range_Energy['bias']*ploti_i.range_Energy['numCh']/ploti_i.range_Energy['Emax'], ploti_i.range_Energy['numCh']])


        # plot data
        ax.plot(ploti_i.graf3[0], ploti_i.graf3[1], ',')

        # refresh canvas
        figure.show()

    def initial4(self):
        # create an axis
        ploti_i = self.update()

        figure = plt.figure()

        ax = figure.add_subplot(111)
        
        

        ######dimension
        ax.set_xlim([ploti_i.range_Energy['bias']*ploti_i.range_Energy['numCh']/ploti_i.range_Energy['Emax'], ploti_i.range_Energy['numCh']])


        # plot data
        ax.plot(ploti_i.graf4[0], ploti_i.graf4[1], ',')

        # refresh canvas
        figure.show()

    def plot(self, ploti):

        # random data

        # clearing old figure
        self.figure.clear()

        # create an axis
        ax1 = self.figure.add_subplot(221)
        ax2 = self.figure.add_subplot(222)
        ax3 = self.figure.add_subplot(223)
        ax4 = self.figure.add_subplot(224)

        
        

        ######dimension
        ax1.set_xlim([ploti.range_Energy['bias']*ploti.range_Energy['numCh']/ploti.range_Energy['Emax'], ploti.range_Energy['numCh']])
        ax2.set_xlim([ploti.range_Energy['bias']*ploti.range_Energy['numCh']/ploti.range_Energy['Emax'], ploti.range_Energy['numCh']])
        ax3.set_xlim([ploti.range_Energy['bias']*ploti.range_Energy['numCh']/ploti.range_Energy['Emax'], ploti.range_Energy['numCh']])
        ax4.set_xlim([ploti.range_Energy['bias']*ploti.range_Energy['numCh']/ploti.range_Energy['Emax'], ploti.range_Energy['numCh']])
        ######


        # plot data
        ax1.plot(ploti.graf1[0], ploti.graf1[1], ',-')
        ax2.plot(ploti.graf2[0], ploti.graf2[1], ',-')
        ax3.plot(ploti.graf3[0], ploti.graf3[1], ',')
        ax4.plot(ploti.graf4[0], ploti.graf4[1], ',')

        # refresh canvas
        self.canvas.draw()

    def update(self):
        ploti_next = class_of_plot()

        #####table_E
        flag_table = True
        line_Energy_characteristic = []
        for iter in range(self.table_peak.rowCount()-1):
            str_cell = str(self.table_peak.item(iter, 0).text())
            if(str_cell == ''): continue
            jter = 0
            for jter in range(3):
                try:
                    float_cell = float(self.table_peak.item(iter, jter).text())
                    if(float_cell < 0):
                        flag_table = False
                        self.table_peak.item(iter, jter).setToolTip("invalid value: the value cannot be negative")
                        self.table_peak.item(iter, jter).setBackground(QtGui.QColor(255, 36, 0))
                except:
                    flag_table = False
                    self.table_peak.item(iter, jter).setToolTip("invalid value: the value must be a float")
                    self.table_peak.item(iter, jter).setBackground(QtGui.QColor(255, 36, 0))

        if(flag_table == True):
            for iter in range(self.table_peak.rowCount()):
                try:
                      if(float(self.table_peak.item(iter, 0).text()) >=  0 and float(self.table_peak.item(iter, 1).text()) >= 0 and float(self.table_peak.item(iter, 2).text()) >= 0):
                            line_Energy_characteristic.append({'E': float(self.table_peak.item(iter, 0).text()), 'I': float(self.table_peak.item(iter, 1).text()), 'FWHM': float(self.table_peak.item(iter, 2).text())})
                except:
                     None

            line_Energy_characteristic.sort(key=operator.itemgetter('E'))
            ploti_next.line_Energy_characteristic = line_Energy_characteristic
        
            self.table_peak.setRowCount(0)
            self.table_peak.setRowCount(1)

            for iter in range (len(ploti_next.line_Energy_characteristic)):
                    self.table_peak.setItem(iter, 0, QTableWidgetItem(str(ploti_next.line_Energy_characteristic[iter]['E'])))
                    self.table_peak.setItem(iter, 1, QTableWidgetItem(str(ploti_next.line_Energy_characteristic[iter]['I'])))
                    self.table_peak.setItem(iter, 2, QTableWidgetItem(str(ploti_next.line_Energy_characteristic[iter]['FWHM'])))
                    rowPosition = self.table_peak.rowCount() 
                    self.table_peak.insertRow(rowPosition)
            self.table_peak.resizeColumnsToContents()
        #######

        ####back? i'll be back
        f = True
        try:
            ploti_next.back_function_dict['la'] == float(self.lxEdit.text())
            
        except:
            self.lxEdit.setStyleSheet("background-color: rgb(255, 36, 0);")
            self.lxEdit.setToolTip("invalid value: the value must be a float, value hase benn set 0")
            ploti_next.back_function_dict['la'] = 0
            f = False
        if(f): self.lxEdit.setStyleSheet("background-color: rgb(255, 255, 255);")

        f = True
        try:
            ploti_next.back_function_dict['lb'] = float(self.lyEdit.text())
        except:
            self.lyEdit.setStyleSheet("background-color: rgb(255, 36, 0);")
            self.lyEdit.setToolTip("invalid value: the value must be a float, value hase benn set 0")
            ploti_next.back_function_dict['lb'] = 0
            f = False
        if(f): self.lyEdit.setStyleSheet("background-color: rgb(255, 255, 255);")
        
        f = True
        try:
            ploti_next.back_function_dict['ea'] = float(self.exEdit.text())
        except:
            self.exEdit.setStyleSheet("background-color: rgb(255, 36, 0);")
            self.exEdit.setToolTip("invalid value: the value must be a float, value hase benn set 0")
            ploti_next.back_function_dict['ea'] = 0   
            f = False
        if(f): self.exEdit.setStyleSheet("background-color: rgb(255, 255, 255);")
        
        f = True
        try:
            ploti_next.back_function_dict['eb'] = float(self.eyEdit.text())
        except:
            self.eyEdit.setStyleSheet("background-color: rgb(255, 36, 0);")
            self.eyEdit.setToolTip("invalid value: the value must be a float, value hase benn set 0")
            ploti_next.back_function_dict['eb'] = 0
            f = False
        if(f): self.eyEdit.setStyleSheet("background-color: rgb(255, 255, 255);")
        ##########


        ##########range E
        f = True
        try:
            if(float(self.Emax_Edit.text()) >= 1):
                ploti_next.range_Energy['Emax'] = float(self.Emax_Edit.text())
            else:
                raise Exception("invalid argument")
        except:
            self.Emax_Edit.setStyleSheet("background-color: rgb(255, 36, 0);")
            self.Emax_Edit.setToolTip("invalid value: the value must be a float and > 0, value hase benn set 1")
            ploti_next.range_Energy['Emax'] = 1
            f = False
        if(f): self.Emax_Edit.setStyleSheet("background-color: rgb(255, 255, 255);")


        f = True
        try:
            if(float(self.nc_Edit.text()) >= 1):
                ploti_next.range_Energy['numCh'] = float(self.nc_Edit.text())
            else:
                raise Exception("invalid argument")
        except:
            self.nc_Edit.setStyleSheet("background-color: rgb(255, 36, 0);")
            self.nc_Edit.setToolTip("invalid value: the value must be a float and > 0, value hase benn set 1")
            ploti_next.range_Energy['numCh'] = 1
            f = False
        if(f): self.nc_Edit.setStyleSheet("background-color: rgb(255, 255, 255);")

        f = True
        try:
            ploti_next.range_Energy['bias'] = float(self.b_Edit.text())
        except:
            self.b_Edit.setStyleSheet("background-color: rgb(255, 36, 0);")
            self.b_Edit.setToolTip("invalid value: the value must be a float and > 0, value hase benn set 0")
            ploti_next.range_Energy['bias'] = 0
            f = False
        if(f): self.b_Edit.setStyleSheet("background-color: rgb(255, 255, 255);")


        #############


        ploti_next.update_graf()
        self.plot(ploti_next)

        return ploti_next




if __name__ == '__main__':
    app = QApplication(sys.argv)

    main = Window()
    main.show()
    sys.exit(app.exec_())
