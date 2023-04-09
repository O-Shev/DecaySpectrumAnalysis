from scipy.optimize import curve_fit
from scipy.odr import ODR, Model, Data, RealData
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import copy

import sys
from PyQt5.QtCore import Qt
from PyQt5 import QtGui
from PyQt5.QtGui import QPainter, QColor, QPen
from PyQt5.QtWidgets import QDialog, QApplication, QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QTableWidget, QTableWidgetItem, QLineEdit, QCheckBox, QSpinBox, QSlider, QComboBox, QDoubleSpinBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from scipy.signal import savgol_filter

class Window(QDialog):
    
    def fon(self, arr, ea, eb, la, lb):
        if(np.size(arr)>1):
            y = []
            for x in arr:
                y.append(ea*np.exp(-1*eb*x) + la*x + lb)
        elif(np.size(arr) == 1):
            x = arr
            y = ea*np.exp(-1*eb*x) + la*x + lb
        else: print("че то не то с входными данными в фон")
        return y

    def gaussian(self, arr, A, mu, sigma):
        y = []
        for x in arr:
            y.append((A * math.exp(-0.5*((x-mu)/sigma)**2) / sigma / math.sqrt(2*math.pi)))
        return y

    def readingFile(self):
        inputf = open('spectr.txt', 'r', encoding="utf8")
        s = []
        for linestr in inputf:
            tempstr = linestr.split(' ')
            for iters in tempstr:
                tempstrlist = iters.split('\t')
                for tempiters in tempstrlist:
                    mas = tempiters.split('\n')
                    for  i in mas:
                        if(i != ' ' and i != '\n' and i  != '\t' and i != ''):
                            try:
                                s.append(int(i))
                            except:
                                s.append(i)
        grafX = []
        grafY = []
        i = 0
        for i in range(int((len(s) - 2))):
            if(i%2 == 0):
                grafX.append(s[i+2])
            else: grafY.append(s[i+2])
        graf = np.vstack((grafX, grafY))
        return graf

    def readingFileS(self):
        self.config_plot = dict.fromkeys(['smoothingA', 'smoothingB', 'pdf', 'aad', 'smsearchA', 'smsearchB', 'BasDer', '2ndDer', '1stDer', 'countf', 'mf', 'kf'])
        configf = open('config.txt', 'r', encoding="utf8")
        
        for linestr in configf:
            s = []
            tempstr = linestr.split(' ')
            for iters in tempstr:
                tempstrlist = iters.split('\t')
                for tempiters in tempstrlist:
                    s.append(tempiters)
            while(s.count('') != 0):
                s.remove('')
            if(s[0][0] != '#'):
                if(s[0] == 'smoothing'):
                    self.config_plot['smoothingA'] = int(s[1])
                if(s[0] == 'smoothing'):
                    self.config_plot['smoothingB'] = int(s[2])
                if(s[0] == 'pdf'):
                    self.config_plot['pdf'] = float(s[1])
                if(s[0] == 'aad'):
                    self.config_plot['aad'] = int(s[1])
                if(s[0] == 'smsearch'):
                    self.config_plot['smsearchA'] = int(s[1])
                if(s[0] == 'smsearch'):
                    self.config_plot['smsearchB'] = int(s[2])

                if(s[0] == 'BasDer'):
                    self.config_plot['BasDer'] = int(s[1])
                if(s[0] == '2ndDer'):
                    self.config_plot['2ndDer'] = int(s[1])
                if(s[0] == '1stDer'):
                    self.config_plot['1stDer'] = int(s[1])
                if(s[0] == 'float'):
                    self.config_plot['countf'] = int(s[3])
                if(s[0] == 'float'):
                    self.config_plot['mf'] = int(s[1])
                if(s[0] == 'float'):
                    self.config_plot['kf'] = int(s[2])
        print(self.config_plot)

    def smooth(self, graf, smooth_range):
        #graf_smooth = np.copy(graf)
        #shape = np.size(graf[0])
        #def func(x, A, B, C):
        #    y = np.empty(np.size(x))
        #    y = x*x*A + B*x + C
        #    return y     
        #k = smooth_range
        #for i in range(0, shape):    
        #    if(i < k):
        #        tempY = graf[1][0:(2*k+1)]
        #        tempX = graf[0][0:(2*k+1)]
        #        popt, pcov = curve_fit(func, tempX, tempY, (1, 1, 1))
        #        a, b, c = popt
        #        graf_smooth[1][i] = func(graf_smooth[0][i], a, b, c)
        #    elif(i >= k and (shape-i) > k): 
        #        tempY = graf[1][i-k:i+k]
        #        tempX = graf[0][i-k:i+k]
        #        popt, pcov = curve_fit(func, tempX, tempY, (1, 1, 1))
        #        a, b, c = popt
        #        graf_smooth[1][i] = func(graf_smooth[0][i], a, b, c)
        #    else:
        #        tempY = graf[1][(shape-2*k+1):shape]
        #        tempX = graf[0][(shape-2*k+1):shape]
        #        popt, pcov = curve_fit(func, tempX, tempY, (1, 1, 1))
        #        a, b, c = popt
        #        graf_smooth[1][i] = func(graf_smooth[0][i], a, b, c)

        if(smooth_range%2 == 0): smooth_range +=1
        yhat = savgol_filter(graf[1], smooth_range, 3)
        graf_smooth = np.vstack((graf[0], yhat))
        
        return graf_smooth

    def my_derivative(self, graf_smooth, m):
        i = 0
        x = []
        y = []
        A, B  = np.shape(graf_smooth)
        for iter in graf_smooth[0]:
            i+=1
    
            if(i<=m or (B - 1 - i) <= 2*m): continue
            i-=1
            Begin = i - m
            End = i + 2*m -1
            D = 0
            V = 0
            L = 0
            for j in range(Begin, End):
                if(i-m <= j <= i-1): L = -1
                if(i<=j<= i+m-1): L= 2
                if(i+m <= j<=i+2*m-1): L=-1

                D += L*graf_smooth[1][j]
                V += L*L*graf_smooth[1][j]
            c = D/(V**(1/2))
    
            x.append(i+m/2)
            y.append(c)
            i+=1

        C = np.vstack((x, y))
        return C

    def second_derivative(self, graf_smooth):
        y = []
        x = graf_smooth[0][1:(np.size(graf_smooth[0])-1)]
        for i in range(1, (np.size(graf_smooth[0])-1)):
            y.append(self.spin_var_2der.value() * -1*(graf_smooth[1][i+1] - 2*graf_smooth[1][i] + graf_smooth[1][i-1]))

        graf = np.vstack((x, y))
        return  graf

    def first_derivative(self, graf_smooth):
        y = [(self.spin_var_1der.value()*(graf_smooth[1][i+1] - graf_smooth[1][i])) for i, _ in enumerate(graf_smooth[0][0:-2])]
        x = graf_smooth[0][0:-2]
        return np.vstack((x, y))      

    def search_peaks_of_C(self, C, m, k):
        #if ci > m -> near peak
        sizeA, sizeB = np.shape(C)
        lines_c = []
        for_delt_C = []

        temp_index = []
        # k - mean of C

        flag = False
        mean_last_i = 0
        for i in range(1, sizeB-1):
            if(k == 0): mean_i = C[1][i]
            elif(i < k):
                temp = C[1][0:(2*k)]
                mean_i = np.sum(temp)/np.shape(temp)
            elif(i >= k and (sizeB-1-i) > k): 
                temp = C[1][i-k:i+k]
                mean_i = np.sum(temp)/np.shape(temp)
            else:
                temp = C[1][(sizeB-1-2*k):sizeB-1]
                mean_i = np.sum(temp)/np.shape(temp)
        
            if(mean_i > m):
                if(mean_i<mean_last_i and flag !=True):
                    lines_c.append(C[0][i-1])
                    temp_index.append(i-1)
                    flag = True
            if(flag == True and mean_i>mean_last_i):
                flag = False
            if(mean_last_i > m and mean_i <= m):
                flag = False

            mean_last_i = mean_i
    

        for i in range(0, np.size(lines_c)):
            xl = 0
            xr = 0

            kaef = 0
            j = temp_index[i]
            while(True):
                xl=C[0][j]
                if(C[1][j] <=kaef): break
                j -= 1

            j = temp_index[i]
            while(True):
                xr=C[0][j]
                if(C[1][j] <=kaef): break
                j += 1
            
            for_delt_C.append([xl, xr])
        for_delt_C = np.array(for_delt_C)
        for_delt_C = for_delt_C.transpose()
        return lines_c, for_delt_C

    def fit_user(self, graf_smooth, a, b, C, lines_C, for_delt_C): 
        delt_C = []
        for i in range(0, int(np.size(for_delt_C)/2)):
            delt_C.append(for_delt_C[1][i] - for_delt_C[0][i])

        a = int(a)
        b = int(b)
        x = graf_smooth[0][a:b]
        y = []
        for i in range(a, b): y.append(graf_smooth[1][i])
        mas_for_fit = np.vstack((x, y))

        iter_m = 0
        for iterm in lines_C:
            if(a < iterm < b): 
                mu = int(iterm)
                sig = delt_C[iter_m]
            iter_m +=1
        A = graf_smooth[1][mu]

        xlf = 0
        xrf = 0
        k = 5
        C_min_for_xf = -10
    
        #####xlf
        if(True):
            iter_for_graf_temp = np.where(graf_smooth[0] == mu)
            iter_for_graf = iter_for_graf_temp[0]

            iter_for_c_temp = np.where(C[0] == mu)
            iter_for_c = iter_for_c_temp[0]

            ii = iter_for_c[0]
            i = iter_for_graf[0]
            mean_i = 0
            mean_last_i = 100
            sizeB = np.size(graf_smooth[1])
            limit_of_iter = 5000
            limit_i = 0
            while(True):
                if(limit_i > limit_of_iter): 
                    print("limit_while")
                    break
                limit_i += 1
                if(k == 0): mean_i = graf_smooth[1][i]
                elif(i < k):
                    temp = graf_smooth[1][0:(2*k)]
                    mean_i = np.sum(temp)/np.shape(temp)
                elif(i >= k and (sizeB-1-i) > k): 
                    temp = graf_smooth[1][i-k:i+k]
                    mean_i = np.sum(temp)/np.shape(temp)
                else:
                    temp = graf_smooth[1][(sizeB-1-2*k):sizeB-1]
                    mean_i = np.sum(temp)/np.shape(temp)

                if(mean_i > mean_last_i and C[1][ii] < C_min_for_xf):
                     T = np.where(graf_smooth[0] ==(graf_smooth[0][i]-20))
                     xlf = T[0][0]
                
                     break
                mean_last_i = mean_i  
                i-=1
                ii-=1
        #######

        #####xrf
        if(True):
            iter_for_graf_temp = np.where(graf_smooth[0] == mu)
            iter_for_graf = iter_for_graf_temp[0]

            iter_for_c_temp = np.where(C[0] == mu)
            iter_for_c = iter_for_c_temp[0]

            ii = iter_for_c[0]
            i = iter_for_graf[0]
            mean_i = 0
            mean_last_i = 100
            sizeB = np.size(graf_smooth[1])
            limit_of_iter = 5000
            limit_i = 0
            while(True):
                if(limit_i > limit_of_iter): 
                    print("limit_while")
                    break
                if(k == 0): mean_i = graf_smooth[1][i]
                elif(i < k):
                    temp = graf_smooth[1][0:(2*k)]
                    mean_i = np.sum(temp)/np.shape(temp)
                elif(i >= k and (sizeB-1-i) > k): 
                    temp = graf_smooth[1][i-k:i+k]
                    mean_i = np.sum(temp)/np.shape(temp)
                else:
                    temp = graf_smooth[1][(sizeB-1-2*k):sizeB-1]
                    mean_i = np.sum(temp)/np.shape(temp)

                if(mean_i < mean_last_i and C[1][ii] < C_min_for_xf):
                     T = np.where(graf_smooth[0] ==(graf_smooth[0][i]+20))
                     xrf = T[0][0]

                     break
                mean_last_i = mean_i  
                i+=1
                ii+=1
        #######
        ylf = int(graf_smooth[1][xlf])
        yrf = int(graf_smooth[1][xrf])
        xlf = int(graf_smooth[0][xlf])
        xrf = int(graf_smooth[0][xrf])
   

        for q in range(0, np.size(mas_for_fit[1])):
            x_temp = int(mas_for_fit[0][q])
            mas_for_fit[1][q] -= ((x_temp-xlf)*(yrf-ylf)/(xrf-xlf)) + ylf
        
    
        Y = 0

        popt, pcov = curve_fit(gaussian, mas_for_fit[0], mas_for_fit[1], (A, mu, sig))    
        A, mu, sig = popt
        y = gaussian(x, A, mu, sig)
        mas_fit = np.vstack((x, y))


        S = 0
        for s_i in mas_fit[1]:
            S+=s_i
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(mas_for_fit[0], mas_for_fit[1], linestyle='-', marker=',')
        ax.plot(mas_fit[0], mas_fit[1], linestyle='-', marker=',')
        sig = abs(sig)
        ax.set_title("mu=" + str(int(mu)) + "  sig=" + str(int(sig)) + "   FWHM" + str(int(sig*2.355)) + "   S=" + str(int(S)))
        fig.show()

    def fit_auto_for_C(self, graf_smooth, C):
        fit_ae = []
        fit_var = []
        #############search
        if(True):
            sizeA, sizeB = np.shape(C)

            m = self.qspin_search_m.value()
            k = self.qspin_search_k.value()

            flagb = False
            flage = False
            mean_i = 0
            mean_last_i = 0
            temp_line_b = 0
            temp_line_e = 0
            iter = 0

            lines_peak = []
            lines = []
            temp_line_mb = []
            temp_line_me = []


            if(self.combo_der.currentIndex() != 2):
                for i in range(1, sizeB-1):
                    if(k == 0): mean_i = C[1][i]
                    elif(i < k):
                        temp = C[1][0:(2*k)]
                        mean_i = np.sum(temp)/np.shape(temp)
                    elif(i >= k and (sizeB-1-i) > k): 
                        temp = C[1][i-k:i+k]
                        mean_i = np.sum(temp)/np.shape(temp)
                    else:
                        temp = C[1][(sizeB-1-2*k):sizeB-1]
                        mean_i = np.sum(temp)/np.shape(temp)

                    if(mean_i > mean_last_i):
                        if(flagb == False):
                            temp_line_b = C[0][i]
                            iter = i
                            flagb = True
                    elif(mean_i < mean_last_i):
                        if(mean_i > m and flagb == True):
                            temp_line_mb.append(int(temp_line_b))
                            lines_peak.append(int(C[0][i]))
                            flage = True
                        flagb = False

                    if(flage == True):
                        if(mean_i > mean_last_i):
                            temp_line_me.append(int(C[0][i]))
                            flage = False
        
                    mean_last_i = mean_i
            else:
                for i in range(1, sizeB-1):
                    if(k == 0): mean_i = C[1][i]
                    elif(i < k):
                        temp = C[1][0:(2*k)]
                        mean_i = np.sum(temp)/np.shape(temp)
                    elif(i >= k and (sizeB-1-i) > k): 
                        temp = C[1][i-k:i+k]
                        mean_i = np.sum(temp)/np.shape(temp)
                    else:
                        temp = C[1][(sizeB-1-2*k):sizeB-1]
                        mean_i = np.sum(temp)/np.shape(temp)

                    if(mean_i > mean_last_i):
                        if(flagb == False):
                            temp_line_b = C[0][i]
                            iter = i
                            flagb = True
                    elif(mean_i < mean_last_i):
                        if(mean_i > m and flagb == True):
                            temp_line_mb.append(int(temp_line_b))
                            flage = True
                        flagb = False

                    if(flage == True):
                        if(mean_i > mean_last_i and -0.1 < mean_i < 0.1):
                            temp_line_me.append(int(C[0][i]))
                            flage = False
        
                    mean_last_i = mean_i
                for i, _ in enumerate(temp_line_mb):
                    b, = np.where(graf_smooth[0] == temp_line_mb[i])
                    e, = np.where(graf_smooth[0] == temp_line_me[i])
                    iteri, = np.where(graf_smooth[1] == np.max(graf_smooth[1][int(b):int(e)]))
                    lines_peak.append(int(graf_smooth[0][int(iteri)]))
            lines = np.vstack((lines_peak, temp_line_mb, temp_line_me))
        #############
        #####fon
        fon_x =[]
        fon_y=[]
        for iter_fon_1 in range(1, int(lines[1][0])): fon_x.append(iter_fon_1)
        for iter_fon in range(1, int(np.size(lines[0]))): 
            for iter_fon_2 in range(int(lines[2][iter_fon-1]), int(lines[1][iter_fon])): fon_x.append(iter_fon_2)
        for iter_fon_3 in range(lines[2][int(np.size(lines[0])-1)], int(np.max(graf_smooth[0]))): fon_x.append(iter_fon_3)

        for iter_fon_x in fon_x:
            j, = np.where(graf_smooth[0] == iter_fon_x)
            fon_y.append(graf_smooth[1][j[0]])

        popt, pcov = curve_fit(self.fon, fon_x, fon_y, (60, 0.001, -0.001, np.min(graf_smooth[1])))    
        fon_ea, fon_eb, fon_la, fon_lb = popt
        fon_var = [fon_ea, fon_eb, fon_la, fon_lb]

        #######fit

        size = np.size(lines)/3
        for i in range(0, int(size)):
            for_fit_x = np.arange(lines[1][i], lines[2][i], 1)
            index_b_for_y, = np.where(graf_smooth[0] == lines[1][i])
            index_e_for_y, = np.where(graf_smooth[0] == lines[2][i])
            for_fit_y = []  

            for iter_y in range(index_b_for_y[0], index_e_for_y[0]):
                for_fit_y.append(self.graf[1][iter_y] - self.fon(graf_smooth[0][iter_y], fon_ea, fon_eb, fon_la, fon_lb))
        

            index_line, = np.where(graf_smooth[0] == lines[0][i])
            A = graf_smooth[1][index_line[0]]
            mu = graf_smooth[0][index_line[0]]
            sig = lines[2][i] - lines[1][i]
            popt, pcov = curve_fit(self.gaussian, for_fit_x, for_fit_y, (A, mu, sig))    
            A, mu, sig = popt
            x = for_fit_x[:]
            y = self.gaussian(x, A, mu, sig)
            mas_fit = np.vstack((x, y))

            S = 0
            for s_i in mas_fit[1]:
                S+=s_i

            dict_vf = dict.fromkeys(['mu', 'sig', 'S'])
            dict_vf['mu'] = mu
            dict_vf['sig'] = sig
            dict_vf['S'] = S
            fit_var.append(dict_vf)
            #figa = plt.figure()
            #ax = figa.add_subplot(111)
            #ax.plot(for_fit_x, for_fit_y, linestyle='-', marker=',')
            #ax.plot(mas_fit[0], mas_fit[1], linestyle='-', marker=',')
            #ax.set_title(str("mu=" + str(int(mu)) +"   sig=" + str(int(sig)) + "   S=" + str(int(S))))
            #figa.show()
        ##############
            fit_ae.append([for_fit_x, for_fit_y, mas_fit[0], mas_fit[1]])

        lines = lines[0][:]

        return lines, fon_var, fit_ae, fit_var

    def fit_max_method(self, graf_smooth):
        peakX = []
        peakY = []
        peak = []
        flag = False
        iter = 0
        fonX = []
        fonY = []
        h = self.spin_var_maxh.value()
        p = self.spin_var_maxp.value()
        cut = 10
        last_i = 0
        for i in range(p, np.size(graf_smooth[0])-p):  
            N1 = graf_smooth[1][i-p]+h*np.sqrt( graf_smooth[1][i])
            N2 = graf_smooth[1][i+p]+h*np.sqrt( graf_smooth[1][i])
            if(graf_smooth[1][i] > N1 and graf_smooth[1][i] > N2):
                if(flag != True and i-last_i<2):
                    peakX.append(graf_smooth[0][i])
                    peakY.append(graf_smooth[1][i])
            else:
                if peakX:
                    delta = int((peakX[-1]-peakX[0]))
                    for x in range(int(peakX[0]-1), int(peakX[0]) - delta, -1): 
                        peakX.insert(0, x)
                        j, = np.where(graf_smooth[0] == int(x))
                        peakY.insert(0, float(graf_smooth[1][j]))
                    for x in range(int(peakX[-1]+1), int(peakX[-1])+delta):
                        peakX.append(x)
                        j, = np.where(graf_smooth[0] == int(x))
                        peakY.append(float(graf_smooth[1][j]))
                    peak.append([peakX, peakY]) 
                peakX = []
                peakY = []
            last_i = i
        return peak

    def change_graf_fit(self, value):
        self.figure_s.clear()
        self.ax_s = self.figure_s.add_subplot(111)
        self.ax_s.plot(self.fit_ae[value][0], self.fit_ae[value][1], linestyle='', marker='.', label = 'initial')
        self.ax_s.plot(self.fit_ae[value][2], self.fit_ae[value][3], linestyle='-', marker=',', label = 'fit')
        str_s = "mu=" + str(int(self.fit_var[value]['mu'])) + "  sig=" + str(int(self.fit_var[value]['sig'])) + "   S=" +str(int(self.fit_var[value]['S']))
        self.ax_s.set_title(str_s)
        self.ax_s.legend()
        self.canvas_s.draw()
            
        
        self.lineR.remove()
        self.lineG.remove()
        self.lineR = self.ax.scatter(self.lines, self.y_for_point, color = 'r')
        self.lineG = self.ax.scatter(self.lines[value], self.y_for_point[value], color = 'g')



        self.canvas.draw()

    def fit_for_max_method(self, graf_smooth, peak):
        lines = [(i[0][0] + (i[0][-1]-i[0][0])/2) for i in peak]
        fon_x = []
        fon_y = []
        for p in peak:
            fon_x.append(p[0][0])
            fon_x.append(p[0][-1])
            fon_y.append(p[1][0])
            fon_y.append(p[1][-1])
        popt, pcov = curve_fit(self.fon, fon_x, fon_y, (60, 0.001, -0.001, np.min(graf_smooth[1])))    
        fon_ea, fon_eb, fon_la, fon_lb = popt
        fon_var = [fon_ea, fon_eb, fon_la, fon_lb]

        fit_ae = []
        fit_var = []
        for i, p in enumerate(peak):
            index_line, = np.where(graf_smooth[0] == int(lines[i]))
            A = graf_smooth[1][index_line[0]]
            mu = graf_smooth[0][index_line[0]]
            sig = (p[0][-1] - p[0][0])/3
            for_fit_x = p[0]
            for_fit_y = [p[1][j] - self.fon(xxx, fon_var[0], fon_var[1], fon_var[2], fon_var[3]) for j, xxx in enumerate(p[0])]
            
            popt, pcov = curve_fit(self.gaussian, for_fit_x, for_fit_y, (A, mu, sig))    
            A, mu, sig = popt

            y = self.gaussian(for_fit_x[:], A, mu, sig)
            fit_ae.append([for_fit_x, for_fit_y, for_fit_x, y])

            S = 0
            for s_i in for_fit_x:
                S+=s_i
            dict_vf = dict.fromkeys(['mu', 'sig', 'S'])
            dict_vf['mu'] = mu
            dict_vf['sig'] = sig
            dict_vf['S'] = S
            fit_var.append(dict_vf)
       
        return lines, fon_var, fit_ae, fit_var

    def float_method(self):
        m = self.spin_var_float_m.value()
        k = (self.spin_var_float_k.value()//2)
        flags = []
        indices = []

        graf = self.graf

        for i in range(k, len(graf[0]) - k):
            xi1 = i - k
            xi2 = i + k
            
            x1 = graf[0][xi1]
            x2 = graf[0][xi2]
            y1 = graf[1][xi1]
            y2 = graf[1][xi2]

            def line(x):
                return (y2 - y1)*(x - x1)/(x2-x1) + y1

            S1 = 0
            S2 = 0
            for xi in range(xi1, xi2):
                yt = line(graf[0][xi])
                y = graf[1][xi]

                if (y >= yt): S1 += y
                else: S2 += y


            flags.append(True if (S1>(m*np.sqrt(S2))) else False)
            indices.append(i)

        index_peak = []
        last_index = 0
        last_f = False
        count = 0
        count_var = self.spin_var_float.value()
        for i, f in enumerate(flags):
            if(last_f == False and f == True):
                count = 0
                last_index = indices[i]
            elif(last_f == True and f == True):
                count +=1
            if(last_f == True and f == False and count > count_var):
                index_peak.append([last_index, indices[i]])

            last_f = f
        return index_peak
    
    def search(self, pressed):
        
        if(pressed):
            #################
            C = []
            if(self.combo_der.currentIndex() == 0): 
               
                try:
                    C = self.C
                    self.flag_fit_auto = True
                    self.lines, self.fon_var, self.fit_ae, self.fit_var = self.fit_auto_for_C(self.graf_smooth, C)
                except:
                    self.flag_fit_auto = False
                    print("fit_auto_for_C error 0 :")
                    return 0
            elif(self.combo_der.currentIndex() == 1): 
                
                try:
                    C = self.C_2
                    self.flag_fit_auto = True
                    self.lines, self.fon_var, self.fit_ae, self.fit_var = self.fit_auto_for_C(self.graf_smooth, C)
                except:
                    self.flag_fit_auto = False
                    print("fit_auto_for_C error 1")
                    return 0
            elif(self.combo_der.currentIndex() == 2): 
                
                
                try:
                    C = self.C_4
                    self.flag_fit_auto = True
                    self.lines, self.fon_var, self.fit_ae, self.fit_var = self.fit_auto_for_C(self.graf_smooth, C)
                except:
                    self.flag_fit_auto = False
                    print("fit_auto_for_C error 2")
                    return 0
            elif(self.combo_der.currentIndex() == 3): 
                
                try:
                    C = self.C_3
                    self.flag_fit_auto = True
                    self.lines, self.fon_var, self.fit_ae, self.fit_var = self.fit_auto_for_C(self.graf_smooth, C)
                except:
                    self.flag_fit_auto = False
                    print("fit_auto_for_C error 3 :")
                    return 0
            elif(self.combo_der.currentIndex() == 4): 
                self.flag_fit_auto = True
                try:
                    self.flag_fit_auto = True
                    self.lines, self.fon_var, self.fit_ae, self.fit_var = self.fit_for_max_method(self.graf_smooth, self.peak_of_max_method)
                except:
                    self.flag_fit_auto = False
                    print("fit_auto_for_C error 4 :")
                    return 0
            elif(self.combo_der.currentIndex() == 5): 
                self.flag_fit_auto = True
                try:
                    self.flag_fit_auto = True
                    self.lines, self.fon_var, self.fit_ae, self.fit_var = self.fit_for_float_method(self.C_5)
                except:
                    self.flag_fit_auto = False
                    print("fit_auto_for_C error 5 :")
                    return 0

            self.chek_fon.setStyleSheet("QCheckBox { color: black }")
            ################################


            self.lay_search = QVBoxLayout()

            self.figure_s = plt.figure()
            self.canvas_s = FigureCanvas(self.figure_s)
            self.toolbar_s = NavigationToolbar(self.canvas_s, self)
            self.lay_search.addWidget(self.toolbar_s)
            self.lay_search.addWidget(self.canvas_s)

            #######
            self.figure_s.clear()
            self.ax_s = self.figure_s.add_subplot(111)
            self.ax_s.plot(self.fit_ae[0][0], self.fit_ae[0][1], linestyle='', marker='.', label = 'initial')
            self.ax_s.plot(self.fit_ae[0][2], self.fit_ae[0][3], linestyle='-', marker=',', label = 'fit')
            str_s = "mu=" + str(int(self.fit_var[0]['mu'])) + "  sig=" + str(int(self.fit_var[0]['sig'])) + "   S=" +str(int(self.fit_var[0]['S']))
            self.ax_s.set_title(str_s)
            self.ax_s.legend()
            self.canvas_s.draw()
            
            self.y_for_point = []
            for i, v in enumerate(self.lines):
                j,  = np.where(self.graf_smooth[0] == v)
                self.y_for_point.append(self.graf_smooth[1][j])
            self.lineR = self.ax.scatter(self.lines, self.y_for_point, color = 'r')
            self.lineG = self.ax.scatter(self.lines[0], self.y_for_point[0], color = 'g')
            self.canvas.draw()
            #######

            

            self.sld = QSlider(Qt.Horizontal, self)
            self.sld.setRange(0, np.size(self.lines)-1)
            self.sld.setPageStep(1)
            self.sld.valueChanged[int].connect(self.change_graf_fit)

            self.lay_search.addWidget(self.sld)

            self.lvl1.addLayout(self.lay_search)

            w_c = self.canvas.size().width()
            w_s = self.size().width()
            h_s= self.size().height()
            self.resize((w_c+w_s), h_s)
        else: 
            if(self.flag_fit_auto != True): return 0
            self.lineR.remove()
            self.lineG.remove()

            w_c = self.canvas.size().width()
            w_s = self.size().width()
            h_s= self.size().height()
            self.resize((w_s-w_c), h_s)

            self.sld.deleteLater()
            self.toolbar_s.deleteLater()
            self.canvas_s.deleteLater()

    def plot(self):
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)

        self.val_smooth_C = self.spin_var_myder.value()


        if(self.chek_graph.isChecked()):
            self.ax.plot(self.graf[0], self.graf[1], linestyle='', marker=',', label = 'initial graph')
        if(self.chek_smooth.isChecked()):
            self.graf_smooth = self.smooth(self.graf, self.spin_smooth2.value())
            for i in range(1, self.spin_smooth.value()):
                self.graf_smooth = self.smooth(self.graf_smooth, self.spin_smooth2.value())
            self.ax.plot(self.graf_smooth[0], self.graf_smooth[1], linestyle='-', marker=',', label = 'smooth')
        if(self.chek_myder.isChecked()):
            self.C = self.my_derivative(self.graf_smooth, self.val_smooth_C)
            self.ax.plot(self.C[0][:], self.C[1][:], linestyle='-', marker=',', label = 'Bas der')
        if(self.chek_fon.isChecked()):
            x = self.graf_smooth[0]
            y = self.fon(x, self.fon_var[0], self.fon_var[1], self.fon_var[2], self.fon_var[3])
            self.ax.plot(x, y, linestyle='-', marker=',', label = 'background')
        if(self.chek_2der.isChecked()):
            self.C_2 = self.second_derivative(self.graf_smooth)
            self.ax.plot(self.C_2[0][:], self.C_2[1][:], linestyle='-', marker=',', label = '2nd der')
        # this is search by smoothing 
        if(self.chek_smooth_S.isChecked()): 
            self.C_3 = self.smooth(self.graf, self.spin_smooth2_S.value())
            for i in range(1, self.spin_smooth_S.value()):
                self.C_3 = self.smooth(self.C_3, self.spin_smooth2_S.value())
            self.C_3[1]  = self.graf_smooth[1] - self.C_3[1] 
            self.ax.plot(self.C_3[0], self.C_3[1], linestyle='-', marker=',', label = 'smoothing search')

        if(self.chek_max.isChecked()):
            self.peak_of_max_method = self.fit_max_method(self.graf_smooth)
            [self.ax.plot(self.peak_of_max_method[i][0], self.peak_of_max_method[i][1], linestyle='-', marker=',', label = str('peak' + str(i))) for i in range(0, 3)]

        if(self.chek_1der.isChecked()):
            self.C_4 = self.first_derivative(self.graf_smooth)
            self.ax.plot(self.C_4[0], self.C_4[1], linestyle='-', marker=',', label = 'first derivative')
        if(self.chek_float.isChecked()):
            self.C_5 = self.float_method()
            for i, v in enumerate(self.C_5):
                x = self.graf_smooth[0][v[0]:v[1]]
                y = self.graf_smooth[1][v[0]:v[1]]
                self.ax.plot(x, y, linestyle='-', marker=',', label = str('(float method) peak ' + str(i)))

       

        self.ax.legend()
        self.canvas.draw()

    def __init__(self, parent=None):
        super(Window, self).__init__(parent)
        
        self.graf = self.readingFile()
        self.readingFileS()
        
       
        self.val_iter_C = 5

#---------------------------------------------------------------------------------------------------------------------------
        self.chek_myder = QCheckBox('Basenko derivative', self)
        self.chek_2der = QCheckBox('2nd derivative', self)
        self.chek_1der = QCheckBox('1st derivative', self)
        self.chek_smooth = QCheckBox('smoothing', self)
        self.chek_graph = QCheckBox('initial graph', self)
        self.chek_fon = QCheckBox('background', self)
        self.chek_max = QCheckBox('max method', self)
        self.chek_float = QCheckBox('floating line', self)
        self.chek_fon.setStyleSheet("QCheckBox { color: red }")

        self.chek_myder.setChecked(False)
        self.chek_smooth.setChecked(True)
        self.chek_graph.setChecked(True)
        self.chek_fon.setChecked(False)
        self.chek_max.setChecked(False)
        self.chek_1der.setChecked(False)

        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)


        lvl22 = QVBoxLayout() 
        lvl22.addWidget(self.toolbar)
        lvl22.addWidget(self.canvas)
        lvl21 = QVBoxLayout()

        btn_plot = QPushButton("plot", self)
        btn_plot.clicked.connect(self.plot)


        btn_search = QPushButton("peaks search", self)
        btn_search.setCheckable(True)
        btn_search.clicked[bool].connect(self.search)


        self.spin_var_myder = QSpinBox(self)
        self.spin_var_myder.setMinimum(1)
        self.spin_var_myder.setMaximum(1000)
        self.spin_var_myder.setValue(self.config_plot['BasDer'])
        self.spin_var_myder.setToolTip("gain")
        spin_layout1 = QHBoxLayout()
        spin_layout1.addWidget(self.chek_myder)
        spin_layout1.addWidget(self.spin_var_myder)



        self.spin_var_2der = QSpinBox(self)
        self.spin_var_2der.setMinimum(0)
        self.spin_var_2der.setMaximum(1000)
        self.spin_var_2der.setValue(self.config_plot['2ndDer'])
        self.spin_var_2der.setToolTip("gain")
        spin_layout2 = QHBoxLayout()
        spin_layout2.addWidget(self.chek_2der)
        spin_layout2.addWidget(self.spin_var_2der)

        self.spin_var_1der = QSpinBox(self)
        self.spin_var_1der.setMinimum(0)
        self.spin_var_1der.setMaximum(1000)
        self.spin_var_1der.setValue(self.config_plot['1stDer'])
        self.spin_var_1der.setToolTip("gain")
        spin_layout3 = QHBoxLayout()
        spin_layout3.addWidget(self.chek_1der)
        spin_layout3.addWidget(self.spin_var_1der)
        
        self.spin_var_float = QSpinBox(self)
        self.spin_var_float.setMinimum(0)
        self.spin_var_float.setMaximum(1000)
        self.spin_var_float.setValue(self.config_plot['countf'])
        self.spin_var_float.setToolTip('there is a peak if the condition has been met count times in a row')
        self.spin_var_float_m = QDoubleSpinBox(self)
        self.spin_var_float_m.setMinimum(0)
        self.spin_var_float_m.setMaximum(1000)
        self.spin_var_float_m.setSingleStep(0.05)
        self.spin_var_float_m.setToolTip('dropout parameter q for S1 > q*sqrt(S2)')
        self.spin_var_float_m.setValue(self.config_plot['mf'])
        self.spin_var_float_k = QSpinBox(self)
        self.spin_var_float_k.setMinimum(0)
        self.spin_var_float_k.setMaximum(1000)
        self.spin_var_float_k.setToolTip('segment length')
        self.spin_var_float_k.setValue(self.config_plot['kf'])
        spin_layout5 = QHBoxLayout()
        spin_layout5.addWidget(self.chek_float)
        spin_layout5.addWidget(self.spin_var_float_m)
        spin_layout5.addWidget(self.spin_var_float_k)
        spin_layout5.addWidget(self.spin_var_float)


        self.spin_var_maxh = QDoubleSpinBox(self)
        self.spin_var_maxh.setMinimum(0)
        self.spin_var_maxh.setMaximum(1000)
        self.spin_var_maxp = QSpinBox(self)
        self.spin_var_maxp.setMinimum(0)
        self.spin_var_maxp.setMaximum(1000)
        spin_layout4 = QHBoxLayout()
        self.spin_var_maxh.setToolTip("h for N(i) > N(i+-p) +h*sqrt(N(i))")
        self.spin_var_maxp.setToolTip("p for N(i) > N(i+-p) +h*sqrt(N(i))")
        self.spin_var_maxh.setValue(2)
        self.spin_var_maxp.setValue(400)
        self.spin_var_maxh.setSingleStep(0.05)
        self.spin_var_maxp.setSingleStep(1)
        spin_layout4.addWidget(self.chek_max)
        spin_layout4.addWidget(self.spin_var_maxh)
        spin_layout4.addWidget(self.spin_var_maxp)


        smooth_layout = QHBoxLayout()
        smooth_layout.addWidget(self.chek_smooth)
        self.spin_smooth = QSpinBox(self)
        self.spin_smooth.setMinimum(1)
        self.spin_smooth.setValue(self.config_plot['smoothingA'])
        self.spin_smooth.setMaximum(9999)
        self.spin_smooth.setToolTip("number of smoothing procedures")
        smooth_layout.addWidget(self.spin_smooth)
        self.spin_smooth2 = QSpinBox(self)
        self.spin_smooth2.setMinimum(2)
        self.spin_smooth2.setValue(self.config_plot['smoothingB'])
        self.spin_smooth2.setMaximum(9999)
        self.spin_smooth2.setToolTip("number of dots to smooth")
        smooth_layout.addWidget(self.spin_smooth2)




        lvl21.addWidget(btn_plot)
        
        lvl21.addLayout(smooth_layout)
        lvl21.addWidget(self.chek_graph)
        lvl21.addWidget(self.chek_fon)
        lvl21.addWidget(btn_search)

        lay_m_ = QHBoxLayout()
        self.qspin_search_m = QDoubleSpinBox()
        self.qspin_search_m.setMinimum(0)
        self.qspin_search_m.setMaximum(900)
        self.qspin_search_m.setSingleStep(0.05)
        self.qspin_search_m.setValue(3)
        self.qspin_search_m.setToolTip("if dev > m => near peak")
        label_m = QLabel("peak determining factor:")
        lay_m_.addWidget(label_m)
        self.qspin_search_m.setValue(self.config_plot['pdf'])
        lay_m_.addWidget(self.qspin_search_m)
        lvl21.addLayout(lay_m_)

        lay_k_ = QHBoxLayout()
        self.qspin_search_k = QSpinBox()
        self.qspin_search_k.setMinimum(1)
        self.qspin_search_k.setMaximum(1000)
        self.qspin_search_k.setValue(5)
        self.qspin_search_k.setToolTip("mean of C (averaging)")
        label_k = QLabel("smoothing micro fluctuations:")
        lay_k_.addWidget(label_k)
        self.qspin_search_k.setValue(self.config_plot['aad'])
        lay_k_.addWidget(self.qspin_search_k)
        lvl21.addLayout(lay_k_)

        self.combo_der = QComboBox(self)
        self.combo_der.addItems(["Basenko der", "2nd der", "1st der", "smoothing method", "max method"])

        lvl21.addWidget(self.combo_der)

        self.chek_smooth_S = QCheckBox('smoothing search', self)
        smooth_layout_S = QHBoxLayout()
        smooth_layout_S.addWidget(self.chek_smooth_S)
        self.spin_smooth_S = QSpinBox(self)
        self.spin_smooth_S.setMinimum(1)
        self.spin_smooth_S.setValue(self.config_plot['smsearchA'])
        self.spin_smooth_S.setMaximum(9999)
        self.spin_smooth_S.setToolTip("number of smoothing procedures")
        self.spin_smooth_S.setValue(int(np.size(self.graf[0])/4))
        smooth_layout_S.addWidget(self.spin_smooth_S)
        self.spin_smooth2_S = QSpinBox(self)
        self.spin_smooth2_S.setMinimum(2)
        self.spin_smooth2_S.setValue(self.config_plot['smsearchB'])
        self.spin_smooth2_S.setMaximum(9999)
        self.spin_smooth2_S.setToolTip("number of dots to smooth")
        self.spin_smooth2_S.setValue(int(np.size(self.graf[0])/4))
        smooth_layout_S.addWidget(self.spin_smooth2_S)
        lvl21.addLayout(smooth_layout_S)

        lvl21.addLayout(spin_layout1)
        lvl21.addLayout(spin_layout2)
        lvl21.addLayout(spin_layout3)
        lvl21.addLayout(spin_layout4)
        lvl21.addLayout(spin_layout5)
        

        self.spin_smooth2_S.setValue(self.config_plot['smsearchB'])
        self.spin_smooth_S.setValue(self.config_plot['smsearchA'])
        self.spin_smooth2.setValue(self.config_plot['smoothingB'])

        self.lvl1 = QHBoxLayout()
        self.lvl1.addLayout(lvl21)
        self.lvl1.addLayout(lvl22)
        self.setLayout(self.lvl1)


if __name__ == '__main__':
    app = QApplication(sys.argv)

    main = Window()
    main.show()
    sys.exit(app.exec_())