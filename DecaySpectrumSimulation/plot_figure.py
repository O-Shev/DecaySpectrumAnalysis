import matplotlib.pyplot as plt
import numpy as np
import random
import operator

class plot_ax():
    inputf = open('input.txt', 'r', encoding="utf8")

    line_Energy_characteristic = []
    back_function_dict = dict.fromkeys(['la', 'lb', 'ea', 'eb'])
    range_Energy = dict.fromkeys(['Emax', 'numCh', 'bias'])
    config_plot = dict.fromkeys(['background', 'broad', 'statVar', 'dimension', 'window'])
    config_plot['background'] = 'True'
    config_plot['broad'] = 'FWHM'
    config_plot['statVar'] = 'G'
    config_plot['dimension'] = 'C'


    flag_check = True

    def find_nearest_above(self, my_array, target):
        diff = my_array - target
        mask = np.ma.less_equal(diff, 0)
        if np.all(mask):
            return None
        masked_diff = np.ma.masked_array(diff, mask)
        return masked_diff.argmin()

    def __init__(self):
        for linestr in self.inputf:
            s = []
            tempstr = linestr.split(' ')
            for iters in tempstr:
                tempstrlist = iters.split('\t')
                for tempiters in tempstrlist:
                    s.append(tempiters)
            while(s.count('') != 0):
                s.remove('')

            if(s[0][0] != '#'):
                if(s[0] == 'line'):
                    self.line_Energy_characteristic.append({'E': float(s[1]), 'I': float(s[2]), 'FWHM': float(s[3])})
                if(s[0] == 'B.liner'):
                    self.back_function_dict['la'] = float(s[1])
                    self.back_function_dict['lb'] = float(s[2])
                if(s[0] == 'B.exp'):
                    self.back_function_dict['ea'] = float(s[1])
                    self.back_function_dict['eb'] = float(s[2])
                if(s[0] == 'rE'):
                    self.range_Energy['Emax'] = float(s[1])
                    self.range_Energy['numCh'] = float(s[2])
                    self.range_Energy['bias'] = float(s[3])
        
        self.line_Energy_characteristic.sort(key=operator.itemgetter('E'))

        self.grafX = np.linspace(1, int(self.range_Energy['numCh']), int(self.range_Energy['numCh']))
        self.grafY = np.zeros(int(self.range_Energy['numCh']))
        self.graf1 = np.vstack((self.grafX, self.grafY))

    def update_graf(self):
        self.writef = open('spectr.txt', 'w')
        for iterE in self.line_Energy_characteristic:
                    iter = self.find_nearest_above(self.graf1[0], (iterE['E']*(self.range_Energy['numCh'])/(self.range_Energy['Emax'])))
                    self.graf1[1][iter] += iterE['I']

        self.graf2 = np.array(self.graf1)
        if(self.config_plot['background'] == 'True'):
                     #y=a*exp(-b*x)
                     self.graf2[1] += (self.back_function_dict['ea']*np.exp(-1*self.back_function_dict['eb']*self.graf2[0]))
                     #y=a*x+b
                     self.graf2[1] += (self.back_function_dict['la']*self.graf2[0] + self.back_function_dict['lb'])
                     ###
        elif(self.config_plot['background'] == 'False'):
             None
        else:
             flag_check = False
             print("Error: background value in config file is undefined")
        

        self.graf3 = np.array(self.graf2)
        ###���� � ������р
        if(self.config_plot['broad'] == 'FWHM'):
            for self.iterFWHM in self.line_Energy_characteristic:
                sig = self.iterFWHM['FWHM']/2.355*self.range_Energy['numCh']/self.range_Energy['Emax']
                gaus = (np.exp(-1*np.power(self.graf3[0]-(self.iterFWHM['E']*self.range_Energy['numCh']/self.range_Energy['Emax']), 2) / (2 * np.power(sig, 2))))
                self.graf3[1] += self.iterFWHM['I']*gaus
        
       
        ###���� ��� �����
        elif(self.config_plot['broad'] == 'None'):
            for self.iterE in self.line_Energy_characteristic:
                iter = self.find_nearest_above(self.graf3[0], (self.iterE['E']*self.range_Energy['numCh']/self.range_Energy['Emax']))
                self.graf3[1][iter] += self.iterE['I']
        else:
            flag_check = False
            print("Error: broad value in config file is undefined")

         #statistical variance########################

        self.graf4 = np.array(self.graf3)
        flag_check_temp2 = True
        if(self.config_plot['statVar'] == 'P'):
            A, B = np.shape(graf4)
            i = 0
            for i in range(B):
                lamda = self.graf4[1][i]
                if(lamda >= 0):
                    self.graf4[1][i] = np.random.poisson (lam = lamda, size = None)
                else:
                    flag_check_temp2 = False
        elif(self.config_plot['statVar'] == 'G'):
            A, B = np.shape(self.graf4)
            i = 0
            for i in range(B):
                if(self.graf4[1][i] > 10):
                    mu = self.graf4[1][i]
                    sigma = np.sqrt(self.graf4[1][i])
                    self.graf4[1][i] = random.gauss(mu, sigma)
                else:
                    lamda = self.graf4[1][i]
                    if(lamda >= 0):
                        self.graf4[1][i] = np.random.poisson (lam = lamda, size = None)
                    else:
                        flag_check_temp2 = False
        elif(self.config_plot['statVar'] == 'None'):
            None
        else:
            flag_check = False
            print("Error: statVar value in config file is undefined")


        if(self.config_plot['dimension'] == 'C'):
            self.writef.write('Chanel' + '\t' + '\t' +'intensity' + '\n')
        elif(self.config_plot['dimension'] == 'E'):
            self.writef.write('Energy' + '\t' +'intensity' + '\n')
        A, B = np.shape(self.graf4)
        i = 0
        for i in range(B):
            z = str(self.graf4[1][i])
            s = z.split('.')
            zx = str(self.graf4[0][i])
            sx = zx.split('.')
            self.writef.write(str(sx[0]) + '\t' + str(s[0]) + '\n')