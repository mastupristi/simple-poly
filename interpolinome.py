#!/usr/bin/env python3

# sudo apt install python3-numpy python3-scipy python3-matplotlib
# https://matplotlib.org/

import sys
#import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument("jsonin", help="input json file")

args = parser.parse_args()
if not args.jsonin:
    sys.exit(1)
    
fin = open(args.jsonin, 'r')
fin_data = fin.read()
fin.close()


data = json.loads(fin_data)

degree = data['degree']
xdata = data['data']['x']
ydata = data['data']['y']
weight = data['data']['w']

X = np.ones((len(xdata),1))
for column in range(degree):
    X =  np.column_stack((X[:,0]*xdata, X))

#print(X)

P = np.diag(weight)

beta = np.linalg.inv(X.transpose() @ P @ X) @ X.transpose() @ P @ ydata

print(beta)

N = len(ydata)

ymean = sum(np.sqrt(P) @ ydata)/N

errorV = ydata - X @ beta

SQR = errorV.transpose() @ P @ errorV

bluttro = np.sqrt(P) @ ydata - np.ones(len(xdata)) * ymean

SQT = bluttro.transpose() @ bluttro

R_2 = 1 - SQR/SQT

R_2_C = 1 - (SQR / (N - (degree+1)))/ ( SQT / (N-1) )

print("R^2 ", R_2)
print("R^2 corrected ", R_2_C)

RangeX = max(xdata) - min(xdata)
meanX = (min(xdata) + max(xdata))/2
ExtRange = RangeX*1.2
NewMin = meanX - ExtRange/2
NewMaX = meanX + ExtRange/2

newx = np.linspace(NewMin, NewMaX, 100)

X1 = np.ones((len(newx),1))
for column in range(degree):
    X1 =  np.column_stack((X1[:,0]*newx, X1))
    
newy = X1 @ beta

plt.plot(xdata, ydata, label='data', marker='o')
plt.plot(newx, newy, label='poli')
plt.show()
