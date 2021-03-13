#!/usr/bin/env python3

# sudo apt install python3-numpy python3-scipy python3-matplotlib
# https://matplotlib.org/

import sys
#import os
import argparse
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import jsonschema
from jsonschema import validate

# Describe what kind of json you expect.
versionSchema = {
    "type": "object",
    "properties": {
        "data": {
            "type": "object",
            "properties": {
                "x": {
                    "type": "array",
                    "items": {
                        "type": "number"
                    },
                    "minItems": 1,
                    "uniqueItems": True
                },
                "y": {
                    "type": "array",
                    "items": {
                        "type": "number"
                    },
                    "minItems": 1,
                },
                "w": {
                    "type": "array",
                    "items": {
                        "type": "number"
                    },
                    "minItems": 1,
                }
            },
            "required": ["x", "y"],
            "additionalProperties": False
        },
        "degree": {"type": "integer"},
        "bit": {"type": "integer"},
    },
    "required": ["degree", "bit"],
    "additionalProperties": False
}

def checkData(data):
    try:
        validate(instance=data, schema=versionSchema)
    except jsonschema.exceptions.ValidationError as err:
        print("Invalid data file. Nothing Done", file=sys.stderr)
        sys.exit(1)

    degree = data['degree']
    bit = data['bit']
    xdata = data['data']['x']
    ydata = data['data']['y']
    if len(xdata) != len(ydata):
        print("x and y must have the same length", file=sys.stderr)
        sys.exit(1)

    if 'w' in data['data']:
        weight = data['data']['w']
        if len(xdata) != len(weight):
            print("w must have the same length of x and y", file=sys.stderr)
            sys.exit(1)
    else:
        weight = [1] * len(xdata)

    return xdata, ydata, degree, bit, weight

def computeInterpolation(xdata, ydata, degree, weight):
    X = np.ones((len(xdata),1))
    for column in range(degree):
        X =  np.column_stack((X[:,0]*xdata, X))

    # normalize weight
    weight /= np.sum(weight)
    P = np.diag(weight)

    beta = np.linalg.inv(X.transpose() @ P @ X) @ X.transpose() @ P @ ydata

    # we want to know how well fit the curve
    # For this reason the theory tells us to calculate the correct R^2 parameter: the closer it is to 1 the better the
    # curve fits.
    N = len(ydata)
    ymean = sum(np.sqrt(P) @ ydata)/N
    errorV = ydata - X @ beta
    SQR = errorV.transpose() @ P @ errorV
    h = np.sqrt(P) @ ydata - np.ones(len(xdata)) * ymean
    SQT = h.transpose() @ h
    R_2 = 1 - (SQR / (N - (degree+1)))/ ( SQT / (N-1) )
    print("R^2 ", R_2)

    # we also want to have the magnitude of the error. For this we can calculate the standard deviation of the error
    sigma = np.sqrt(SQR / (len(xdata) - (degree+1)))
    print("error standar deviation sigma ", sigma)

    return beta

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--plot", help="plot curve", action='store_true', default=False)
    parser.add_argument("-a", "--alt", help="alternative coefficient print", action='store_true', default=False)
    parser.add_argument("jsonin", help="input json file")

    args = parser.parse_args()
    if not args.jsonin:
        sys.exit(1)

    plot = args.plot
    alternativePrints = args.alt
    with open(args.jsonin, 'r') as f:
        data = json.load(f)

    xdata, ydata, degree, bit, weight = checkData(data)

    beta = computeInterpolation(xdata, ydata, degree, weight)

    p = np.polynomial.Polynomial(np.flip(beta))
    if alternativePrints:
        deg = p.degree()
        for c in beta[:-1]:
            print("%.15g x**%d" % (c, deg))
            deg -=1
        print("%.15g" % beta[-1])
    else:
        print(p)

    if plot:
        newx = np.linspace(0, (1<<bit) - 1)

        newy = p(newx)
        error = ydata - p(np.array(xdata))

        # plot figures
        dpi = 96

        fig1, ax1 = plt.subplots(1, 1, figsize=(1440/dpi, 900/dpi), dpi=dpi)
        ax1.grid()

        plt.plot(xdata, ydata, label='data', marker='o', figure=fig1)
        plt.plot(newx, newy, label='poli', figure=fig1)

        fig2, ax2 = plt.subplots(1, 1, figsize=(1440/dpi, 900/dpi), dpi=dpi)

        ax2.grid()
        plt.plot(xdata, error, label='error', marker='o', figure=fig2)
        plt.show()

if __name__ == '__main__':
    main()
    sys.exit(0)
