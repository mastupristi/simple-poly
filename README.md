# polinoimial interpolator

this simple project aims to give a polynomial interpolation of a given dataset.

It can be usefull to find the polynomial that fit better a temerature sensor curve (given by points)

## requirements

This project uses json schema, numpy >= 1.20.1 and matplotlib.

on ubuntu 20.10 yoiu can install by typing:

```bash
$ sudo apt install python3-jsonschema python3-matplotlib python3-pip
$ pip3 install -I numpy
```
here we use `pip3` to install numpy because in ubuntu 20.10 repositories there is only version 1.18.4 of numpy.
The parameter `-I` to pip3 is used to ignore any numpy already installed in the system through `apt`.

## examples

```bash
$ python3 interpolinome.py -a -p examples/data1.json
```
