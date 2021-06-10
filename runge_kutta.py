#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from attractors import Attractors

class RK(Attractors):

    def __init__(self , initial_r, attractor, **kwargs):
        super(RK, self).__init__(attractor, **kwargs)
        self.r = initial_r
        self.X = []
        self.Y = []
        self.Z = []

    def Euler(self, a, b, N):
        h = (b-a)/N
        time_scale = np.arange(a,b,h)
        attractor_func = getattr(RK, self.attractor)
        
        for _ in time_scale:
            self.X.append(self.r[0])
            self.Y.append(self.r[1])
            self.Z.append(self.r[2])

            k1 = h*attractor_func(self, self.r)
            self.r += k1
    
    def RK2(self, a, b, N, method='heun'):
        h = (b-a)/N
        time_scale = np.arange(a,b,h)
        attractor_func = getattr(RK, self.attractor)

        def heun():
            rt = self.r
            k1 = h*attractor_func(self, self.r)

            self.r = self.r+ k1
            k2 = h*attractor_func(self, self.r)
            self.r = rt

            self.r += (k1+k2)/2
        
        def imp_poly():
            rt = self.r
            k1 = h*attractor_func(self, self.r)

            self.r = self.r+ k1/2
            k2 = h*attractor_func(self, self.r)
            self.r = rt

            self.r += k2

        def ralston():
            rt = self.r
            k1 = h*attractor_func(self, self.r)

            self.r = self.r+ 3*k1/4
            k2 = h*attractor_func(self, self.r)
            self.r = rt

            self.r += (k1+2*k2)/3

        for _ in time_scale:
            self.X.append(self.r[0])
            self.Y.append(self.r[1])
            self.Z.append(self.r[2])
            eval(method)()

    def RK3(self, a, b, N):
        h = (b-a)/N
        time_scale = np.arange(a,b,h)
        attractor_func = getattr(RK, self.attractor)
    
        for _ in time_scale:
            self.X.append(self.r[0])
            self.Y.append(self.r[1])
            self.Z.append(self.r[2])
            
            rt = self.r
            k1 = h*attractor_func(self, self.r)

            self.r = self.r + k1/2
            k2 = h*attractor_func(self, self.r)
            self.r = rt

            self.r = self.r - k1 + 2*k2
            k3 = h*attractor_func(self, self.r)
            self.r = rt

            self.r += (k1+4*k2+k3)/6
    
    def RK4(self, a, b, N):
        h = (b-a)/N
        time_scale = np.arange(a,b,h)
        attractor_func = getattr(RK, self.attractor)

        for _ in time_scale:
            self.X.append(self.r[0])
            self.Y.append(self.r[1])
            self.Z.append(self.r[2])
            
            rt = self.r
            k1 = h*attractor_func(self, self.r)
            
            self.r = self.r+ k1/2
            k2 = h*attractor_func(self, self.r)
            self.r = rt

            self.r = self.r + k2/2
            k3 = h*attractor_func(self, self.r)
            self.r = rt

            self.r = self.r+k3
            k4=h*attractor_func(self, self.r)
            self.r = rt

            self.r += (k1+2*k2+2*k3+k4)/6
    
    def RK5(self, a, b, N):
        h = (b-a)/N
        time_scale = np.arange(a,b,h)
        attractor_func = getattr(RK, self.attractor)

        for _ in time_scale:
            self.X.append(self.r[0])
            self.Y.append(self.r[1])
            self.Z.append(self.r[2])
            
            rt = self.r
            k1 = h*attractor_func(self, self.r)
            
            self.r = self.r+ k1/4
            k2 = h*attractor_func(self, self.r)
            self.r = rt

            self.r = self.r + k2/8 + k1/8
            k3 = h*attractor_func(self, self.r)
            self.r = rt

            self.r = self.r+k3 - k2/2 + k3
            k4=h*attractor_func(self, self.r)
            self.r = rt

            self.r = self.r - 3*k1/16 + 9*k4/16
            k5=h*attractor_func(self, self.r)
            self.r = rt

            self.r = self.r - 3*k1/7 + 2*k2/7 + 12*k3/7 - 12*k4/7 + 8*k5/7
            k6=h*attractor_func(self, self.r)
            self.r = rt

            self.r += (7*k1+32*k3+12*k4+32*k5+7*k6)/90