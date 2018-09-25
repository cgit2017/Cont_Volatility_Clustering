#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 19:22:43 2018

@author: charlie
"""

import collections
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# Probability of updating threshold
# Should be 0 < s << 1
s = 0.1
auto_corr = []

class Agent():
    def __init__(self, market):
        self.phi = 0
        # Not sure if this is a good theta???
        self.theta = random.gauss(5, 5)
        self.market = market
        self.returns = []
        
    def update_theta(self):
        # Random uniformly distributed integer [0, 1]
        u = random.random()
        if abs(np.log(u)) < s * abs(self.market.r) + abs(np.log(u)) >= s* self.theta:
        #if u < s:
            self.theta = abs(self.market.r)
            
    def update_demand(self, epsilon):
        if epsilon > self.theta:
            self.phi = 1
        if epsilon < (-1 * self.theta):
            self.phi = -1
        if abs(epsilon) <= self.theta:
            self.phi = 0
            
    def update_returns(self):
        pass
        # Need to figure out how to calculate each agent's returns
        
            
class Market():
    def __init__(self, num_agents):
        self.num_agents = num_agents
        self.traders = [Agent(self) for a in range(self.num_agents)]
        self.price = 100
        self.epsilon = 0
        self.price_hist = [random.gauss(self.price, 1), random.gauss(self.price, 1)]
        self.r_hist = []
        self.r_real_hist = []
        self.price_hist2 = collections.deque(self.price_hist, maxlen=2)
        # Magnitude of return |r|
        self.r = 0
        # Return with sign
        self.r_real = 0
        self.theta_hist = []
        
    def update_theta_hist(self):
        self.theta_hist.append(self.traders[0].theta)
        
    def gen_forecast(self):
        # Need to figure out why Cont includes D = 0.001 and
        # why the parameter in the model is D**2
        # D = 0.001
        self.epsilon = random.gauss(0, 4)
        
    def update_price(self):
        zt = sum([trader.phi for trader in self.traders]) / len(self.traders)
        self.price = self.price + zt
        self.price_hist.append(self.price)
        self.price_hist2.append(self.price)
        
        
    def calc_return(self):
        # Calculate absolute return |rt| = |ln(pt/pt-1)| 
        self.r = np.log((self.price_hist2[-1] / self.price_hist2[-2]))
        self.r_hist.append(self.r)
        # Calculate the return relative to previous price
        self.r_real = (self.price_hist2[-1] - self.price_hist2[-2])
        self.r_real_hist.append(self.r_real)
        
        
if __name__ == "__main__":
    m = Market(1500)
    for t in tqdm(range(10000)):
        # Generate the forecast
        m.gen_forecast()
        for a in m.traders:
            a.update_theta()
            a.update_demand(m.epsilon)
        m.update_price()
        m.calc_return()
        m.update_theta_hist()
        
    plt.plot(m.price_hist)
    rh = pd.Series(m.r_hist)
    ph = pd.Series(m.price_hist)
    rrh = pd.Series(m.r_real_hist)
    
    
    rhc = rh.dropna()
    rhc = rhc.iloc[1000:]
    rha = rhc.abs()
    
    #pd.plotting.autocorrelation_plot(rha)
    #pd.plotting.autocorrelation_plot(rhc)
    #pd.plotting.autocorrelation_plot(rhc.iloc[1000:1600])
    #pd.plotting.autocorrelation_plot(rha.iloc[1000:1050])
    
    #plt.hist(rhc, bins=100)
    
    #plt.hist(rhc, bins=50, log=True, range=(-0.1, 0.1))
    
            