#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 22:38:07 2024

@author: martkr
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.ticker import AutoLocator
from ufuncs import Environment



#%% Test

env = Environment(initial_speed=6, speed_up=1.01, perc_paddle=0.2, 
                            shrink=0.1, max_speed_paddle=8, steps_per_frame = 20, sight = 0.5)

m = np.zeros(40)
for i in range(40):
    status = np.load(f"./Cache/124/status_ep{i}.pkl", allow_pickle=True)
    m[i]=status['values'][0][2][-11:-1].mean()
plt.plot(m, color = [0,0,0])

m_pre=np.zeros(40)
for i in range(40):
    m_pre[i] = status['values'][0][2][1:10].mean()

m_pro = np.zeros(40)                      
for i in range(40): 
    m_pro[i] = m[i]/m_pre[i] -1
        
xx = 0 
fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
for i in range(40):
    ax.vlines(i,0, m_pro[i], color=[0,0,0])  # The vertical stems.

# format x-axis with 4-month intervals
ax.xaxis.set_major_locator(AutoLocator)
plt.setp(ax.get_xticklabels(), ha="right")

# remove y-axis and spines
#ax.yaxis.set_visible(False)
ax.spines[["left", "top", "right"]].set_visible(False)

ax.margins(y=0.1)
plt.show()


#%% First trial
env1 = np.load("./Cache/120/environments.npz", allow_pickle=True)
env1 = env1['arr_0'].item()
m1 ={}

test_scores1 = np.load("./Cache/120/test_scores_array.npy")
test_values1 = {}
test_per1 = {}
count = 0

for i in range(41):
    status = np.load(f"./Cache/120/status_ep{i}.pkl", allow_pickle=True)
    if i == 0: 
        for j in range(6):
            m1[f"{status['env_idx'][j]}"] = np.array(((i, status['values'][j][2][-11:-1].mean())))
            test_values1[f"{status['env_idx'][j]}"] = np.array(((i,test_scores1[count][1])))
            test_per1[f"{status['env_idx'][j]}"] = np.array(((i,test_scores1[count][1]/test_scores1[count][0] -1)))
            count = count + 1
    else:
        for j in range(8):
            if f"{status['env_idx'][j]}" not in m1.keys():
                m1[f"{status['env_idx'][j]}"] = np.array(((i, status['values'][j][2][-11:-1].mean())))
                test_values1[f"{status['env_idx'][j]}"] = np.array(((i,test_scores1[count][1])))
                test_per1[f"{status['env_idx'][j]}"] = np.array(((i,test_scores1[count][1]/test_scores1[count][0] -1)))
                count = count + 1
            else: 
                m1[f"{status['env_idx'][j]}"] = np.vstack(((m1[f"{status['env_idx'][j]}"], ((i, status['values'][j][2][-11:-1].mean())))))
                test_values1[f"{status['env_idx'][j]}"] = np.vstack(((test_values1[f"{status['env_idx'][j]}"], ((i, test_scores1[count][1])))))
                test_per1[f"{status['env_idx'][j]}"] = np.vstack(((test_per1[f"{status['env_idx'][j]}"], ((i, test_scores1[count][1]/test_scores1[count][0] -1)))))
                count = count+1
c= {}
for i in env1: 
    for j in range(len(env1[i])):
        if f"{env1[i][j]._idx}" not in c.keys():
            c[f"{env1[i][j]._idx}"] = env1[i][j]._color

for i in m1:
    if len(m1[i].shape) > 1:
        plt.plot(m1[i][:,0], m1[i][:,1], color = c[i])
plt.figure()
for i in m1:
    if len(test_values1[i].shape) > 1:
        plt.plot(test_values1[i][:,0], test_values1[i][:,1], color = c[i])

m1_pre={}
for i in range(41):
    status = np.load(f"./Cache/120/status_ep{i}.pkl", allow_pickle=True)
    if i == 0: 
        for j in range(6):
            m1_pre[f"{status['env_idx'][j]}"] = np.array(status['values'][j][2][1:10].mean())
    else:
        for j in range(8):
            if f"{status['env_idx'][j]}" not in m1_pre.keys():
                m1_pre[f"{status['env_idx'][j]}"] = np.array(status['values'][j][2][1:10].mean())
            else: 
                m1_pre[f"{status['env_idx'][j]}"] = np.vstack(((m1_pre[f"{status['env_idx'][j]}"], status['values'][j][2][1:10].mean())))

m1_pro = {}                       
for i in m1_pre: 
    if i not in m1_pro.keys():
        if len(m1[i].shape) > 1: 
            m1_pro[i] = m1[i][:,1]/m1_pre[i].flatten()
        
xx = 0 
fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
for i in m1_pro:
    ax.vlines(m1[i][:,0]+xx,0 , m1_pro[i]-1, color=c[i])  # The vertical stems.
    xx = xx + 1/8
    if xx == 1:
        xx == 0

# format x-axis with 4-month intervals
ax.xaxis.set_major_locator(AutoLocator)
plt.setp(ax.get_xticklabels(), ha="right")

# remove y-axis and spines
#ax.yaxis.set_visible(False)
ax.spines[["left", "top", "right"]].set_visible(False)

ax.margins(y=0.1)
plt.show()


#%% Second Trial

env2 = np.load("./Cache/125/environments.npz", allow_pickle = True)
env2 = env2['arr_0'].item()

col2 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

#Datenmatritzen erstellen 

m2 = np.zeros((8, 40))
m2_pre = np.zeros((8, 40))
m2_pro = np.zeros((8, 40))
test_scores2 = np.load("./Cache/125/test_scores_array.npy")
test_values2 = np.zeros((8,40))
test_per = np.zeros((8,40))
#colors2 = {}
count = 0

for i in range(40):
    status = np.load(f"./Cache/125/status_ep{i}.pkl", allow_pickle=True)
    for j in range(8):
        m2[j][i]=status['values'][j][2][-11:-1].mean()
        m2_pre[j][i]=status['values'][j][2][:10].mean()
        m2_pro[j][i]= m2[j][i]/m2_pre[j][i] -1
        # if f"{status['env_idx'][j]}" not in test_values2.keys():
        #     test_values2[f"{status['env_idx'][j]}"] = ((i, test_scores2[count][1]))
        #     colors2[f"{status['env_idx'][j]}"] = col2[j]
        # else: 
        #     test_values2[f"{status['env_idx'][j]}"] = np.vstack(((test_values2[f"{status['env_idx'][j]}"], ((i, test_scores2[count][1])))))
        test_values2[j][i] = test_scores2[count][1]
        test_per[j][i] = test_scores2[count][1]/test_scores2[count][0] -1
        count = count + 1

#Plot

for i in range(40):
    if (i+1) %4 ==0:
        x = np.array(range(i-3, i+1))
        for k in range(8):
            plt.plot(x, m2[k][i-3:i+1], color = col2[k])
    
#Test Plot 
plt.figure()
# for i in test_values2:
#     plt.plot(test_values2[i][:,0], test_values2[i][:,1], color = colors2[i])
for i in range(40):
    if (i+1) %4 ==0:
        x = np.array(range(i-3, i+1))
        for k in range(8):
            plt.plot(x, test_values2[k][i-3:i+1], color = col2[k])

#Bars Plot
y = np.arange(40)
fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
for i in range(8):
    ax.vlines((y+i/8), 0, m2_pro[i], color=col2[i])  
ax.xaxis.set_major_locator(AutoLocator)
plt.setp(ax.get_xticklabels(), ha="right")
ax.spines[["left", "top", "right"]].set_visible(False)
ax.margins(y=0.1)
plt.show()

#BarsPlot Test 
y = np.arange(40)
fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
for i in range(8):
    ax.vlines((y+i/8), 0, test_per[i], color=col2[i])  
ax.xaxis.set_major_locator(AutoLocator)
plt.setp(ax.get_xticklabels(), ha="right")
ax.spines[["left", "top", "right"]].set_visible(False)
ax.margins(y=0.1)
plt.show()


