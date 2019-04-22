#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 14:43:40 2018

@author: mromiario
"""

from pylab import genfromtxt
from random import randint
import numpy as np


###############M Romi Ario - 1301154311########################
 #Ambil data txt//Data training
data = genfromtxt("/Users/mromiario/Downloads/Data_Tugas_3_RL.txt")
data_q = np.array(data)



w, h = 10, 10;

#Matrix Reward Raw/ Definisi matrix 10x10 dari soal
Matrix = [[0 for x in range(w)] for y in range(h)] 
state = [[-1 for x in range(w)] for y in range(h)] 

for j in range(10) :
    for i in range(10) :    
        Matrix[j][i] = data_q[9-j][i]

count_state = 0
for j in range(10) :
    for i in range(10) :    
        state[j][i] = count_state
        count_state +=1

#Matrix R. Generate ke matriks 100x4
w_r = 4
h_r = 100
Matrix_r = [[0 for x in range(w_r)] for y in range(h_r)] #Matriks reward; state x action
r_state = [[0 for x in range(w_r)] for y in range(h_r)] 

#Matrix R / Action to North
count = 0
for j in range(1,10) :
    for i in range(10) :
        Matrix_r[count][0] = Matrix[j][i]
        r_state[count][0] = state[j][i]
        count+=1
#Matrix R/Action to South

count_s = 10
for j in range(9) :
    for i in range(10) :
        Matrix_r[count_s][1] = Matrix[j][i]
        r_state[count_s][1] = state[j][i]
        count_s+=1
#Matrix R/Action to East

count_e = -1
for j in range(10) :
    count_e+=1
    for i in range(1,10) :
        Matrix_r[count_e][2] = Matrix[j][i]
        r_state[count_e][2] = state[j][i]
        count_e+=1

#Matrix R/Action to West

count_w = 0
for j  in range(10) :
    count_w+=1
    for i in range(9) :
        Matrix_r[count_w][3] = Matrix[j][i]
        r_state[count_w][3] = state[j][i]
        count_w += 1


#MATRIX Q

Q = [[0 for x in range(w_r)] for y in range(h_r)]     

#Menyimpan step dan reward
sequence_total = []
reward_all = []

gamma = 0.8
episodes = 1000

for i in range(episodes) :
    reward =0
    sequence = []
#Initial state
    initial_state = 0
    while(initial_state != 99) :
        
        #Memilih action yang mungkin 
        possible = randint(0,3)
        while (Matrix_r[initial_state][possible] == 0) :
            possible = randint(0,3)
        #Menuju next state
        next_state = r_state[initial_state][possible]
        #Replace nilai Q next state
        max = np.max(Q[next_state])

        
        if (Matrix_r[next_state][0] != 0) :
            Q0 = Matrix_r[next_state][0] + gamma*max
            Q[next_state][0] = Q0
     
        if (Matrix_r[next_state][1] != 0) :
            Q1 = Matrix_r[next_state][1] + gamma*max
            Q[next_state][1] = Q1  

        if (Matrix_r[next_state][2] != 0) :
            Q2 = Matrix_r[next_state][2] + gamma*max
            Q[next_state][2] = Q2  

        if (Matrix_r[next_state][3] != 0) :
            Q3 = Matrix_r[next_state][3] + gamma*max
            Q[next_state][3] = Q3 

            
        #Replace nilai state / current state    
        Q[initial_state][possible] = Matrix_r[initial_state][possible]+gamma*(np.max(Q[next_state]))
        
        reward = reward + Matrix_r[initial_state][possible]
        
        sequence.append(initial_state)
        #initial state dirubah menjadi next_state
        initial_state = next_state
        
    reward_all.append(reward)
    sequence_total.append(sequence)

print("Reward maksimal : ",np.max(reward_all)) #Nilai reward yang paling baik
print("Sequence : ",sequence_total[np.argmax(reward_all)]) #Sequence state sampai goals
print(len(sequence_total[np.argmax(reward_all)])) #Banyaknya step

