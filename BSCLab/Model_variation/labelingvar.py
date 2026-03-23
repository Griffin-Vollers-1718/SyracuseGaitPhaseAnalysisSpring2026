# This module is for determining the ground truth labels using heuristics
# Define Labeling function

import numpy as np
import pandas as pd


## Labeling Parameters

check = 20   # How many values will the function check surrounding a point to see if there is a local value greater than it
buffer = 20  # How many values the function will skip to gaurentee adequate checking
bottom = 4   # bottom threshold for heel strike
top = 11     # top threshold for heel strike

##


def Label_Max(data):
    # This function finds the maximum values of the angular velocity
    # In a good step the toe off will rotate more than 11 "value" and the heel strike would typically
    # be aroud 9. It then looks around these points to see if there are any points around it that 
    # are a higher value than it. This helps determine the true value of the heel and toe strike
    
    max_points = np.zeros(len(data))
    mid_points = np.zeros(len(data))
    for i in range(len(data)):
        if i <= buffer:
            continue
        if data[i] > data[i-1] and data[i] > data[i+1]:
            if data[i] > top:
                max_points[i] = data[i]
            if data[i] <= top and data[i] > bottom:
                mid_points[i] = data[i]
    TO_points = max_points
    HS_points = mid_points
    for g in range(len(max_points)):
        if max_points[g] != 0:
            for q in range(check):
                if max_points[g-q] > max_points[g] or max_points[g+q] > max_points[g]:
                    TO_points[g] = 0
        if mid_points[g] != 0:
            for h in range(check):
                if mid_points[g-h] > mid_points[g] or mid_points[g+h] > mid_points[g]:
                    HS_points[g] = 0
    Gait_Strikes = TO_points + HS_points
    # Toe_Off = TO_points
    # Heel_Strike = HS_points
    GStrike = Gait_Strikes
    # If looking for check on labels, use these 
    # Steps1 = len(Toe_Off[Toe_Off != 0])
    # Steps2 = len(Heel_Strike[Heel_Strike != 0])
    # Steps3 = len(GStrike[GStrike != 0])
    # HS10 = len(Heel_Strike[Heel_Strike > 10.7]) # This was to determine the maximum value of the Heel Strike
    return GStrike


#This Function takes in the Toe Off Points and Heel Strike Points and spits out a binary array
#Stance = 0
#Swing = 1

def Gait_Label(data):
    Gait_Labels = np.zeros(len(data))
    tick = 0
    for i in range(len(data)):
        if data[i] != 0:
            if data[i] > 11:
                Gait_Labels[i] = 0
                for h in range(tick):
                    Gait_Labels[i-h] = 0
                tick = 0
            else:
                Gait_Labels[i] = 1
                for h in range(tick+1):
                    Gait_Labels[i-h] = 1
                tick = 0
        else:
            tick = tick + 1
    labels = Gait_Labels
    Gait_Label = pd.DataFrame(labels)
    return labels, Gait_Label

def Basic_Regrssion(labels, gait):
    USwitch = np.zeros(len(labels))
    FSwitch = np.zeros(len(labels))
    StanceP = np.zeros(len(labels))
    SwingP = np.zeros(len(labels))
    indices = np.where(gait != 0)
    indices = indices[0]
    count = -1
    indice = 0
    for i, label in enumerate(labels):
        if gait[i] != 0:
            indice += 1
            count += 1
            if indice == len(indices):
                break
            FSwitch[i] = count
            count = 0
            USwitch[i] = 0
        elif gait[i] == 0 and indice == 0:
            count += 1
            FSwitch[i] = count
            USwitch[i] = indices[indice]- count
        else:
            count += 1
            FSwitch[i] = count
            USwitch[i] = indices[indice] - indices[indice-1] - count
        if indice == 0:
            if label == 0:
                StanceP[i] = (FSwitch[i]/indices[indice])*100
                SwingP[i] = (USwitch[i]/indices[indice])*100
            else:
                StanceP[i] = (USwitch[i]/indices[indice])*100
                SwingP[i] = (FSwitch[i]/indices[indice])*100
        elif gait[i] != 0:
            if label == 0:
                StanceP[i] = 100.00
                SwingP[i] = 0.00
            else:
                StanceP[i] = 0.00
                SwingP[i] = 100.00
        else:
            if label == 0:
                StanceP[i] = (FSwitch[i]/(indices[indice]- indices[indice - 1]))*100
                SwingP[i] = (USwitch[i]/(indices[indice]- indices[indice - 1]))*100
            else:
                StanceP[i] = (USwitch[i]/(indices[indice]- indices[indice - 1]))*100
                SwingP[i] = (FSwitch[i]/(indices[indice]- indices[indice - 1]))*100
    Percentage = np.array([StanceP, SwingP])
    #Percentage = StanceP
    Percentage = Percentage.T
    return Percentage

def tran(x):
    X = x.T
    return X
        
