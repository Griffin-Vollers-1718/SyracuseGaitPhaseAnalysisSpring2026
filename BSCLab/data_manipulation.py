# This is where all the data will live
import scipy.io as sio
import numpy as np
import pandas as pd


def load_miguel_data():
    S1data = sio.loadmat('./data/S1-1ms.mat',squeeze_me = False)
    S08data = sio.loadmat('./data/S1-08ms.mat',squeeze_me = False)
    # Extracts the data based on the header info from the loadmat function 
    AngularX1 = S1data['footAngularVelocityX']
    AngularY1 = S1data['footAngularVelocityY']
    AngularZ1 = S1data['footAngularVelocityZ']
    LinearX1 = S1data['footLinearAccelerationX']
    LinearY1 = S1data['footLinearAccelerationY']
    LinearZ1 = S1data['footLinearAccelerationZ']
    # It doesn't look like the Linear Acceleration collects anything
    # Seperates the data into data which is a force value in N and time measured in s
    AXd1 = AngularX1['data'] 
    AXd1 = np.reshape(AXd1[0,0],(121595,))
    AXt1 = AngularX1['time']
    AXt1 = np.reshape(AXt1[0,0],(121595,))
    #AX1 = np.vstack((AXd1, AXt1))
    
    AYd1 = AngularY1['data']
    AYd1 = np.reshape(AYd1[0,0],(121595,))
    AYt1 = AngularY1['time']
    AYt1 = np.reshape(AYt1[0,0],(121595,))
    
    AZd1 = AngularZ1['data']
    AZd1 = np.reshape(AZd1[0,0],(121595,))
    AZt1 = AngularZ1['time']
    AZt1 = np.reshape(AZt1[0,0],(121595,))
    
    LXd1 = LinearX1['data']
    LXd1 = np.reshape(LXd1[0,0],(121595,))
    LXt1 = LinearX1['time']
    LXt1 = np.reshape(LXt1[0,0],(121595,))
    
    LYd1 = LinearY1['data']
    LYd1 = np.reshape(LYd1[0,0],(121595,))
    LYt1 = LinearY1['time']
    LYt1 = np.reshape(LYt1[0,0],(121595,))
    
    LZd1 = LinearZ1['data']
    LZd1 = np.reshape(LZd1[0,0],(121595,))
    LZt1 = LinearZ1['time']
    LZt1 = np.reshape(LZt1[0,0],(121595,))
    
    
    # 0.8 m/s data
    AngularX08 = S08data['footAngularVelocityX']
    AngularY08 = S08data['footAngularVelocityY']
    AngularZ08 = S08data['footAngularVelocityZ']
    LinearX08 = S08data['footLinearAccelerationX']
    LinearY08 = S08data['footLinearAccelerationY']
    LinearZ08 = S08data['footLinearAccelerationZ']
    # It doesn't look like the Linear Acceleration collects anything
    # Seperates the data into data which is a force value in N and time measured in s
    AXd08 = AngularX08['data'] 
    AXd08 = np.reshape(AXd08[0,0],(124143,))
    AXt08 = AngularX08['time']
    AXt08= np.reshape(AXt08[0,0],(124143,))
    #AX08 = np.vstack((AXd08, AXt08))
    
    AYd08 = AngularY08['data']
    AYd08 = np.reshape(AYd08[0,0],(124143,))
    AYt08 = AngularY08['time']
    AYt08 = np.reshape(AYt08[0,0],(124143,))
    
    AZd08 = AngularZ08['data']
    AZd08 = np.reshape(AZd08[0,0],(124143,))
    AZt08 = AngularZ08['time']
    AZt08 = np.reshape(AZt08[0,0],(124143,))
    
    LXd08 = LinearX08['data']
    LXd08 = np.reshape(LXd08[0,0],(124143,))
    LXt08 = LinearX08['time']
    LXt08 = np.reshape(LXt08[0,0],(124143,))
    
    LYd08 = LinearY08['data']
    LYd08 = np.reshape(LYd08[0,0],(124143,))
    LYt08 = LinearY08['time']
    LYt08 = np.reshape(LYt08[0,0],(124143,))
    
    LZd08 = LinearZ08['data']
    LZd08 = np.reshape(LZd08[0,0],(124143,))
    LZt08 = LinearZ08['time']
    LZt08 = np.reshape(LZt08[0,0],(124143,))
    
    Data_08 = np.vstack((AXd08,AYd08, AZd08, LXd08, LYd08, LZd08))
    Time_08 = np.vstack((AXt08, AYt08, AZt08, LXt08, LYt08, LZt08))
    Data_1 = np.vstack((AXd1,AYd1, AZd1, LXd1, LYd1, LZd1))
    Time_1 = np.vstack((AXt1, AYt1, AZt1, LXt1, LYt1, LZt1))
    
    return Data_1, Time_1, Data_08, Time_08


def sheppard_data(file_path, feat_cols, label_cols):
    data = pd.read_csv(file_path)
    curr_data = data[[feat_cols]]
    label_data = data[[label_cols]]
    time_data = data[['loop_time']]
    return curr_data, label_data, time_data

 
data = sheppard_data('BSCLab/data/S02/20230425_1543_S02_T01_RIGHT.csv')
""" 
Need to figure out how I want to download the code sequentially
Also need to convert to numpy arrays
"""