# This space will be used to test the online version of the code

import pytorchNN
import torch
import numpy as np
from data_nn import load_data
import time

[Data_1, Time_1, Data_08, Time_08] = load_data()

AXD08 = Data_08[0]

# Model Architecture
input_size = 1
hidden_sizes = [64, 64, 64]
output_size = 1
dropout_rate = 0.3
## 

model = pytorchNN.TorchNN(
    input_size=input_size,
    hidden_sizes=hidden_sizes,
    output_size=output_size,
    dropout_rate=dropout_rate,
    use_batch_norm=False
)
checkpoint = torch.load('best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  # Set to evaluation mode

## Prediction Data
x_pred_list = Data_1[0,-200:]
pred_list = np.zeros(len(x_pred_list))
TIMEOUT = 2 

# Record the start time
start_time = time.time() 

pred = model(x_pred_list)


# Online Portion Starts Here

""" 
Con is just a placeholder variable, all we need to do is find an actual variable in the sensor
that can indicate to the system that it should be running. There are also variables to cut the program once it loops
but these can be adjusted I just need to know more about the data aquistion.

Things to replace:
    
    con: Replace with a start indicator from the sensor
    the entire for loop: This would just be replaced with the while loop
    Timeout: Feature can be adjusted to include longer periods
    time.sleep: this function is likely not necessary in the actual program but is a good
                representation of what online would look like
    buffer: Allows for the prediction function to lag slightly behind the real time
"""

con = True 
buffer = False

try:
    while con == True:
        if buffer == True:
            time.sleep(1)
        for i, x in enumerate(x_pred_list):
            curr_pred = x
            curr_pred = np.array([curr_pred])
            time.sleep(0.05)
            pred = pytorchNN.predict(model, curr_pred)
            pred_list[i] = pred[0,]
            print(pred_list[i])
            if i == len(x_pred_list):
                con = False
        if time.time() - start_time > TIMEOUT:
            break
        
except KeyboardInterrupt:
    print("\nArray updates stopped by user.")
    
    
"""
1/28/26 Additions

if old != new
    check time_since_gait switching to make sure its not too short
    
Not Complete yet

Set time with Miguel to learn sensor and exoskeleton setup

"""

""" 
2/04/26 Additions

Add into Ros2
put it onto github

""" 
        
    