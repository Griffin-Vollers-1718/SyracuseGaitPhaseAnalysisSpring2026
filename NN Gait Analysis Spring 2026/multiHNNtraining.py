# Multi Headed NN Training File

from data_nn import load_data
import graphing
import labeling
import MultiHeadedNN

[Data_1, Time_1, Data_08, Time_08] = load_data()


## Hyper Parameters for Neural Network

batch_size = 32
epochs = 15
learning_rate = 1e-3
input_size = 3
hidden_sizes = [64, 128]
output_size = 2
dropout_rate = 0.50
patience = 10
batch_norm = False
loss_weights = (1.0, 1.0)

##

"Vector Hyper-parameters are for hyper parameter tuning"

# ## Vector Hyper-Parameters
# batch_sizeV = [64, 32, 128]
# epochsV = [10,20,40]
# learning_rateV = [1e-2,1e-3, 1e-4]
# neuronsV = [64, 128, 256]

# ##


## Data Manipulation Section Here

"AXd1 is the angular velocity of all three planes of the foot"
AXd1 = Data_1[0:3,:]


"""
Finds the ground strikes and toe offs to then label the data as either stance or 
swing. This is then use to determine the percentage of stance and swing by 
taking the total number of time steps till the next major event (GS or TO) 
and dividing it by the number of steps that have occured towards that event
(Linear Interpolation)

"""
GStrike = labeling.Label_Max(AXd1[0])
labels, Gait_Labels= labeling.Gait_Label(GStrike)
SPerc = labeling.Basic_Regrssion(labels, GStrike)/100


""" 
This section divides the data into 4 seperate arrays. 2 for training and 2 for
validation. The model will further cut up the training data so it can perform its
own validation but I also wanted to have some data to check the predictions

"""
model_x_percent = 0.9
len_train = int(model_x_percent*len(AXd1[0]))
X = AXd1[:, :len_train]
X = labeling.tran(X)
X_val = AXd1[0:3, len_train:len(AXd1[0])-200]
X_val = labeling.tran(X_val)
y = labels[:len_train]
y_perc = SPerc[:len_train,  :]
y_val = labels[len_train:len(labels)-200]


"This code prepares the data to be used in a Pytorch model"
# Prepare data

train_loader, val_loader = MultiHeadedNN.prepare_data(X, y, y_perc, batch_size= batch_size)

## Model Creation


model = MultiHeadedNN.MultiHeadNN(
    input_size= input_size,
    hidden_size= hidden_sizes,
    dropout_rate= dropout_rate,
    use_batch_norm= batch_norm
    )

print(f"Model architecture:\n{model}\n")
##

## Train Model

history = MultiHeadedNN.train_model(
    model=model,
    train_loader=train_loader,
    val_loader= val_loader,
    num_epochs= epochs,
    learning_rate= learning_rate,
    patience= patience,
    loss_weights = loss_weights
)
"Graphs the loss over epochs for both the training and validation data"
graphing.plot_figures_MH(history)

##

## Predictions

[binary_predictions, percentage_predictions] = MultiHeadedNN.predict(model, X_val)


"Plots where the model breaks down and the Confusion Matrix associated"

graphing.plot_predictions(binary_predictions, y_val, X_val)

CF_M = graphing.Confusion_Matrix(y_val, binary_predictions)

##


## The following spaced is used to describe the next steps of the project

""" 

Recorded 1/20/2026
Notes next steps: Add in the Regrssor
                  Make real time 
                  
Additional things to add:
                  Add in more speeds to the model
                  
Recorded 1/28/2026
Next Steps: Read some papers on the regression modeling
            Add in more speeds
            Discuss with Miguel about the real time
            
                      

"""

""" 
To make real time, need to create a while loop that will continuously 
run the models prediction function

"""
