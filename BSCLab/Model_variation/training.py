# Binary Classifier Training File

#%%
import importlib
from datavar import load_data
import graphingvar
import labelingvar
import dnnvar
import Gait_LSTM
import Gait_Transformer
import pandas as pd

importlib.reload(Gait_LSTM)
importlib.reload(Gait_Transformer)
importlib.reload(dnnvar)


[Data_1, Time_1, Data_08, Time_08, _, _] = load_data()


## Hyper Parameters for Neural Network

batch_size = 32
epochs = 120
learning_rate = 1e-3
input_size = 3
hidden_sizes = [64, 64, 64]
output_size = 1
dropout_rate = 0.15
patience = 100

##

"Vector Hyper-parameters are for hyper parameter tuning"

# ## Vector Hyper-Parameters
# batch_sizeV = [64, 32, 128]
# epochsV = [10,20,40]
# learning_rateV = [1e-2,1e-3, 1e-4]
# neuronsV = [64, 128, 256]

# ##


## Data Manipulation Section Here

AXd1 = Data_1[0:3,:]
print(f"Shape of AXd1: {AXd1.shape}")

GStrike = labelingvar.Label_Max(AXd1[0])
labels, Gait_Labels= labelingvar.Gait_Label(GStrike)


""" 
This section divides the data into 4 seperate arrays. 2 for training and 2 for
validation. The model will further cut up the training data so it can perform its
own validation but I also wanted to have some data to check the predictions

"""
model_x_percent = 0.80
len_train = int(model_x_percent*len(AXd1[0]))
X = AXd1[:, :len_train]
X = labelingvar.tran(X)
X_val = AXd1[0:3, len_train:len(AXd1[0])-200]
X_val = labelingvar.tran(X_val)
y = labels[:len_train].reshape(1,-1)
y_val = labels[len_train:len(labels)-200].reshape(1,-1)
y = y.T
y_val = y_val.T

print(X.shape)
print(y.shape)

"This code prepares the data to be used in a Pytorch model"
# Prepare data
train_loader_NN, val_loader_NN = dnnvar.prepare_data(X, y, batch_size= batch_size)
train_loader_LSTM_50, val_loader_LSTM_50 = Gait_LSTM.prepare_data(X, y, batch_size= batch_size, window = 50)
train_loader_LSTM_120, val_loader_LSTM_120 = Gait_LSTM.prepare_data(X, y, batch_size= batch_size, window = 120)
train_loader_Transformer_50, val_loader_Transformer_50 = Gait_Transformer.prepare_data(X, y, batch_size= batch_size, window = 50)
train_loader_Transformer_120, val_loader_Transformer_120 = Gait_Transformer.prepare_data(X, y, batch_size= batch_size, window = 120)

training_sets = [train_loader_NN, train_loader_LSTM_50, train_loader_LSTM_120, train_loader_Transformer_50, train_loader_Transformer_120]
validation_sets = [val_loader_NN, val_loader_LSTM_50, val_loader_LSTM_120, val_loader_Transformer_50, val_loader_Transformer_120]
## Model Creation


model_NN = dnnvar.BinaryClassifier(
    input_size= input_size,
    hidden_size= hidden_sizes,
    dropout_rate= dropout_rate,
    use_batch_norm=False
)

model_LSTM = Gait_LSTM.GaitLSTM(
    input_size= input_size,
    hidden_size= hidden_sizes[0],
    num_layers=2,
    output_size= output_size
)

model_Transformer = Gait_Transformer.GaitTransformer(
    input_dim = input_size,
)


##

## Train Model
history_NN = dnnvar.train_model(
    model=model_NN,
    train_loader=train_loader_NN,
    val_loader= val_loader_NN,
    num_epochs= epochs,
    learning_rate= learning_rate,
    patience= patience
)
history_LSTM_50 = Gait_LSTM.train_model(
    model=model_LSTM,
    train_loader=train_loader_LSTM_50,
    val_loader= val_loader_LSTM_50,
    num_epochs= epochs,
    learning_rate= learning_rate,
    patience= patience,
    save_path= './LSTM_50_model.pth'
)
history_Transformer_50 = Gait_Transformer.train_model(
    model=model_Transformer,
    train_loader=train_loader_Transformer_50,
    val_loader= val_loader_Transformer_50,
    num_epochs= epochs,
    learning_rate= learning_rate,
    patience= patience,
    save_path= './Transformer_50_model.pth'
)
history_LSTM_120 = Gait_LSTM.train_model(
    model=model_LSTM,
    train_loader=train_loader_LSTM_120,
    val_loader= val_loader_LSTM_120,
    num_epochs= epochs,
    learning_rate= learning_rate,
    patience= patience,
    save_path= './LSTM_120_model.pth'
)
history_Transformer_120 = Gait_Transformer.train_model(
    model=model_Transformer,
    train_loader=train_loader_Transformer_120,
    val_loader= val_loader_Transformer_120,
    num_epochs= epochs,
    learning_rate= learning_rate,
    patience= patience,
    save_path= './Transformer_120_model.pth'
)

"Graphs the loss over epochs for both the training and validation data"

# Graphing Loss
graphingvar.plot_figures(history_NN)
graphingvar.plot_figures(history_LSTM_50)
graphingvar.plot_figures(history_Transformer_50)
graphingvar.plot_figures(history_LSTM_120)
graphingvar.plot_figures(history_Transformer_120)

# Calculating Metrics
metrics_NN = dnnvar.calc_metrics(model_NN, val_loader_NN)
metrics_LSTM_50 = dnnvar.calc_metrics(model_LSTM, val_loader_LSTM_50)
metrics_Transformer_50 = dnnvar.calc_metrics(model_Transformer, val_loader_Transformer_50)
metrics_LSTM_120 = dnnvar.calc_metrics(model_LSTM, val_loader_LSTM_120)
metrics_Transformer_120 = dnnvar.calc_metrics(model_Transformer, val_loader_Transformer_120)

## Predictions

predictions_NN = dnnvar.predict(model_NN, X_val)
print(f"\nSample predictions: {predictions_NN.flatten()}")
graphingvar.plot_predictions(predictions_NN, y_val, X_val)
CF_M_NN = graphingvar.Confusion_Matrix(y_val, predictions_NN)

predictions_LSTM_50 = Gait_LSTM.predict(model_LSTM, X_val)
print(f"\nSample predictions: {predictions_LSTM_50.flatten()}")
graphingvar.plot_predictions(predictions_LSTM_50, y_val, X_val)
CF_M_LSTM_50 = graphingvar.Confusion_Matrix(y_val, predictions_LSTM_50)

predictions_Transformer_50 = Gait_Transformer.predict(model_Transformer, X_val)
print(f"\nSample predictions: {predictions_Transformer_50.flatten()}")
graphingvar.plot_predictions(predictions_Transformer_50, y_val, X_val)
CF_M_Transformer_50 = graphingvar.Confusion_Matrix(y_val, predictions_Transformer_50)

predictions_LSTM_120 = Gait_LSTM.predict(model_LSTM, X_val)
print(f"\nSample predictions: {predictions_LSTM_120.flatten()}")
graphingvar.plot_predictions(predictions_LSTM_120, y_val, X_val)
CF_M_LSTM_120 = graphingvar.Confusion_Matrix(y_val, predictions_LSTM_120)

predictions_Transformer_120 = Gait_Transformer.predict(model_Transformer, X_val)
print(f"\nSample predictions: {predictions_Transformer_120.flatten()}")
graphingvar.plot_predictions(predictions_Transformer_120, y_val, X_val)
CF_M_Transformer_120 = graphingvar.Confusion_Matrix(y_val, predictions_Transformer_120)

##






# %%
