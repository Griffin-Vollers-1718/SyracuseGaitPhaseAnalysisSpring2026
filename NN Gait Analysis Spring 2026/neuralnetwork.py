# This file is where the neural network class will live

import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

def split_data(features, labels):
    [features_train, features_test, labels_train, labels_test] = train_test_split(features, labels, test_size = 0.2, random_state = 42)

    features_train = pd.DataFrame(features_train)
    features_test = pd.DataFrame(features_test)
    labels_train = pd.DataFrame(labels_train)
    labels_test = pd.DataFrame(labels_test)
    
    return features_train, features_test, labels_train, labels_test


class Agent(object):
    def __init__(self, features, labels,
                learning_rate, neurons,
                feat_t, label_t, epochs,
                batch_size):
        self.features = features
        self.labels = labels
        self.learning_rate = learning_rate
        self.neurons = neurons
        self.f_t = feat_t
        self.l_t = label_t
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.model = self.create_model()
        self.history, self.hist_NN, self.model_pred = self.create_history()
        self.plot_figures()
        #self.save_model()
        #self.model, self.history = self.load_model()
        
        
    
    def create_model(self):
        num_features = self.features.shape[1]
        inp = Input(shape=(num_features,))
        dense1 = Dense(self.neurons, activation = 'relu')(inp)
        drop1 = Dropout(0.3)(dense1)
        dense2 = Dense(self.neurons, activation = 'relu')(drop1)
        drop2 = Dropout(0.15)(dense2)
        out = Dense(1, activation = 'sigmoid')(drop2)
        opt = Adam(self.learning_rate)
        
        model = Model(inputs = [inp], outputs = [out])
        model.compile(loss = 'binary_crossentropy', metrics = ['accuracy'], optimizer = opt)
        
        return model
        
    def create_history(self):
        history = self.model.fit(
            self.features, self.labels,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(self.f_t, self.l_t),
            verbose=1
        )

        hist_NN = pd.DataFrame(history.history)
        model_pred = self.model.predict(self.f_t)
        model_mse, model_mae = self.model.evaluate(self.f_t, self.l_t, verbose=0)
        print(f"Test MSE: {model_mse:.4f}, Test MAE: {model_mae:.4f}")
        
        return history, hist_NN, model_pred
    
    def plot_figures(self):
        #Plotting
        
        # --- Plot loss and MAE ---
        plt.figure(figsize=(12, 5))

        # Plot Loss
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['loss'], label='Train Loss')
        if 'val_loss' in self.history.history:
            plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.legend()

        # Plot MAE
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['accuracy'], label='Train Accuracy')
        if 'val_accuracy' in self.history.history:
            plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        # plt.tight_layout()
        # plot_filename = f'Model_BatchSize_{batch_size}_Epochs_{epochs}_LearningRate_{learning_rate}_Neurons_{neurons}_.png'
        # plt.savefig(f'NN_Graphs/{plot_filename}')
        # plt.show()
        
    
    def save_model(self, model_name):
        self.model.save(model_name)
        hist_csv = 'NN_Training_History.csv'
        with open(hist_csv, mode='w') as f:
            self.hist_NN.to_csv(f)
    
    def load_old_model(self, name_model):
        model = load_model(name_model)
        history = pd.read_csv('NN_Training_History.csv')
        
        return model, history
        