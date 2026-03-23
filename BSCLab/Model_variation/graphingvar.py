# This will be where the graphing modules are stored

import matplotlib.pyplot as plt
import numpy as np


# For verification of data filtering using high pass filter, refer to Neural Network Gait Analysis\GaitAnalysis

def True_Labels_Sag1(AXd, AXt):
    # This figure displays the Angular Velocity of a couple steps to help show how
    # we are deciding to find the true labels for stance and swing

    fig, ax = plt.subplots()
    ax.plot(AXt[850:1600], AXd[850:1600])
    ax.set_title("Angular Velocity of the Sagittal Plane of the Ankle")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Radians per Second [rad/s]")
    ax.set_ylim([-8.5, 15])
    ax.set_xlim([AXt[850],AXt[1600]])
    point_xt, point_yt = AXt[1145], AXd[1145]
    point_xh, point_yh = AXt[1327], AXd[1327]
    plt.annotate("Toe Off", # The text to display
                 xy=(point_xt, point_yt), # The point to annotate
                 xytext=(point_xt - 0.15, point_yt - 0.25), # Position of the text
                 arrowprops=dict(facecolor='black', shrink=0.05), # Arrow properties
                 horizontalalignment='right',
                 verticalalignment='center')
    plt.annotate("Heel Strike", # The text to display
                 xy=(point_xh, point_yh), # The point to annotate
                 xytext=(point_xh + 0.15, point_yh - 0.25), # Position of the text
                 arrowprops=dict(facecolor='black', shrink=0.05), # Arrow properties
                 horizontalalignment='left',
                 verticalalignment='center')
    plt.axvspan(AXt[850],AXt[1145], color = "green", alpha = 0.2)
    plt.axvspan(AXt[1146],AXt[1327], color = "red", alpha = 0.2)
    plt.axvspan(AXt[1327],AXt[1600], color = "green", alpha = 0.2)
    # The next lines place the Stance and Swing text on the screen
    plt.annotate("Stance", xy=(AXt[998], 8.5), weight ='bold',
                 horizontalalignment ='center',
                 verticalalignment = 'center')
    plt.annotate("Swing", xy = (AXt[1236], 8.5), weight = 'bold',
                 horizontalalignment ='center',
                 verticalalignment = 'center')
    plt.annotate("Stance", xy=(AXt[1464], 8.5), weight = 'bold',
                 horizontalalignment ='center',
                 verticalalignment = 'center')


def True_Labels_Sag1_Test(AXd, AXt):
    fig, bx = plt.subplots()
    bx.plot(AXt[40000:42000],AXd[40000:42000])
    bx.set_title("Sagittal Plane for 4 Seconds")
    bx.set_xlabel("Time [s]")
    bx.set_ylabel("Radians per Second [rad/s]")
    bx.set_xlim([AXt[40000],AXt[42000]])


def Check_Max_Plots(AXd,AXt,Toe_Off, Heel_Strike):
    start = 0
    Length = round(len(AXd)/2000) - 1
    o = 100
    for i in range(Length):
        end = start + 2000
        s = start/400
        e = end/400
        plt.figure(o)
        plt.plot(AXt[start:end],AXd[start:end])
        plt.scatter(AXt[start:end], Toe_Off[start:end], color = 'green')
        plt.scatter(AXt[start:end], Heel_Strike[start:end])
        plt.title(f'Toe and Heel Strike from time {s} to {e}')
        plt.xlabel("Time [s]")
        plt.ylabel("Radians per Second [rad/s]")
        plot_filename = f'Check_Plots_from_{s}_to{e}.png'
        plt.savefig(f'NN_Video/{plot_filename}')
        start = end
        o = o + 1
        

def Check_Binary_Plots(labels, AXt, AXd):
    start = 0
    Length = round(len(labels)/2000) - 1
    #Length = 4
    o = 100
    Stance = np.zeros(2000)
    Swing = np.zeros(2000)
    for i in range(Length):
        end = start + 2000
        s = start/400
        e = end/400
        if labels[start] == 0:
            Stance[0] = AXt[start]
            if start == 0:
                Stance[0] = 1e-7
        else:
            Swing[0] = AXt[start]
        if labels[end] == 0:
            Stance[-1] = AXt[end]
        else:
            Swing[-1] = AXt[end]
        for g in range(end-start):
            if g == 1:
                continue
            if labels[g + start] == 0 and labels[(g-1) + start] == 1:
                Stance[g] = AXt[g + start]
                Swing[g-1] = AXt[(g-1) + start]
            elif labels[g + start] == 1 and labels[(g-1) + start] == 0:
                Swing[g] = AXt[g + start]
                Stance[g-1] = AXt[(g-1) + start]
        count_swing = len(Swing[Swing != 0])
        count_stance = len(Stance[Stance != 0])
        Swing = Swing[Swing != 0]
        Stance = Stance[Stance != 0]
        plt.figure(o)
        plt.plot(AXt[start:end],AXd[start:end])
        q = 0
        q2 = 0
        for y in range(0,count_swing-1,2):
            plt.axvspan(Swing[q], Swing[q+1], color = 'red', alpha = 0.2)
            q = q + 2
        for p in range(0,count_stance-1,2):
            plt.axvspan(Stance[q2], Stance[q2+1], color = 'green', alpha = 0.2)
            q2 = q2 + 2
        plt.title(f'Gait Phase from time {s} to {e}')
        start = end
        Stance = np.zeros(2000)
        Swing = np.zeros(2000)
        o = o + 1
    return count_swing, count_stance, Swing, Stance
     

def Confusion_Matrix(labels_test, model_pred):
    # labels_test = labels_test.to_numpy()
    # model_pred_round = np.round(model_pred)
    Confusion_Matrix = np.zeros(len(model_pred))
    for i in range(len(model_pred)):
        if model_pred[i] == 1:
            if labels_test[i] == 1:
                Confusion_Matrix[i] = 1
            else:
                Confusion_Matrix[i] = 2
        if model_pred[i] == 0:
            if labels_test[i] == 1:
                Confusion_Matrix[i] = 3
            else:
                Confusion_Matrix[i] = 4
    C1 = len(Confusion_Matrix[Confusion_Matrix == 1])
    C2 = len(Confusion_Matrix[Confusion_Matrix == 2])
    C3 = len(Confusion_Matrix[Confusion_Matrix == 3])
    C4 = len(Confusion_Matrix[Confusion_Matrix == 4])
    
    P1 = (C1/len(model_pred))*100
    P2 = (C2/len(model_pred))*100
    P3 = (C3/len(model_pred))*100
    P4 = (C4/len(model_pred))*100
    
    C = P1 + P4
    I = P2 + P3
    
    
    
    print(  "            Predicted\n"
            "          +-------+-------+\n"
            "          |Stance | Swing |\n"
            "+---------+-------+-------+\n"
           f"| Stance  | {C4} | {C2}    |\n"
            "+---------+-------+-------+\n"
           f"|  Swing  | {C3}   | {C1}  |\n"
            "+---------+-------+-------+\n"
            "                           \n"
            "                           \n"
            f"Predicted-Actual          \n"
            "_____________________      \n"
            f"Stance-Stance = {P4}      \n"
            f"Swing-Stance  = {P2}      \n"
            f"Stance-Swing  = {P3}      \n"
            f"Swing-Swing  = {P1}       \n "
            "                           \n"
            f"Correct Percent   = {C}   \n"
            f"Incorrect Percent = {I}   \n"
            
            )
    return C1, C2, C3, C4, P1, P2, P3, P4, C, I

def plot_figures(history):
    #Plotting
    
    # --- Plot loss and MAE ---
    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()

    # Plot MAE
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    if 'val_acc' in history:
        plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Model Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # plt.tight_layout()
    # plot_filename = f'Model_BatchSize_{batch_size}_Epochs_{epochs}_LearningRate_{learning_rate}_Neurons_{neurons}_.png'
    # plt.savefig(f'NN_Graphs/{plot_filename}')
    # plt.show()

def plot_predictions(predictions, true_vals, X_val):
   window = 2000
   num_windows = len(X_val) // window
   o = 200

   for i in range(num_windows):
       
       start = i * window
       end = start + window
       o += 1
       
       # Get data for current window
       X_window = X_val[start:end]
       true_window = (true_vals[start:end])
       pred_window = (predictions[start:end]).T
       
       # Track prediction correctness
       correct = (true_window == pred_window).flatten().tolist()
       
       # Find transition points where prediction correctness changes
       transitions = [0]  # Start of window
       for j in range(1, len(correct)):
           if correct[j] != correct[j-1]:
               transitions.append(j)
       transitions.append(len(correct) - 1)  # End of window
       
       # Create plot
       plt.figure(o, figsize=(12, 5))
       plt.plot(X_window)
       
       # Shade regions based on prediction correctness
       for k in range(len(transitions) - 1):
           start_idx = transitions[k]
           end_idx = transitions[k + 1]
           
           if correct[start_idx]:
               plt.axvspan(start_idx, end_idx, color='green', alpha=0.2)
           else:
               plt.axvspan(start_idx, end_idx, color='red', alpha=0.2)
       
       # Labels
       time_start = start / 400
       time_end = end / 400
       plt.title(f'Toe and Heel Strike from time {time_start:.2f} to {time_end:.2f}')
       plt.xlabel("Time [s]")
       plt.ylabel("Radians per Second [rad/s]")
       
       # Uncomment to save
       # plot_filename = f'Check_Plots_from_{time_start:.2f}_to_{time_end:.2f}.png'
       # plt.savefig(f'NN_Video/{plot_filename}')
       
       plt.show()

def plot_figures_MH(history):
    #Plotting
    
    # --- Plot loss and MAE ---
    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_binary_loss'], label='Train Binary Loss')
    if 'val_binary_loss' in history:
        plt.plot(history['val_binary_loss'], label='Validation Binary Loss')
    plt.title('Model Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('BSE Loss')
    plt.legend()

    # Plot MAE
    plt.subplot(1, 2, 2)
    plt.plot(history['train_binary_acc'], label='Train Binary Accuracy')
    if 'val_binary_acc' in history:
        plt.plot(history['val_binary_acc'], label='Validation Binary Accuracy')
    plt.title('Model Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Binary Accuracy')
    plt.legend()
    
    plt.figure(figsize=(12, 5))
    
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_percentage_loss'], label='Train Percentage Loss')
    if 'val_percentage_loss' in history:
        plt.plot(history['val_percentage_loss'], label='Validation Percentage Loss')
    plt.title('Model Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()

    # Plot MAE
    plt.subplot(1, 2, 2)
    plt.plot(history['train_percentage_mae'], label='Train Percentage MAE')
    if 'val_percentage_mae' in history:
        plt.plot(history['val_percentage_mae'], label='Validation Percentage MAE')
    plt.title('Model Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()

    # plt.tight_layout()
    # plot_filename = f'Model_BatchSize_{batch_size}_Epochs_{epochs}_LearningRate_{learning_rate}_Neurons_{neurons}_.png'
    # plt.savefig(f'NN_Graphs/{plot_filename}')
    plt.show()

