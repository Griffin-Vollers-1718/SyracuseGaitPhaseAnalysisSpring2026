""" This code is for the Ros2 subscriber"""
import rclpy
from rclpy.node import Node 
from geometry_msgs.msg import Twist
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime

class BinaryClassifier(nn.Module):
    def __init__(self, 
                 input_size: int,
                 hidden_size: list[int],
                 dropout_rate: float = 0.2,
                 use_batch_norm: bool = True):
        
        super(BinaryClassifier, self).__init__()
        
        self.use_batch_norm = use_batch_norm
        
        # Input branch with batch normalization
        layers = [nn.Linear(input_size, hidden_size[0])]
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_size[0]))
        layers.extend([
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        ])
        self.input_branch = nn.Sequential(*layers)
        
        # Shared layers with batch normalization
        hidden_layers = [nn.Linear(hidden_size[0], hidden_size[1])]
        if use_batch_norm:
            hidden_layers.append(nn.BatchNorm1d(hidden_size[1]))
        hidden_layers.extend([
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size[1], 64)
        ])
        if use_batch_norm:
            hidden_layers.append(nn.BatchNorm1d(64))
        hidden_layers.append(nn.ReLU())
        
        self.hidden = nn.Sequential(*hidden_layers)
        
        #Output
        self.output = nn.Sequential(
            nn.Linear(64, 1),
            )
    
    def forward(self, input1):
        x = self.input_branch(input1)
        x = self.hidden(x)
        out = self.output(x)
        return out

class NNSubscriber(Node):
    def __init__(self):
        super().__init__('Neural_Net_subscriber_node')
        
        input_size = 3
        hidden_sizes = [64, 64, 64]
        output_size = 1
        dropout_rate = 0.1
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = BinaryClassifier(input_size, hidden_sizes, output_size, dropout_rate).to(self.device)
        self.model.load_state_dict(torch.load('model_weights.pth', map_location=self.device))
        self.model.eval()  # Set to evaluation mode
        
        self.predictions = []
        self.timestamps = []
        
        self.subscription = self.create_subscription(
            Twist, # Needs to be changed depending on topic data. This is the most likely one.
            'topic_name',  #Changed to fit the subscribed topic
            self.listener_callback,
            10)
        self.get_logger().info('Subscriber Running on Ros2 Humble')
        self.subscription
        
    def listener_callback(self, msg):
        data = self.parse_message(msg)
        data_t = torch.FloatTensor(data).unsqueeze(0).to(self.device)
        """ Need to understand the msg being sent from the topic so I can 
        break it down and turn it into a tensor"""
        with torch.no_grad():
            logits = self.model(data_t)
            probabilities = torch.sigmoid(logits)
            prediction = (probabilities > 0.5).float()
            
        self.predictions.append(prediction)
        self.timestamps.append(datetime.now())
        
        self.get_logger().info(f'Prediction: {prediction} at {self.timestamps[-1]}')
        
    def parse_message(self, msg):
        "Need to come back and fix this later"
        return 
    def get_predictions(self):
        """Return all saved predictions"""
        return self.predictions

    def get_timestamps(self):
        """Return all timestamps"""
        return self.timestamps
   
    def save_to_csv(self, filename='predictions.csv'):
        """Save predictions with timestamps to CSV"""
        import csv
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'Prediction'])
            for ts, pred in zip(self.timestamps, self.predictions):
                writer.writerow([ts.strftime("%Y-%m-%d %H:%M:%S.%f"), pred.tolist()])
        
def main(args=None):
    rclpy.init(args=args)
    node = NNSubscriber()
    try:
        # Keep the node running until an exception (e.g., Ctrl+C) occurs
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save_to_csv('final_predictions.csv')
        # Destroy the node explicitly
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()