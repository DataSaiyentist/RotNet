### --- Project models for reduced MNIST ---


import torch
import torch.nn as nn
import torch.nn.functional as F


# --- MLPC
class MLPC(nn.Module) :
    """MLP classifier with extracted features"""
    
    def __init__(self) :
        super().__init__()
        
        self.linear1 = nn.Linear(in_features = 112, out_features = 128)
        self.linear2 = nn.Linear(in_features = 128, out_features = 64)
        self.dropout = nn.Dropout(p = 0.2)
        self.linear3 = nn.Linear(in_features = 64, out_features = 10)
    
    def forward(self, x) :
        
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.dropout(x)
        x = self.linear3(x)
        
        return x


# --- CNN
class CNN(nn.Module) :
    """CNN for classification purpose"""
    
    def __init__(self) :
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 3, padding = "same")
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, padding = "same")
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.fc1 = nn.Linear(in_features = 6272, out_features = 1024)
        self.dropout = nn.Dropout(p = 0.5)
        self.fc2 = nn.Linear(in_features = 1024, out_features = 10)
        
    def forward(self, x) :
        
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


# --- ROTNET
class ROTNET(nn.Module) :
    """Rotnet for digit recognition task"""
    
    def __init__(self) :
        super().__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 3, padding = "same")
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, padding = "same")
        self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, padding = "same")
        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        # Second convolutional block
        self.conv4 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = "same")
        self.conv5 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = "same")
        self.conv6 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = "same")
        self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        # Fully-connected layers
        self.fc1 = nn.Linear(in_features = 3136, out_features = 1024)
        self.fc2 = nn.Linear(in_features = 1024, out_features = 4)
        
    def forward(self, x) :
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pool2(F.relu(self.conv6(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x