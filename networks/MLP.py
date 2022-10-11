import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    
    def __init__(self, n_features, l1, l2, l3):
        super(Net, self).__init__()
        
        self.fc1 = nn.Linear(n_features, l1)
        self.fc1_bn = nn.BatchNorm1d(l1)

        self.fc2 = nn.Linear(l1, l2)
        self.fc2_bn = nn.BatchNorm1d(l2)

        self.fc3 = nn.Linear(l2, l3)
        self.fc3_bn = nn.BatchNorm1d(l3)

        self.fc4 = nn.Linear(l3,1)
        
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.fc3_bn(self.fc3(x)))
        x = self.dropout(x)
        x = self.fc4(x)
        return x
