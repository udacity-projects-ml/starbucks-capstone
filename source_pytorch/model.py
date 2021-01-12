# torch imports
import torch.nn.functional as F
import torch.nn as nn


## TODO: Complete this classifier
class BinaryClassifier(nn.Module):
    """
    Define a neural network that performs binary classification.
    The network should accept your number of features as input, and produce 
    a single sigmoid value, that can be rounded to a label: 0 or 1, as output.
    
    Notes on training:
    To train a binary classifier in PyTorch, use BCELoss.
    BCELoss is binary cross entropy loss, documentation: https://pytorch.org/docs/stable/nn.html#torch.nn.BCELoss
    """

    ## TODO: Define the init function, the input params are required (for loading code in train.py to work)
    def __init__(self, input_features, hidden_dim, output_dim):
        """
        Initialize the model by setting up linear layers.
        Use the input parameters to help define the layers of your model.
        :param input_features: the number of input features in your training/test data
        :param hidden_dim: helps define the number of nodes in the hidden layer(s)
        :param output_dim: the number of outputs you want to produce
        """
        super(BinaryClassifier, self).__init__()

        # define any initial layers, here
        
        self.fc1 = nn.Linear(input_features, hidden_dim)
        self.drop_fc1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)        
        self.drop_fc2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.drop_fc3 = nn.Dropout(0.3)
        self.fc4 = nn.Linear(hidden_dim, output_dim)
        self.sig = nn.Sigmoid()
        
                
        #self.fc1 = nn.Linear(input_features, hidden_dim)
        #self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        #self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        #self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        #self.fc5 = nn.Linear(hidden_dim, output_dim)
        #self.drop = nn.Dropout(0.4)
        #self.sig = nn.Sigmoid()
        

    
    ## TODO: Define the feedforward behavior of the network
    def forward(self, x):
        """
        Perform a forward pass of our model on input features, x.
        :param x: A batch of input features of size (batch_size, input_features)
        :return: A single, sigmoid-activated value as output
        """
        
        # define the feedforward behavior
        out = F.relu(self.fc1(x)) # activation on hidden layer
        out = self.drop_fc1(out)
        out = F.relu(self.fc2(out))
        out = self.drop_fc2(out)
        out = F.relu(self.fc3(out))
        out = self.drop_fc3(out)
        out = self.fc4(out)
        return self.sig(out) # returning class score
    
        # define the feedforward behavior
        #out = F.relu(self.fc1(x)) # activation on hidden layer
        #out = F.relu(self.fc2(out))
        #out = F.relu(self.fc3(out))
        #out = F.relu(self.fc4(out))
        #out = self.drop(out)
        #out = self.fc5(out)
        #return self.sig(out) # returning class score