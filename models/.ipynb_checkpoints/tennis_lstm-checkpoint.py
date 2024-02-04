import numpy as np
import pandas as pd
import torch
import torch.nn as nn

class TennisLSTM(nn.Module):

    #input dim: dimension size of hidden state
    #num_layer: number of lstms in stack
    #predict mask, whether or not to predict confidence mask
    def __init__(self, input_dim, feature_dim, hidden_dim, batch_size,
                    num_layers=1, predict_mask=False, **kwargs):
        super(TennisLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.predict_mask = predict_mask
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.feature_dim = feature_dim
        self.feature_fc = nn.Linear(self.input_dim, self.feature_dim)
        self.lstm = nn.LSTM(self.feature_dim, self.hidden_dim, self.num_layers, dropout=0.1)
        if predict_mask:
            output_dim = 2
        else:
            output_dim = 1
        self.linear = nn.Linear(self.hidden_dim, output_dim)

    def get_blank_hidden(self):
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).to(self.device),
            torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).to(self.device))

    def forward(self, input):
        hidden = self.get_blank_hidden()
        feature_out = self.feature_fc(input)
        lstm_out, _ = self.lstm(feature_out.view(feature_out.size(1), self.batch_size, -1), hidden)
        linear_output = self.linear(lstm_out)
        out = torch.sigmoid(linear_output)
        if self.predict_mask:
            y_pred = out[:,:,0]
            mask = out[:,:,1]
            return y_pred.view(-1), mask.view(-1)
        else:
            return out.view(-1)
