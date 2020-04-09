import torch
import torch.nn as nn
import numpy as np

    
class LSTM_Init_To_Many(nn.Module):
    def __init__(self, input_size=6, hidden_layer_size=100, output_size=2, n_hidden_layer=2, dummy_input_size=1, seq_len=150):
        super().__init__()
        self.dummy_input_size = dummy_input_size
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(self.dummy_input_size, hidden_layer_size, batch_first=True)
        
        self.linear_in_h = nn.Sequential(nn.Linear(input_size, 200), nn.ReLU(), nn.Linear(200, hidden_layer_size))
        self.linear_in_c = nn.Sequential(nn.Linear(input_size, 200), nn.ReLU(), nn.Linear(200, hidden_layer_size))
        
        self.n_hidden_layer = n_hidden_layer
        self.output_size = output_size
        self.linear_out = nn.Linear(hidden_layer_size, self.output_size)
        
        self.seq_len = seq_len
        
    def forward(self, init_input):
        self.init_h = self.linear_in_h(init_input).unsqueeze(0)
        self.init_c = self.linear_in_c(init_input).unsqueeze(0)
        
        self.hidden_cell = (self.init_h, self.init_c)
        self.dummy_input_sequence = torch.zeros(init_input.shape[0], self.seq_len, self.dummy_input_size).to(init_input.device)
        
        lstm_out, self.hidden_cell = self.lstm(self.dummy_input_sequence, self.hidden_cell)
        assert lstm_out.shape == (init_input.shape[0], self.seq_len, self.hidden_layer_size)
        
        predictions = self.linear_out(torch.reshape(lstm_out, (init_input.shape[0] * self.seq_len, self.hidden_layer_size)))
        predictions = torch.reshape(predictions, (init_input.shape[0], self.seq_len, self.output_size))
        return predictions
    