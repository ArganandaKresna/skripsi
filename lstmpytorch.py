import torch
import torch.nn as nn

class PyTorchLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        output = self.linear(lstm_out)
        return output

# Example usage
model = PyTorchLSTM(1, 16, 1)