import numpy as np
import matplotlib.pyplot as plt

class LSTM:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights
        # Forget gate weights
        self.W_f = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.b_f = np.zeros((hidden_size, 1))
        
        # Input gate weights
        self.W_i = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.b_i = np.zeros((hidden_size, 1))
        
        # Candidate memory weights
        self.W_c = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.b_c = np.zeros((hidden_size, 1))
        
        # Output gate weights
        self.W_o = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.b_o = np.zeros((hidden_size, 1))
        
        # Output layer weights
        self.W_y = np.random.randn(output_size, hidden_size) * 0.01
        self.b_y = np.zeros((output_size, 1))
        
        self.hidden_size = hidden_size
        self.input_size = input_size
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))  # Clip to prevent overflow
    
    def tanh(self, x):
        return np.tanh(x)
    
    def forward(self, x_sequence):
        """
        Forward pass through the LSTM
        x_sequence: list of input vectors, each of shape (input_size, 1)
        Returns: outputs, hidden states, cell states, and gate values
        """
        T = len(x_sequence)  # Sequence length
        
        # Initialize hidden state and cell state
        h = np.zeros((self.hidden_size, 1))
        c = np.zeros((self.hidden_size, 1))
        
        # Store values for backpropagation
        self.x_sequence = x_sequence
        self.h_states = [h]
        self.c_states = [c]
        self.f_gates = []
        self.i_gates = []
        self.c_candidates = []
        self.o_gates = []
        self.outputs = []
        
        # Process each timestep
        for t in range(T):
            # Concatenate previous hidden state with current input
            x_t = x_sequence[t]
            combined = np.vstack((h, x_t))
            
            # Forget gate
            f_t = self.sigmoid(np.dot(self.W_f, combined) + self.b_f)
            
            # Input gate
            i_t = self.sigmoid(np.dot(self.W_i, combined) + self.b_i)
            
            # Candidate cell state
            c_candidate = self.tanh(np.dot(self.W_c, combined) + self.b_c)
            
            # Update cell state
            c = f_t * c + i_t * c_candidate
            
            # Output gate
            o_t = self.sigmoid(np.dot(self.W_o, combined) + self.b_o)
            
            # Update hidden state
            h = o_t * self.tanh(c)
            
            # Output layer
            y_t = np.dot(self.W_y, h) + self.b_y
            
            # Store values
            self.h_states.append(h)
            self.c_states.append(c)
            self.f_gates.append(f_t)
            self.i_gates.append(i_t)
            self.c_candidates.append(c_candidate)
            self.o_gates.append(o_t)
            self.outputs.append(y_t)
        
        return self.outputs
    
    def backward(self, dY, learning_rate=0.01):
        """
        Backward pass through time (BPTT)
        dY: list of gradients from output, each of shape (output_size, 1)
        """
        T = len(dY)
        
        # Initialize gradients
        dW_f = np.zeros_like(self.W_f)
        db_f = np.zeros_like(self.b_f)
        dW_i = np.zeros_like(self.W_i)
        db_i = np.zeros_like(self.b_i)
        dW_c = np.zeros_like(self.W_c)
        db_c = np.zeros_like(self.b_c)
        dW_o = np.zeros_like(self.W_o)
        db_o = np.zeros_like(self.b_o)
        dW_y = np.zeros_like(self.W_y)
        db_y = np.zeros_like(self.b_y)
        
        # Initialize gradients for next step
        dh_next = np.zeros_like(self.h_states[0])
        dc_next = np.zeros_like(self.c_states[0])
        
        # Backward pass through time
        for t in reversed(range(T)):
            # Output layer gradients
            dW_y += np.dot(dY[t], self.h_states[t+1].T)
            db_y += dY[t]
            
            # Hidden state gradient from output and next timestep
            dh = np.dot(self.W_y.T, dY[t]) + dh_next
            
            # Output gate gradients
            do = dh * self.tanh(self.c_states[t+1])
            do_raw = do * self.o_gates[t] * (1 - self.o_gates[t])
            
            # Cell state gradients
            dc = dc_next + dh * self.o_gates[t] * (1 - self.tanh(self.c_states[t+1])**2)
            
            # Input gate gradients
            di = dc * self.c_candidates[t]
            di_raw = di * self.i_gates[t] * (1 - self.i_gates[t])
            
            # Candidate gradients
            dc_candidate = dc * self.i_gates[t]
            dc_candidate_raw = dc_candidate * (1 - self.c_candidates[t]**2)
            
            # Forget gate gradients
            df = dc * self.c_states[t]
            df_raw = df * self.f_gates[t] * (1 - self.f_gates[t])
            
            # Combined input for all gates
            combined = np.vstack((self.h_states[t], self.x_sequence[t]))
            
            # Weight updates
            dW_o += np.dot(do_raw, combined.T)
            db_o += do_raw
            
            dW_i += np.dot(di_raw, combined.T)
            db_i += di_raw
            
            dW_c += np.dot(dc_candidate_raw, combined.T)
            db_c += dc_candidate_raw
            
            dW_f += np.dot(df_raw, combined.T)
            db_f += df_raw
            
            # Gradients for previous timestep
            dcombined = (np.dot(self.W_o.T, do_raw) + 
                        np.dot(self.W_i.T, di_raw) + 
                        np.dot(self.W_c.T, dc_candidate_raw) + 
                        np.dot(self.W_f.T, df_raw))
            
            dh_next = dcombined[:self.hidden_size]
            dc_next = self.f_gates[t] * dc
        
        # Update weights
        self.W_f -= learning_rate * dW_f
        self.b_f -= learning_rate * db_f
        self.W_i -= learning_rate * dW_i
        self.b_i -= learning_rate * db_i
        self.W_c -= learning_rate * dW_c
        self.b_c -= learning_rate * db_c
        self.W_o -= learning_rate * dW_o
        self.b_o -= learning_rate * db_o
        self.W_y -= learning_rate * dW_y
        self.b_y -= learning_rate * db_y

def mean_squared_error(y_true, y_pred):
    """Calculate mean squared error"""
    return np.mean([np.sum((yt - yp)**2) for yt, yp in zip(y_true, y_pred)])

# Example usage and training
def create_sine_wave_data(seq_length=50, num_sequences=100):
    """Create synthetic sine wave data for training"""
    X, Y = [], []
    for _ in range(num_sequences):
        # Random phase and frequency
        phase = np.random.uniform(0, 2*np.pi)
        freq = np.random.uniform(0.5, 2.0)
        
        # Generate sequence
        t = np.linspace(0, 4*np.pi, seq_length)
        sequence = np.sin(freq * t + phase)
        
        # Create input-output pairs (predict next value)
        x_seq = [sequence[i:i+1].reshape(-1, 1) for i in range(seq_length-1)]
        y_seq = [sequence[i+1:i+2].reshape(-1, 1) for i in range(seq_length-1)]
        
        X.append(x_seq)
        Y.append(y_seq)
    
    return X, Y

# Training example
def train_lstm_example():
    # Create data
    X_train, Y_train = create_sine_wave_data(seq_length=20, num_sequences=100)
    
    # Initialize LSTM
    lstm = LSTM(input_size=1, hidden_size=16, output_size=1)
    
    # Training loop
    losses = []
    for epoch in range(100):
        total_loss = 0
        for x_seq, y_seq in zip(X_train, Y_train):
            # Forward pass
            outputs = lstm.forward(x_seq)
            
            # Calculate loss and gradients
            loss = mean_squared_error(y_seq, outputs)
            total_loss += loss
            
            # Calculate output gradients
            dY = [2 * (outputs[i] - y_seq[i]) for i in range(len(outputs))]
            
            # Backward pass
            lstm.backward(dY, learning_rate=0.01)
        
        avg_loss = total_loss / len(X_train)
        losses.append(avg_loss)
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")
    
    # Plot training loss
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    
    # Test the trained LSTM
    plt.subplot(1, 2, 2)
    test_sequence = [np.array([[np.sin(i * 0.2)]]) for i in range(30)]
    predictions = lstm.forward(test_sequence)
    
    true_values = [np.sin(i * 0.2) for i in range(1, 31)]
    pred_values = [p[0, 0] for p in predictions]
    
    plt.plot(range(1, 31), true_values, label='True')
    plt.plot(range(1, 31), pred_values, label='Predicted')
    plt.title('LSTMs Predictions')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_lstm_example()