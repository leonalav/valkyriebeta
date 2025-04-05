import torch
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        in_channels = config.hidden_size
        
        for out_channels, kernel_size in zip(config.conv_channels, config.conv_kernel_sizes):
            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                nn.LayerNorm([out_channels, config.max_seq_length]),
                nn.ReLU(),
                nn.Dropout(config.hidden_dropout)
            ))
            in_channels = out_channels
            
        # GRU layers
        self.gru = nn.GRU(
            input_size=config.conv_channels[-1],
            hidden_size=config.gru_hidden_size,
            num_layers=config.gru_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.hidden_dropout if config.gru_num_layers > 1 else 0
        )
        
        # Output projection
        self.output_proj = nn.Linear(
            config.gru_hidden_size * 2,  # * 2 for bidirectional
            config.hidden_size
        )
        
    def forward(self, hidden_states):
        # Convert to channel-first format for convolutions
        x = hidden_states.transpose(1, 2)
        
        # Apply convolutional layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
            
        # Convert back to sequence-first format for GRU
        x = x.transpose(1, 2)
        
        # Apply GRU
        x, _ = self.gru(x)
        
        # Project output
        output = self.output_proj(x)
        
        return output 