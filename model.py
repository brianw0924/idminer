import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_sequence


class FrameEncoder(nn.Module):
    def __init__(self, trunk_params):
        super().__init__()
        self.linear = nn.Linear(
            trunk_params["input_size"],
            trunk_params["hidden_size"]
        )

    def forward(self, x):
        if isinstance(x, list):
            videos = []
            for video in x:
                videos.append(self.linear(video.to('cuda:0')))
            videos = pack_sequence(sequences=videos)
            videos.device = videos.data.device
            return videos
        else:
            return self.linear(x)

class VideoEncoder(nn.Module):
    def __init__(self, embedder_params):
        super().__init__()
        self.gru = nn.GRU(
            input_size = embedder_params["input_size"],
            hidden_size = embedder_params["hidden_size"],
            num_layers = embedder_params["num_layers"],
            dropout = embedder_params["dropout"],
            bidirectional = embedder_params["bidirectional"],
            batch_first = True
        )
        self.linear = nn.Linear(
            embedder_params["hidden_size"] * 2,
            embedder_params["hidden_size"] * 2
        )

    def forward(self, x) -> torch.Tensor:
        x = self.gru(x)
        '''
        gru output:
        (B, L, D * hidden_size)
        (D * num_layers, B, hidden_size)
        '''
        _, h_n = x
        return self.linear(torch.cat((h_n[-2], h_n[-1]), dim=1))
    
