import torch
from torch import nn
class LSTMAutoencoderSrivastava(nn.Module):
    def __init__(self, n_features, hidden_size=90, n_layers=10, batch_first=False):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.batch_first = batch_first
        self.time_axis = 1 if batch_first else 0
        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=batch_first,
        )
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=n_features,
            num_layers=n_layers,
            batch_first=batch_first,
        )

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        h = h[-1].view(1, 1, -1)
        x_flipped = torch.flip(x[1:], dims=[self.time_axis])
        input = torch.cat((h, x_flipped), dim=self.time_axis)
        x_hat, _ = self.decoder(input)
        x_hat = torch.flip(x_hat, dims=[self.time_axis])

        return x_hat

'''
Architecture inspired by Cho et al. 2014 (https://arxiv.org/abs/1406.1078).
 Decoding occurs in natural order and the decoder is only provided with the 
 encoding at every timestep.
'''
class LSTMAutoencoderCho(nn.Module):
    def __init__(self, n_features, hidden_size=90, n_layers=1, batch_first=False):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.batch_first = batch_first
        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=batch_first,
        )
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=n_features,
            num_layers=n_layers,
            batch_first=batch_first,
        )

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        target_shape = (
            (-1, x.shape[0], -1) if self.batch_first else (x.shape[0], -1, -1)
        )
        h = h[-1].expand(target_shape)
        x_hat, _ = self.decoder(h)
        return x_hat

'''
LSTM Encoder-Decoder architecture by Sutskever et al. 2014 (https://arxiv.org/abs/1409.3215). 
The decoder only gets access to its own prediction of the previous timestep. Decoding also 
takes performed backwards.
'''
class LSTMDecoder(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        sequence_length=None,
        predict_backward=True,
        num_layers=1,
    ):
        super().__init__()

        self.cell = nn.LSTMCell(input_size, hidden_size)
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.predict_backward = predict_backward
        self.sequence_length = sequence_length
        self.num_layers = num_layers
        self.lstm = (
            None
            if num_layers <= 1
            else nn.LSTM(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=num_layers - 1,
            )
        )
        self.linear = (
            None if input_size == hidden_size else nn.Linear(hidden_size, input_size)
        )

    def forward(self, h, sequence_length=None):
        """Computes the forward pass.

        Parameters
        ----------
        x:
            Input of shape (batch_size, input_size)

        Returns
        -------
        Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            Decoder outputs (output, (h, c)) where output has the shape (sequence_length, batch_size, input_size).
        """

        if sequence_length is None:
            sequence_length = self.sequence_length
        x_hat = torch.empty(sequence_length, h.shape[0], self.hidden_size)
        for t in range(sequence_length):
            if t == 0:
                h, c = self.cell(h)
            else:
                input = h if self.linear is None else self.linear(h)
                h, c = self.cell(input, (h, c))
            t_predicted = -t if self.predict_backward else t
            x_hat[t_predicted] = h

        if self.lstm is not None:
            x_hat = self.lstm(x_hat)

        return x_hat, (h, c)


class LSTMAutoencoderSutskever(nn.Module):
    def __init__(self, n_features, hidden_size=90, n_layers=1):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.encoder = nn.LSTM(
            input_size=n_features, hidden_size=hidden_size, num_layers=n_layers
        )
        self.decoder = LSTMDecoder(
            input_size=hidden_size, hidden_size=n_features, predict_backward=True
        )

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        x_hat, _ = self.decoder(h[-1], x.shape[0])
        return x_hat