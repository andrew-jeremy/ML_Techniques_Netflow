from river import compose, preprocessing, metrics, datasets

from river_torch.anomaly import RollingAutoencoder
from river_torch.anomaly  import Autoencoder
from torch import nn, manual_seed
import torch
from tqdm import tqdm
from river import anomaly
#--------------------------------------->
# from river.anomaly.base import SupervisedAnomalyDetector
# DOES NOT WORK AS SUPERVISED. FEATURE IS NOT AVAILABLE IN CURRENT VERSION OF RIVER (0.15.0) - 04/20/2022
#--------------------------------------->

class LSTMAutoencoderSrivastava(nn.Module):
    def __init__(self, n_features, hidden_size=30, n_layers=1, batch_first=False):
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

if __name__ == "__main__":  
    _ = manual_seed(42)
    dataset = datasets.CreditCard().take(5000)
    metric = metrics.ROCAUC(n_thresholds=50)

    module = LSTMAutoencoderSrivastava # Set this variable to your architecture of choice

    ae = RollingAutoencoder(module=module, lr=0.005)
    scaler = preprocessing.StandardScaler()
    #hst = anomaly.HalfSpaceTrees(n_trees=5,height=3,window_size=3,seed=42)
    #hst = anomaly.GaussianScorer()

    for x, y in tqdm(list(dataset)):
        scaler.learn_one(x)
        x = scaler.transform_one(x)
        score = ae.score_one(x=x)
        metric.update(y_true=y, y_pred=score)
        ae.learn_one(x=x, y=y)
    print(f"ROCAUC: {metric.get():.4f}")
