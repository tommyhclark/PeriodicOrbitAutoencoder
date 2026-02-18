import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self):
        N = 51
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(N*6+1, N*N),
            nn.LeakyReLU(),
            nn.Linear(N*N, 2*N),
            nn.LeakyReLU(),
            nn.Linear(2*N,1),
        )
        self.decoder = nn.Sequential(
            nn.Linear(1, 2*N),
            nn.LeakyReLU(),
            nn.Linear(2*N,N*N),
            nn.LeakyReLU(),
            nn.Linear(N*N,N*6+1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)