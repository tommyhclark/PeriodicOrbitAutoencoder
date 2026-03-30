import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self):
        N = 51
        p1 = 5*N
        p2 = 1*N
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(N*6+1, p1),
            nn.LeakyReLU(),
            nn.Linear(p1, p2),
            nn.LeakyReLU(),
            nn.Linear(p2,1),
        )
        self.decoder = nn.Sequential(
            nn.Linear(1, p2),
            nn.LeakyReLU(),
            nn.Linear(p2, p1),
            nn.LeakyReLU(),
            nn.Linear(p1,N*6+1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)