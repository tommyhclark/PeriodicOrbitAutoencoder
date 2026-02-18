import torch
import torch.nn as nn

class AutoencoderSplit(nn.Module):
    def __init__(self):
        super(AutoencoderSplit, self).__init__()
        self.N=51
        self.encoder_states = nn.Sequential(
            nn.Linear(self.N * 6, self.N * self.N),
            nn.LeakyReLU(),
            nn.Linear(self.N * self.N, 2 * self.N),
            nn.LeakyReLU(),
        )

        self.encoder_period = nn.Sequential(
            nn.Linear(1, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 2 * self.N),
            nn.LeakyReLU(),
        )

        self.to_latent = nn.Linear(4 * self.N, 1)
        self.from_latent = nn.Linear(1, 4 * self.N)

        # ----- decoder (unchanged) -----
        self.decoder_states = nn.Sequential(
            nn.Linear(2 * self.N, self.N * self.N),
            nn.LeakyReLU(),
            nn.Linear(self.N * self.N, self.N * 6)
        )

        self.decoder_period = nn.Sequential(
            nn.Linear(2 * self.N, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        state_input  = x[:, :-1]
        period_input = x[:, -1:]

        state_h  = self.encoder_states(state_input)
        period_h = self.encoder_period(period_input)

        combined = torch.cat([state_h, period_h], dim=1)
        z  = self.to_latent(combined)

        dec_h = self.from_latent(z)
        dec_h_state, dec_h_period = torch.split(dec_h, 2 * self.N, dim=1)

        recon_state  = self.decoder_states(dec_h_state)
        recon_period = self.decoder_period(dec_h_period)

        recon = torch.cat([recon_state, recon_period], dim=1)
        return recon

    def encode(self, x):
        """Return the 1-D latent variable."""
        with torch.no_grad():
            state_input  = x[:, :-1]
            period_input = x[:, -1:]
            state_h  = self.encoder_states(state_input)
            period_h = self.encoder_period(period_input)
            combined = torch.cat([state_h, period_h], dim=1)
            return self.to_latent(combined)


    def decode(self, z):
        """Reconstruct from latent z (shape: (B, 1))."""
        dec_h = self.from_latent(z)
        dec_h_state, dec_h_period = torch.split(dec_h, 2 * self.N, dim=1)

        recon_state  = self.decoder_states(dec_h_state)
        recon_period = self.decoder_period(dec_h_period)

        return torch.cat([recon_state, recon_period], dim=1)