import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import argparse
import yaml
from models.autoencoder import Autoencoder  # Make sure this path is correct


def main(config_path):
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = config["device"]

    # test_data = np.loadtxt(config["train_data_path"],delimiter=",")
    # test_data = test_data.T
    test_data = np.loadtxt(config["latent_data_fixed_latent"],delimiter=",")
    test_data = test_data.reshape(-1,1)

    test_tensor = torch.tensor(test_data, dtype=torch.float32)
    
    model = Autoencoder()
    model.load_state_dict(torch.load(config["model_path"], map_location=device))
    model = model.to(device)
    model.eval()

   # Run inference
    model.eval()
    with torch.no_grad():
        test_tensor = test_tensor.to(device)
        decoded_test = model.decode(test_tensor)  
        # decoded_test = model(test_tensor) # Shape: (test_size, 307)
        # latent_test = model.encode(test_tensor)
    decoded_test_np = decoded_test.cpu().numpy().T  # Shape: (307, test_size)
    # latent_test_np = latent_test.cpu().numpy()

    np.savetxt(config["decoded_data_fixed_latent_path"], decoded_test_np, delimiter=",")
    # np.savetxt(config["latent_train_data_path"],latent_test_np, delimiter=",")
    # torch.save(model.state_dict(), config["model_path"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with a trained Autoencoder")
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to the inference config YAML file'
    )
    args = parser.parse_args()
    main(args.config)