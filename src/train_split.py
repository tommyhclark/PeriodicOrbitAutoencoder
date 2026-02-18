import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import time
from models.autoencoder_split import AutoencoderSplit
import yaml
import argparse


def cr3bp_jacobi(states, mu):
    # states: (batch_size, N, 6)
    x, y, z, vx, vy, vz = [states[..., i] for i in range(6)]
    r1 = torch.sqrt((x + mu)**2 + y**2 + z**2)
    r2 = torch.sqrt((x - 1 + mu)**2 + y**2 + z**2)
    U = 0.5*(x**2+y**2)+(1-mu)/r1+mu/r2 + 0.5*mu*(1-mu)
    J = -(vx**2+vy**2+vz**2)+2*U
    return J  # (batch_size, N)

def main(config_path):

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load data
    training_set = np.loadtxt(config["train_data_path"],delimiter=",")
    test_data = np.loadtxt(config["test_data_path"],delimiter=",")
    num_orbits = training_set.shape[1]
    
    # Split training data into train, validation
    np.random.seed(config["random_seed"])
    indices = np.random.permutation(num_orbits)
    train_size = int(config["train_frac"] * num_orbits)
    val_size = int((1-config["train_frac"]) * num_orbits)
    
    train_indices = indices[:train_size]
    train_data = training_set[:, train_indices]
    val_indices = indices[train_size:train_size + val_size]
    val_data = training_set[:, val_indices]
    
    # Transpose data to have shape (num_samples, 607)
    train_data = train_data.T  # Shape: (train_size, 307)
    val_data = val_data.T      # Shape: (val_size, 307)
    test_data = test_data.T    # Shape: (test_size, 307)
    
    # Setup data loaders for training
    train_tensor = torch.tensor(train_data, dtype=torch.float32)
    val_tensor = torch.tensor(val_data, dtype=torch.float32)
    test_tensor = torch.tensor(test_data, dtype=torch.float32)
    train_dataset = TensorDataset(train_tensor)
    val_dataset = TensorDataset(val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    
    # Initialize model and training components
    model = AutoencoderSplit()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    device = config["device"]
    model = model.to(device)
    

    ############## Training loop with validation ################

    # lambda_jacobi = 0.0
    lambda_T = 5/306

    scheduler = StepLR(optimizer, step_size=config["lr_scheduler_step_size"], gamma=config["lr_schedule_gamma"])
    start_time = time.perf_counter()
    for epoch in range(config["num_epochs"]):
        model.train()
        train_loss = 0.0

        for data in train_loader:
            data = data[0].to(device)
            optimizer.zero_grad()
            outputs = model(data)
            
            # Reshape data and outputs (306 = 51*6 states, last is T)
            # N = config["num_MS_points"]
            # states_in = data[:, :N*6].reshape(-1, N, 6)
            # states_out = outputs[:, :N*6].reshape(-1, N, 6)
            
            # Standard MSE reconstruction loss
            mse_loss = nn.MSELoss()(outputs[:,:-1], data[:,:-1])

            # Period Loss
            mse_T_loss = nn.MSELoss()(outputs[:,-1],data[:,-1])
            
            # Unshift positions (x) by adding l2 for correct CR3BP frame
            # mu_tensor = torch.tensor(config['mu'], device=device)
            # l2_tensor = torch.tensor(config['l2'], device=device)
            # states_in_unshift = states_in.clone()
            # states_in_unshift[..., 0] += l2_tensor
            # states_out_unshift = states_out.clone()
            # states_out_unshift[..., 0] += l2_tensor
            
            # Compute accelerations
            # jacobi_in = cr3bp_jacobi(states_in_unshift, mu_tensor)
            # jacobi_out = cr3bp_jacobi(states_out_unshift, mu_tensor)
            
            # Additional MSE on accelerations
            # jacobi_loss = torch.var(jacobi_out)
            
            # Total loss (tune lambda_accel as needed)
            # Hyperparameter; add to config.yaml if desired
            total_loss = mse_loss + lambda_T*mse_T_loss
            
            total_loss.backward()
            optimizer.step()
            train_loss += total_loss.item()
        
        scheduler.step()

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in val_loader:
                data = data[0].to(device)
                outputs = model(data)
                val_loss += nn.MSELoss()(outputs, data).item()
        
        print(f'Epoch [{epoch+1}/{config["num_epochs"]}], '
              f'Train Loss: {train_loss/len(train_loader):.4e}, '
              f'Val Loss: {val_loss/len(val_loader):.4e}')
    
    elapsed_time = time.perf_counter() - start_time
    print(f"Elapsed time: {elapsed_time} seconds")


    # Run inference
    model.eval()
    with torch.no_grad():
        test_tensor = test_tensor.to(device)
        decoded_test = model(test_tensor)  # Shape: (test_size, 307)
        latent_test = model.encode(test_tensor)
    decoded_test_np = decoded_test.cpu().numpy().T  # Shape: (307, test_size)
    latent_test_np = latent_test.cpu().numpy()

    np.savetxt(config["decoded_data_path"], decoded_test_np, delimiter=",")
    np.savetxt(config["latent_data_path"],latent_test_np, delimiter=",")
    torch.save(model.state_dict(), config["model_path"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load configuration file")
    parser.add_argument(
        '--config', 
        type=str, 
        required=True, 
        help='Path to the config YAML file'
    )
    
    args = parser.parse_args()
    main(args.config)