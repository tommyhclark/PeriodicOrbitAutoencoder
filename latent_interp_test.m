%%
clc
clear all
% Load training latent data
latent_train = load("Halo2VerticalDoedel/latent_51pts_5sqrt2em4.txt");


% 2. Compute N_train and new interpolants

N_train = length(latent_train);
fprintf(" %i Training Orbits\n", N_train)


new_c = (latent_train(1:end-1) + latent_train(2:end)) / 2;
writematrix(new_c', "Halo2VerticalDoedel/latent_test_fixed_latent.txt", 'Delimiter', ',');

% 3. Pass through decoder inference in python



% 4. Regenerate Plots

