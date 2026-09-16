%% This file computes multiple shooting guesses (minus Lagrange point)
% to create training datasets for autoencoder runs.
% 
clear all
clc
close all


addpath("src/utils/")
const = load('data/cr3bp_constants.mat','const').const;

%%%%%%%% MUST FIX %%%%%%%%%%%%%
xShift = const.l2; % ATTENTION %
file_ext = "HaloFamilySubsetTest/L2_HaloNGapped_1em3";
save_ext = "_51pts_training";
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

data_table = readtable("data/raw/L2_HaloN_Impact_Lyap_1em3.txt");
data1 = table2array(data_table);
% data = data1([1:1500, 2000:size(data1,1)],:);
data = data1;

% data_table = readtable("data/raw/L2_Lyap_Halo_Axial_5sqrt2em4.txt");
% data2 = table2array(data_table);
% 
% data_table = readtable("data/raw/L2_Axial_Lyap_Vert_Right_5sqrt2em4.txt");
% data3 = table2array(data_table);
% 
% data_table = readtable("data/raw/L2_Vertical_Axial_Small_5sqrt2em4.txt");
% data4 = table2array(data_table);
% 
% 
% 
% 
% data = [data1;data2;data3;data4];



delete(gcp('nocreate'))
parpool("Threads",14);


%%

% Correct Orbits with single shooting and remove collisions with Moon.
mu = const.mu;
x0_corrected = NaN(length(data),7);
for ii=1:length(data)
    disp(ii)
    [x,T,collision] = singleShooting(data(ii,1:6)', data(ii,7), mu, const.moon_R);
    if ~collision
        x0_corrected(ii,1:6) = x(1,:);
        x0_corrected(ii,7) = T;
    end

end

% Remove orbits intersecting the Moon.
x0_corrected = x0_corrected(~any(isnan(x0_corrected),2),:);


% Create MS Initial Guesses
N=51;

x = NaN(N, length(x0_corrected), 6);
x(1,:,:) = x0_corrected(:,1:6);

options = odeset('RelTol', 3e-14, 'AbsTol',1e-16);
parfor jj = 1:length(x0_corrected)
    disp(jj)
    T = x0_corrected(jj,7);  % Period for this orbit
    dt = T / N;
    tspan = 0:dt:(N-1)*dt;  % Equally spaced times from 0 to (N-1)*dt
    [~, y] = ode113(@(t, state) cr3bpStateOnly(state, mu), tspan, x0_corrected(jj,1:6), options);
    x(:,jj,1:6) = y(:,1:6);
end

%%
figure;
hold on

scatter3(reshape(x(:,:,1),[],1),reshape(x(:,:,2),[],1),reshape(x(:,:,3),[],1),1,"k","filled")

axis equal

%% Correct with MS
% Save positions of each orbit
xMS = NaN(N, length(x0_corrected), 6);
bad_idxs = NaN(size(x,2),1);
parfor ii=1:size(x,2)
    disp(ii)
    [v,iter,diff] = multipleShooting(squeeze(x(:,ii,1:6)),x0_corrected(ii,7),mu);
    if iter>2
        bad_idxs(ii) = ii;
    end
    xMS(:,ii,:) = v;
end


% Save Periods of Each Orbit
t = x0_corrected(:,7);


xMS(:,:,1) = xMS(:,:,1) - xShift;  
trainingdata = reshape(permute(xMS, [3, 1, 2]), 6*size(xMS,1), size(xMS,2));
trainingdata(end+1,:) = t;
writematrix(trainingdata,"data/processed/"+file_ext+save_ext)

disp("DONE")
