%% Test Decoder Output
clear all
close all 
clc
delete(gcp('nocreate'))
parpool('threads', 14);
addpath("src/utils")
const = load('data/cr3bp_constants.mat','const').const;

%%%%%%%%%% Must Fix %%%%%%%%%%%%%%%%%%%%%%
xShift = const.l2;
config = readyaml("Halo2VerticalDoedel/config.yaml");
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

test_data = readmatrix(config.test_data_path);
decoded_data = readmatrix(config.decoded_data_path_runs);

N=51;
Norb = size(decoded_data,2);
test_periods = test_data(N*6+1,:);
decoded_periods = decoded_data(N*6+1,:);
test_states = reshape(test_data(1:N*6,:), 6, N, Norb);
test_states = permute(test_states,[3 2 1]);
test_states(:,:,1) = test_states(:,:,1) + xShift;
decoded_states = reshape(decoded_data(1:N*6,:), 6, N, Norb);
decoded_states = permute(decoded_states,[3 2 1]);
decoded_states(:,:,1) = decoded_states(:,:,1) + xShift;

% Correct Decoded States 
clc
mu = const.mu;

decoded_states_corrected = NaN(size(decoded_states));
decoded_states_notcorrected = NaN(size(decoded_states));
decoded_period_corrected = NaN(size(decoded_states,1),1);
initial_constraint_violations = NaN(size(decoded_states,1),1);
initial_step_sizes = NaN(size(decoded_states,1),1);
steps_required = NaN(size(decoded_states,1),1);
parfor ii=1:length(decoded_states)
    fprintf("%i \n", ii)
    [v,iter,diff, corrected_flag, initial_constraint_violation, initial_step_size, corrected_period] = multipleShootingVariablePeriod(squeeze(decoded_states(ii,:,1:6)),decoded_periods(ii),mu);
    if corrected_flag
        decoded_states_corrected(ii,:,1:6) = v;
    else
        decoded_states_notcorrected(ii,:,1:6) = decoded_states(ii,:,1:6);
    end
    initial_constraint_violations(ii) = initial_constraint_violation;
    initial_step_sizes(ii) = initial_step_size/sqrt(N);
    steps_required(ii) = iter;
    decoded_period_corrected(ii) = corrected_period;
end
not_corrected_idxs = find(~isnan(decoded_states_notcorrected(:,1,1)));


% Compute Jacobi constant deviations, Apolune Z, etc.
maxcdev = NaN(Norb,1);
zapolune = NaN(Norb,1);
cdiff = NaN(Norb,N);
orbz = NaN(Norb,N);
torb = NaN(Norb,N);
orbT = NaN(Norb,N);
for jj = 1:Norb
    c = jacobiConstant(squeeze(decoded_states(jj,:,1:6)),mu);
    cdiff(jj,:) = c-mean(c);
    max1 = max(squeeze(decoded_states(jj,:,3)));
    min1 = min(squeeze(decoded_states(jj,:,3)));
    zapolune(jj) = max1;
    if abs(min1)>abs(max1)
        zapolune(jj) = min1;
    end
    orbz(jj,:) = zapolune(jj);
    max1 = max(c-mean(c));
    min1 = min(c-mean(c));
    maxcdev(jj) = max1;
    if abs(min1)>abs(max1)
        maxcdev(jj) = min1;
    end
    torb(jj,:) = 1:N;
    orbT(jj,:) = test_periods(jj)*sign(zapolune(jj));
end

[~,sortidx] = sort(sign(zapolune).*test_periods);
orbz_resorted = orbz(sortidx,:);
torb_resorted = torb(sortidx,:);
cdiff_resorted = cdiff(sortidx,:);
orbT_resorted = orbT(sortidx,:);

corrected_idxs = find(~ismember(1:Norb,not_corrected_idxs));


% Information Used In Words
% Size of corrections (total)
% Correction Steps Required
% Period Correction Required

total_correction = mean(vecnorm(decoded_states_corrected(:,:,:) - decoded_states(:,:,:),2,3),2);
T_diff = abs(decoded_periods-decoded_period_corrected');
abscdiff = abs(cdiff);

fprintf("Median Jacobi Constant Difference %.4e \n",median(abscdiff,'all'))
fprintf("Median Initial Constraint Violation %.4e \n",median(initial_constraint_violations))
fprintf("Median Steps Required %.2e \n",median(steps_required))
fprintf("Median Period Correction %.4e \n",median(T_diff))
fprintf("Median State Correction %.4e \n",median(total_correction,"all"))

