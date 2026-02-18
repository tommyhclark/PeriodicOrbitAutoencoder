%% Test Decoder Output
clear all
% close all 
clc
% parpool('threads', 10);
addpath("src/utils")
const = load('data/cr3bp_constants.mat','const').const;
l2 = const.l2;

config = readyaml("HaloFamilySplitPlanarSupplement/halo_config.yaml");

test_data = readmatrix(config.test_data_path);
decoded_data = readmatrix(config.decoded_data_path);

N=51;
Norb = size(decoded_data,2);
test_periods = test_data(N*6+1,:);
decoded_periods = decoded_data(N*6+1,:);
test_states = reshape(test_data(1:N*6,:), 6, N, Norb);
test_states = permute(test_states,[3 2 1]);
test_states(:,:,1) = test_states(:,:,1) + l2;
decoded_states = reshape(decoded_data(1:N*6,:), 6, N, Norb);
decoded_states = permute(decoded_states,[3 2 1]);
decoded_states(:,:,1) = decoded_states(:,:,1) + l2;

%% Correct Decoded States 
clc
mu = const.mu;
tic
decoded_states_corrected = NaN(size(decoded_states));
decoded_states_notcorrected = NaN(size(decoded_states));
initial_constraint_violations = NaN(size(decoded_states,1),1);

parfor ii=1:length(decoded_states)
    fprintf("%i \n", ii)
    [v,iter,diff, corrected_flag, initial_constraint_violation, initial_step_size] = multipleShooting(squeeze(decoded_states(ii,:,1:6)),decoded_periods(ii),mu);
    if corrected_flag
        decoded_states_corrected(ii,:,1:6) = v;
    else
        decoded_states_notcorrected(ii,:,1:6) = decoded_states(ii,:,1:6);
    end
    initial_constraint_violations(ii) = initial_constraint_violation;
end
not_corrected_idxs = find(~isnan(decoded_states_notcorrected(:,1,1)));
toc



% Integrate Decoded States
clc
decoded_states_integrated = NaN(size(decoded_states_corrected,1),400,6);
options = odeset('RelTol', 3e-14, 'AbsTol',1e-16);

parfor ii=1:length(decoded_states_corrected)
    disp(ii)
    [~,y1] = ode113(@(t,state) cr3bpStateOnly(state, mu), linspace(0,decoded_periods(ii),400), decoded_states_corrected(ii,1,1:6), options);
    decoded_states_integrated(ii,:,1:6) = y1;
end
% delete(gcp("nocreate"))
%% Orbit Family Plot

figure;
hold on
% Plot Decoded Family
for ii = 1:1:Norb
    states = squeeze(decoded_states_corrected(ii,:,:));
    % scatter3(states(:,1),states(:,2),states(:,3),2, "filled","k")
    plot3([states(:,1);states(1,1)],[states(:,2);states(1,2)],[states(:,3);states(1,3)],"LineWidth",1e-3)
end
% Highlight Not Corrected Orbits (decoder output shown)
for ii = 1:length(not_corrected_idxs)
    idx = not_corrected_idxs(ii);
    states = squeeze(decoded_states_notcorrected(idx,:,:));
    scatter3(states(:,1),states(:,2),states(:,3),20,"red")
end

% Plot L2 Lagrange Point as red diamond
scatter3(l2,0,0,"red","filled","diamond")
xlabel("X")
ylabel("Y")
zlabel("Z")
title("AI Generated $L_2$ Halo Family","Interpreter","latex")
axis equal

% Add moon as a 3D sphere
moon_radius = const.moon_R;
[Xm, Ym, Zm] = sphere(50);
Xm = moon_radius * Xm + (1 - mu);
Ym = moon_radius * Ym;
Zm = moon_radius * Zm;
surf(Xm, Ym, Zm, 'FaceColor', "Black", 'EdgeColor', 'none');
view([0 0])

%% Plot Jacobi Constant Deviations 
% Conclusions: Highest Deviations are near Perilune on short period orbits
% (NRHOs)
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

figure; 
hold on
scatter3(torb(not_corrected_idxs,:), orbT(not_corrected_idxs,:),cdiff(not_corrected_idxs,:),40,"red","filled")
scatter3(torb(corrected_idxs,:), orbT(corrected_idxs,:),cdiff(corrected_idxs,:),20,"k","filled")
title("Corrected Orbits")
ylabel("Weird Period")
xlabel("Time along orbit")
zlabel("Jacobi Diff")

%% Performance Plots


%% Initial constraint violations per point histogram
figure;
min_val = min(initial_constraint_violations(initial_constraint_violations > 0));
max_val = max(initial_constraint_violations);
bin_edges = logspace(log10(min_val), log10(max_val), 31);
histogram(initial_constraint_violations,bin_edges,"Normalization","probability","FaceColor",[0, 0.3, 0.6])
hold on
plot(median(initial_constraint_violations)*ones(11),0:0.1:1,"LineStyle","--","Color","k","LineWidth",3)
xlabel("Decoded Trajectory Constraint Violation","FontSize",20,"Interpreter","latex")
ylabel("Fraction of Test Set","FontSize",20,"Interpreter","latex")
set(gca, 'XScale', 'log', 'LineWidth', 2, 'FontSize', 14, 'FontWeight', 'bold', 'TickLabelInterpreter', 'latex');
ylim([0 0.17])
fprintf("Median constraint violation: %.2e\n",median(initial_constraint_violations))

%% Initial constraint violations per point histogram with orbit coloring
figure;
tiledlayout(1,2);
nexttile;
min_val = min(initial_constraint_violations(initial_constraint_violations > 0));
max_val = max(initial_constraint_violations);
bin_edges = logspace(log10(min_val), log10(max_val), 31);
bin_centers = sqrt(bin_edges(1:end-1) .* bin_edges(2:end)); % geometric centers for log bins
[N, ~] = histcounts(initial_constraint_violations, bin_edges, 'Normalization', 'probability');
cmap = parula(numel(bin_centers));
norm_vals = (log10(bin_centers) - log10(min_val)) / (log10(max_val) - log10(min_val));
hold on
for i = 1:numel(bin_centers)
    patch('XData', [bin_edges(i) bin_edges(i+1) bin_edges(i+1) bin_edges(i)], ...
          'YData', [0 0 N(i) N(i)], ...
          'FaceColor', cmap(i,:), 'EdgeColor', 'none');
end
set(gca, 'XScale', 'log', 'LineWidth', 2, 'FontSize', 14, ...
         'FontWeight', 'bold', 'TickLabelInterpreter', 'latex');
xlabel('Decoded Trajectory Constraint Violation','FontSize',20,'Interpreter','latex')
ylabel('Fraction of Test Set','FontSize',20,'Interpreter','latex')
ylim([0 0.17])
clim([min_val max_val])

nexttile;
hold on
cmap =  parula(256);
nC = size(cmap,1);
for ii = 1:Norb
    states = squeeze(decoded_states_integrated(ii,:,:));
    val = initial_constraint_violations(ii);
    idx = max(1, min(nC, round(1 + (nC-1)*(log10(val) - log10(min_val)) / (log10(max_val) - log10(min_val)))));
    plot3([states(:,1); states(1,1)], [states(:,2); states(1,2)], [states(:,3); states(1,3)], ...
          'Color', cmap(idx,:), 'LineWidth', 0.5);
end
xlabel('X','FontSize',20,'Interpreter','latex')
ylabel('Y','FontSize',20,'Interpreter','latex')
zlabel('Z','FontSize',20,'Interpreter','latex')
axis equal
view([0 0])
set(gca, 'LineWidth', 2, 'FontSize', 14, 'FontWeight', 'bold', 'TickLabelInterpreter', 'latex');
colormap(parula)
colorbar
clim([min_val max_val])
title('Decoded Orbits Colored by Constraint Violation','Interpreter','latex')



%% Performance Plots - How close output is to actual

figure;
histogram(abs(test_periods-decoded_periods))