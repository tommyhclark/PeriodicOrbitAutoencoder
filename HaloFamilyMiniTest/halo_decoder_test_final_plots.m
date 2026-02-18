%% Test Decoder Output
clear all
close all 
clc
delete(gcp('nocreate'))
parpool('threads', 14);
addpath("src/utils")
const = load('data/cr3bp_constants.mat','const').const;
xShift = const.l2;
l2 = const.l2;
config = readyaml("HaloFamilyMiniTest/halo_config.yaml");

test_data = readmatrix(config.test_data_path);
decoded_data = readmatrix(config.decoded_data_path);
latent_data = readmatrix(config.latent_data_path);

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
%%
figure;
hold on

scatter3(reshape(decoded_states(:,:,1),[],1),reshape(decoded_states(:,:,2),[],1),reshape(decoded_states(:,:,3),[],1),1,"k","filled")

axis equal
%%
% Correct Decoded States 

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

% Integrate Decoded States
clc
decoded_states_integrated = NaN(size(decoded_states_corrected,1),400,6);
options = odeset('RelTol', 3e-14, 'AbsTol',1e-16);
parfor ii=1:length(decoded_states_corrected)
    % disp(ii)
    [~,y1] = ode113(@(t,state) cr3bpStateOnly(state, mu), linspace(0,decoded_periods(ii),400), decoded_states_corrected(ii,1,1:6), options);
    decoded_states_integrated(ii,:,1:6) = y1;
end

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


%% Plot 1 - Show Decoded Family - multiple views, northern and southern, latent
% clc
z_apolune = decoded_states(:,1,3);
cmap = jet(Norb);
fig = figure('Color','w','Units','pixels');
tileWidth = 300;
tileHeight = tileWidth;
fig.Position(3:4) = [3*tileWidth, 3*tileHeight];  % 3x3 grid

% Top Row: Northern Family - 3 Views
% --- View 1: 3D Side View (XZ plane, Y from back) ---
ax1 = subplot(3,3,1);
hold(ax1,'on');
for ii = 1:100:Norb/2-600
    states_out = squeeze(decoded_states(ii,:,:));
    states = squeeze(decoded_states_integrated(ii,:,:));
    orbitColor = cmap(ii,:);
    scatter3(ax1, states_out(:,1), states_out(:,2), states_out(:,3), ...
             20, orbitColor, 'filled');
    plot3(ax1, [states(:,1);states(1,1)], ...
               [states(:,2);states(1,2)], ...
               [states(:,3);states(1,3)], ...
               'Color',orbitColor,'LineWidth',1e-10);
end
for ii = Norb/2-599:40:Norb/2-100
    states_out = squeeze(decoded_states(ii,:,:));
    states = squeeze(decoded_states_integrated(ii,:,:));
    orbitColor = cmap(ii,:);
    scatter3(ax1, states_out(:,1), states_out(:,2), states_out(:,3), ...
             20, orbitColor, 'filled');
    plot3(ax1, [states(:,1);states(1,1)], ...
               [states(:,2);states(1,2)], ...
               [states(:,3);states(1,3)], ...
               'Color',orbitColor,'LineWidth',1e-10);
end
for ii = Norb/2-99:10:Norb/2
    states_out = squeeze(decoded_states(ii,:,:));
    states = squeeze(decoded_states_integrated(ii,:,:));
    orbitColor = cmap(ii,:);
    scatter3(ax1, states_out(:,1), states_out(:,2), states_out(:,3), ...
             20, orbitColor, 'filled');
    plot3(ax1, [states(:,1);states(1,1)], ...
               [states(:,2);states(1,2)], ...
               [states(:,3);states(1,3)], ...
               'Color',orbitColor,'LineWidth',1e-10);
end
scatter3(ax1, l2,0,0,'red','filled','diamond','DisplayName','L_2');
moon_radius = const.moon_R;
[Xm,Ym,Zm] = sphere(50);
Xm = moon_radius*Xm + (1-mu);
Ym = moon_radius*Ym;
Zm = moon_radius*Zm;
surf(ax1, Xm,Ym,Zm, 'FaceColor','k','EdgeColor','none','DisplayName','Moon');
xlabel(ax1,'X','Interpreter','latex','FontSize',20);
ylabel(ax1,'Y','Interpreter','latex','FontSize',20);
zlabel(ax1,'Z','Interpreter','latex','FontSize',20);
set(ax1, 'LineWidth', 2, 'FontSize', 20, 'FontWeight', 'bold', 'TickLabelInterpreter', 'latex');
axis(ax1,'equal');
xlim(ax1,[0.95 1.25]); ylim(ax1,[-0.15 0.15]); zlim(ax1,[-0.08 0.23]);
view(ax1, [270 0]);
daspect(ax1,[1 1 1]);

% --- Copy to View 2: XY Plane (Top View) ---
ax2 = subplot(3,3,2);
allKids = get(ax1,'Children');
copyobj(allKids, ax2);
view(ax2, [0 0]);
xlabel(ax2,'X','Interpreter','latex','FontSize',20);
ylabel(ax2,'Y','Interpreter','latex','FontSize',20);
zlabel(ax2,'Z','Interpreter','latex','FontSize',20);
set(ax2, 'LineWidth', 2, 'FontSize', 20, 'FontWeight', 'bold', 'TickLabelInterpreter', 'latex');
axis(ax2,'equal');
xlim(ax2, xlim(ax1)); ylim(ax2, ylim(ax1)); zlim(ax2, zlim(ax1));
daspect(ax2,[1 1 1]);

% --- Copy to View 3: XZ Plane (Side View from +Y) ---
ax3 = subplot(3,3,3);
copyobj(allKids, ax3);
view(ax3, [0 90]);
xlabel(ax3,'X','Interpreter','latex','FontSize',20);
ylabel(ax3,'Y','Interpreter','latex','FontSize',20);
zlabel(ax3,'Z','Interpreter','latex','FontSize',20);
set(ax3, 'LineWidth', 2, 'FontSize', 20, 'FontWeight', 'bold', 'TickLabelInterpreter', 'latex');
axis(ax3,'equal');
xlim(ax3, xlim(ax1)); ylim(ax3, ylim(ax1)); zlim(ax3, zlim(ax1));
daspect(ax3,[1 1 1]);

% Middle Row: Southern Family - 3 Views
% --- View 4: 3D Side View ---
ax4 = subplot(3,3,4);
hold(ax4,'on');
for ii = Norb-599:100:Norb
    states_out = squeeze(decoded_states(ii,:,:));
    states = squeeze(decoded_states_integrated(ii,:,:));
    orbitColor = cmap(ii,:);
    scatter3(ax4, states_out(:,1), states_out(:,2), states_out(:,3), ...
             20, orbitColor, 'filled');
    plot3(ax4, [states(:,1);states(1,1)], ...
               [states(:,2);states(1,2)], ...
               [states(:,3);states(1,3)], ...
               'Color',orbitColor,'LineWidth',1e-10);
end
for ii = Norb/2+100:40:Norb-600
    states_out = squeeze(decoded_states(ii,:,:));
    states = squeeze(decoded_states_integrated(ii,:,:));
    orbitColor = cmap(ii,:);
    scatter3(ax4, states_out(:,1), states_out(:,2), states_out(:,3), ...
             20, orbitColor, 'filled');
    plot3(ax4, [states(:,1);states(1,1)], ...
               [states(:,2);states(1,2)], ...
               [states(:,3);states(1,3)], ...
               'Color',orbitColor,'LineWidth',1e-10);
end
for ii = Norb/2:10:Norb/2+99
    states_out = squeeze(decoded_states(ii,:,:));
    states = squeeze(decoded_states_integrated(ii,:,:));
    orbitColor = cmap(ii,:);
    scatter3(ax4, states_out(:,1), states_out(:,2), states_out(:,3), ...
             20, orbitColor, 'filled');
    plot3(ax4, [states(:,1);states(1,1)], ...
               [states(:,2);states(1,2)], ...
               [states(:,3);states(1,3)], ...
               'Color',orbitColor,'LineWidth',1e-10);
end
scatter3(ax4, l2,0,0,'red','filled','diamond','DisplayName','L_2');
surf(ax4, Xm,Ym,Zm, 'FaceColor','k','EdgeColor','none','DisplayName','Moon');
xlabel(ax4,'X','Interpreter','latex','FontSize',20);
ylabel(ax4,'Y','Interpreter','latex','FontSize',20);
zlabel(ax4,'Z','Interpreter','latex','FontSize',20);
set(ax4, 'LineWidth', 2, 'FontSize', 20, 'FontWeight', 'bold', 'TickLabelInterpreter', 'latex');
axis(ax4,'equal');
xlim(ax4,[0.95 1.25]); ylim(ax4,[-0.15 0.15]); zlim(ax4,[-0.23 0.08]);
view(ax4, [270 0]);
daspect(ax4,[1 1 1]);

% --- Copy to View 5: XY Plane ---
ax5 = subplot(3,3,5);
allKids_south = get(ax4,'Children');
copyobj(allKids_south, ax5);
view(ax5, [0 0]);
xlabel(ax5,'X','Interpreter','latex','FontSize',20);
ylabel(ax5,'Y','Interpreter','latex','FontSize',20);
zlabel(ax5,'Z','Interpreter','latex','FontSize',20);
set(ax5, 'LineWidth', 2, 'FontSize', 20, 'FontWeight', 'bold', 'TickLabelInterpreter', 'latex');
axis(ax5,'equal');
xlim(ax5, xlim(ax4)); ylim(ax5, ylim(ax4)); zlim(ax5, zlim(ax4));
daspect(ax5,[1 1 1]);

% --- Copy to View 6: XZ Plane ---
ax6 = subplot(3,3,6);
copyobj(allKids_south, ax6);
view(ax6, [0 90]);
xlabel(ax6,'X','Interpreter','latex','FontSize',20);
ylabel(ax6,'Y','Interpreter','latex','FontSize',20);
zlabel(ax6,'Z','Interpreter','latex','FontSize',20);
set(ax6, 'LineWidth', 2, 'FontSize', 20, 'FontWeight', 'bold', 'TickLabelInterpreter', 'latex');
axis(ax6,'equal');
xlim(ax6, xlim(ax4)); ylim(ax6, ylim(ax4)); zlim(ax6, zlim(ax4));
daspect(ax6,[1 1 1]);

% Bottom Row: 2D Latent Variable Plots
% --- Latent vs Period ---
ax7 = subplot(3,3,7);
scatter(ax7, latent_data, decoded_periods, 20, cmap(1:length(latent_data),:), 'filled');
xlabel(ax7,'Latent Variable','FontSize',20,'Interpreter','latex');
ylabel(ax7,'Orbit Period','FontSize',20,'Interpreter','latex');
set(ax7, 'LineWidth', 2, 'FontSize', 20, 'FontWeight', 'bold', 'TickLabelInterpreter', 'latex');
box(ax7, 'on');
grid(ax7, 'on');

% --- Latent vs Apolune Z ---
ax8 = subplot(3,3,8);
scatter(ax8, latent_data, z_apolune, 20, cmap(1:length(latent_data),:), 'filled');
xlabel(ax8,'Latent Variable','FontSize',20,'Interpreter','latex');
ylabel(ax8,'Apolune Z Value','FontSize',20,'Interpreter','latex');
set(ax8, 'LineWidth', 2, 'FontSize', 20, 'FontWeight', 'bold', 'TickLabelInterpreter', 'latex');
box(ax8, 'on');
grid(ax8, 'on');

% --- Latent vs Jacobi Constant ---
ax9 = subplot(3,3,9);
scatter(ax9, latent_data, jacobiConstant(decoded_states(:,1,1:6),mu), 20, cmap(1:length(latent_data),:), 'filled');
xlabel(ax9,'Latent Variable','FontSize',20,'Interpreter','latex');
ylabel(ax9,'Jacobi Constant','FontSize',20,'Interpreter','latex');
set(ax9, 'LineWidth', 2, 'FontSize', 20, 'FontWeight', 'bold', 'TickLabelInterpreter', 'latex');
box(ax9, 'on');
grid(ax9, 'on');


%% Plot 2 - Jacobi Difference histogram with point coloring
abscdiff = abs(cdiff);
fig = figure;
set(fig,"Position",[584   520   745   395]);
tiledlayout(1,3);
nexttile;
min_val = 1e-6;%min(min(abscdiff));
max_val = max(max(abscdiff));
bin_edges = logspace(log10(min_val), log10(max_val), 31);
bin_centers = sqrt(bin_edges(1:end-1) .* bin_edges(2:end)); % geometric centers for log bins
[Nhist, ~] = histcounts(abscdiff, bin_edges, 'Normalization', 'probability');
cmap = parula(numel(bin_centers));
hold on
for i = 1:numel(bin_centers)
    patch('XData', [bin_edges(i) bin_edges(i+1) bin_edges(i+1) bin_edges(i)], ...
          'YData', [0 0 Nhist(i) Nhist(i)], ...
          'FaceColor', cmap(i,:), 'EdgeColor', 'none');
end
set(gca, 'XScale', 'log', 'LineWidth', 2, 'FontSize', 20, ...
         'FontWeight', 'bold', 'TickLabelInterpreter', 'latex');
xlabel('Jacobi Constant Variation','FontSize',26,'Interpreter','latex')
ylabel('Fraction of Test Set','FontSize',26,'Interpreter','latex')
ylim([0 0.15])
xlim([1e-6 1e-2])
clim([min_val max_val])

nexttile;
hold on
for ii = 1:50:Norb/2
    states_out = squeeze(decoded_states(ii,:,:));
    states = squeeze(decoded_states_integrated(ii,:,:));
    
    vals = squeeze(abscdiff(ii,:));
    vals(vals<1e-6) = 1.1e-6;
    idxs = discretize(vals, bin_edges);

    scatter3(states_out(:,1), states_out(:,2), states_out(:,3), ...
             20, cmap(idxs,:), 'filled');
    % 
    % plot3([states(:,1); states(1,1)], [states(:,2); states(1,2)], [states(:,3); states(1,3)], ...
    %       'Color', [0,0,0,0.2], 'LineWidth', 0.5);
end
scatter3(l2,0,0,'red','filled','diamond','DisplayName','L_2');
moon_radius = const.moon_R;
[Xm,Ym,Zm] = sphere(50);
Xm = moon_radius*Xm + (1-mu);
Ym = moon_radius*Ym;
Zm = moon_radius*Zm;
surf(Xm,Ym,Zm, 'FaceColor','k','EdgeColor','none','DisplayName','Moon');
xlabel('X','FontSize',26,'Interpreter','latex')
ylabel('Y','FontSize',26,'Interpreter','latex')
zlabel('Z','FontSize',26,'Interpreter','latex')
axis equal
view([0 0])
set(gca, 'LineWidth', 2, 'FontSize', 20, 'FontWeight', 'bold', 'TickLabelInterpreter', 'latex','ColorScale','log');
xlim([0.95 1.25]); ylim([-0.15 0.15]); zlim([-0.08 0.23]);
colormap(parula)
% colorbar
clim([min_val max_val])

nexttile;
hold on
for ii = 1:50:Norb/2
    states_out = squeeze(decoded_states(ii,:,:));
    states = squeeze(decoded_states_integrated(ii,:,:));
    
    vals = squeeze(abscdiff(ii,:));
    vals(vals<1e-6) = 1.1e-6;
    idxs = discretize(vals, bin_edges);

    scatter3(states_out(:,1), states_out(:,2), states_out(:,3), ...
             20, cmap(idxs,:), 'filled');
    % 
    % plot3([states(:,1); states(1,1)], [states(:,2); states(1,2)], [states(:,3); states(1,3)], ...
    %       'Color', [0,0,0,0.2], 'LineWidth', 0.5);
end
scatter3(l2,0,0,'red','filled','diamond','DisplayName','L_2');
moon_radius = const.moon_R;
[Xm,Ym,Zm] = sphere(50);
Xm = moon_radius*Xm + (1-mu);
Ym = moon_radius*Ym;
Zm = moon_radius*Zm;
surf(Xm,Ym,Zm, 'FaceColor','k','EdgeColor','none','DisplayName','Moon');
xlabel('X','FontSize',26,'Interpreter','latex')
ylabel('Y','FontSize',26,'Interpreter','latex')
zlabel('Z','FontSize',26,'Interpreter','latex')
axis equal
view([-90 0])
set(gca, 'LineWidth', 2, 'FontSize', 20, 'FontWeight', 'bold', 'TickLabelInterpreter', 'latex','ColorScale','log');
xlim([0.95 1.25]); ylim([-0.15 0.15]); zlim([-0.08 0.23]);
colormap(parula)
colorbar
clim([min_val max_val])

fprintf("Median Jacobi Constant Difference %.4e \n",median(abscdiff,'all'))

%% Plot 3 - Initial constraint violations histogram with orbit coloring
fig = figure;
set(fig,"Position",[584   520   745   395]);
tiledlayout(1,2);
nexttile;
min_val = min(initial_constraint_violations(initial_constraint_violations > 0));
max_val = max(initial_constraint_violations);
bin_edges = logspace(log10(min_val), log10(max_val), 31);
bin_centers = sqrt(bin_edges(1:end-1) .* bin_edges(2:end)); % geometric centers for log bins
[Nhist, ~] = histcounts(initial_constraint_violations, bin_edges, 'Normalization', 'probability');
cmap = parula(numel(bin_centers));
hold on
for i = 1:numel(bin_centers)
    patch('XData', [bin_edges(i) bin_edges(i+1) bin_edges(i+1) bin_edges(i)], ...
          'YData', [0 0 Nhist(i) Nhist(i)], ...
          'FaceColor', cmap(i,:), 'EdgeColor', 'none');
end
set(gca, 'XScale', 'log', 'LineWidth', 2, 'FontSize', 20, ...
         'FontWeight', 'bold', 'TickLabelInterpreter', 'latex');
xlabel('Initial Constraint Violation','FontSize',20,'Interpreter','latex')
ylabel('Fraction of Test Set','FontSize',20,'Interpreter','latex')
ylim([0 0.17])
clim([min_val max_val])

nexttile;
hold on
for ii = 2:50:Norb
    states = squeeze(decoded_states_integrated(ii,:,:));
    val = initial_constraint_violations(ii);
    idx = discretize(val,bin_edges);
    plot3([states(:,1); states(1,1)], [states(:,2); states(1,2)], [states(:,3); states(1,3)], ...
          'Color', cmap(idx,:), 'LineWidth', 0.5);
end
xlabel('X','FontSize',20,'Interpreter','latex')
ylabel('Y','FontSize',20,'Interpreter','latex')
zlabel('Z','FontSize',20,'Interpreter','latex')
axis equal
view([0 0])
set(gca, 'LineWidth', 2, 'FontSize', 20, 'FontWeight', 'bold', 'TickLabelInterpreter', 'latex','ColorScale','log');
colormap(parula)
colorbar
clim([min_val max_val])

fprintf("Median Initial Constraint Violation %.4e \n",median(initial_constraint_violations))

%% Plot 4 - Appendix -  Test vs Decoded Corrected Periods histogram with orbit coloring (invertibility measure)
T_diff = abs(test_periods-decoded_period_corrected');
T_diff(T_diff<1e-6) = 1e-6;
fig = figure;
set(fig,"Position",[584   520   745   395]);
tiledlayout(1,2);
nexttile;
min_val = min(T_diff(T_diff > 0));
max_val = max(T_diff);
bin_edges = logspace(log10(min_val), log10(max_val), 31);
bin_centers = sqrt(bin_edges(1:end-1) .* bin_edges(2:end)); % geometric centers for log bins
[Nhist, ~] = histcounts(T_diff, bin_edges, 'Normalization', 'probability');
cmap = parula(numel(bin_centers));
hold on
for i = 1:numel(bin_centers)
    patch('XData', [bin_edges(i) bin_edges(i+1) bin_edges(i+1) bin_edges(i)], ...
          'YData', [0 0 Nhist(i) Nhist(i)], ...
          'FaceColor', cmap(i,:), 'EdgeColor', 'none');
end
set(gca, 'XScale', 'log', 'LineWidth', 2, 'FontSize', 20, ...
         'FontWeight', 'bold', 'TickLabelInterpreter', 'latex');
xlabel('Absolute Period Difference','FontSize',20,'Interpreter','latex')
ylabel('Fraction of Test Set','FontSize',20,'Interpreter','latex')
ylim([0 0.15])

nexttile;
hold on
for ii = 1:50:Norb
    states = squeeze(decoded_states_integrated(ii,:,:));
    val = T_diff(ii);
    idx = discretize(val,bin_edges);
    plot3([states(:,1); states(1,1)], [states(:,2); states(1,2)], [states(:,3); states(1,3)], ...
          'Color', cmap(idx,:), 'LineWidth', 0.5);
end
xlabel('X','FontSize',20,'Interpreter','latex')
ylabel('Y','FontSize',20,'Interpreter','latex')
zlabel('Z','FontSize',20,'Interpreter','latex')
axis equal
view([0 0])
set(gca, 'LineWidth', 2, 'FontSize', 20, 'FontWeight', 'bold', 'TickLabelInterpreter', 'latex','ColorScale','log');
colormap(parula)
cb = colorbar;
clim([min_val max_val])

fprintf("Median Period Difference (Invertibility) %.4e \n",median(T_diff))


%% Information Used In Words
% Size of corrections (total)
% Correction Steps Required
% Period Correction Required

%%%%%%%%%%%%%% TOTAL CORRECTION %%%%%%%%%%%%%%%%%%%%%%%
total_correction = mean(vecnorm(decoded_states_corrected(:,:,:) - decoded_states(:,:,:),2,3),2);
fig = figure;
set(fig,"Position",[584   520   745   395]);
tiledlayout(1,2);
nexttile;
min_val = min(min(total_correction(total_correction > 0)));
max_val = max(max(total_correction));
bin_edges = logspace(log10(min_val), log10(max_val), 31);
bin_centers = sqrt(bin_edges(1:end-1) .* bin_edges(2:end)); % geometric centers for log bins
[Nhist, ~] = histcounts(total_correction, bin_edges, 'Normalization', 'probability');
cmap = parula(numel(bin_centers));
hold on
for i = 1:numel(bin_centers)
    patch('XData', [bin_edges(i) bin_edges(i+1) bin_edges(i+1) bin_edges(i)], ...
          'YData', [0 0 Nhist(i) Nhist(i)], ...
          'FaceColor', cmap(i,:), 'EdgeColor', 'none');
end
set(gca, 'XScale', 'log', 'LineWidth', 2, 'FontSize', 14, ...
         'FontWeight', 'bold', 'TickLabelInterpreter', 'latex');
xlabel('Total Correction Size','FontSize',20,'Interpreter','latex')
ylabel('Fraction of Test Set','FontSize',20,'Interpreter','latex')
ylim([0 0.15])

nexttile;
hold on
for ii = 1:50:Norb
    states = squeeze(decoded_states_integrated(ii,:,:));
    val = total_correction(ii);
    idx = discretize(val,bin_edges);
    plot3([states(:,1); states(1,1)], [states(:,2); states(1,2)], [states(:,3); states(1,3)], ...
          'Color', cmap(idx,:), 'LineWidth', 0.5);
end
xlabel('X','FontSize',20,'Interpreter','latex')
ylabel('Y','FontSize',20,'Interpreter','latex')
zlabel('Z','FontSize',20,'Interpreter','latex')
axis equal
view([0 0])
set(gca, 'LineWidth', 2, 'FontSize', 20, 'FontWeight', 'bold', 'TickLabelInterpreter', 'latex','ColorScale','log');
colormap(parula)
colorbar
clim([min_val max_val])

%%%%%%%%%%%%%%%% Correction Steps Required %%%%%%%%%%%%%%%%%%%%%%%
fig = figure;
set(fig,"Position",[584   520   745   395]);
tiledlayout(1,2);
nexttile;
min_val = min(steps_required);
max_val = max(steps_required);
bin_edges = 2.5:1:11.5;
bin_centers = sqrt(bin_edges(1:end-1) .* bin_edges(2:end)); % geometric centers for log bins
[Nhist, ~] = histcounts(steps_required, bin_edges, 'Normalization', 'probability');
cmap = parula(numel(bin_centers));
norm_vals = ((bin_centers) - (min_val)) / ((max_val) - (min_val));
ymin_log = 3e-4;
hold on
for i = 1:numel(bin_centers)
    patch('XData', [bin_edges(i) bin_edges(i+1) bin_edges(i+1) bin_edges(i)], ...
          'YData', [ymin_log ymin_log Nhist(i) Nhist(i)], ...
          'FaceColor', cmap(i,:), 'EdgeColor', 'none');
end
set(gca, 'LineWidth', 2, 'FontSize', 14, ...
         'FontWeight', 'bold', 'TickLabelInterpreter', 'latex');
xlabel('Correction Steps Required','FontSize',20,'Interpreter','latex')
ylabel('Fraction of Test Set','FontSize',20,'Interpreter','latex')
ylim([4e-4 0.6])
xlim([2.5,11.5]);

nexttile;
hold on
for ii = 1:50:Norb
    states = squeeze(decoded_states_integrated(ii,:,:));
    val = steps_required(ii);
    idx = discretize(val, bin_edges);
    plot3([states(:,1); states(1,1)], [states(:,2); states(1,2)], [states(:,3); states(1,3)], ...
          'Color', cmap(idx,:), 'LineWidth', 0.5);
end
xlabel('X','FontSize',20,'Interpreter','latex')
ylabel('Y','FontSize',20,'Interpreter','latex')
zlabel('Z','FontSize',20,'Interpreter','latex')
axis equal
view([0 0])
set(gca, 'LineWidth', 2, 'FontSize', 20, 'FontWeight', 'bold', 'TickLabelInterpreter', 'latex');
colormap(parula)
colorbar
clim([min_val max_val])

%%%%%%%%%%%%%%%% PERIOD DIFFERENCE %%%%%%%%%%%%%%%%%%%%%%%
T_diff = abs(decoded_periods-decoded_period_corrected');
fig = figure;
set(fig,"Position",[584   520   745   395]);
tiledlayout(1,2);
nexttile;
min_val = min(T_diff(T_diff > 0));
max_val = max(T_diff);
bin_edges = logspace(log10(min_val), log10(max_val), 31);
bin_centers = sqrt(bin_edges(1:end-1) .* bin_edges(2:end)); % geometric centers for log bins
[Nhist, ~] = histcounts(T_diff, bin_edges, 'Normalization', 'probability');
cmap = parula(numel(bin_centers));
hold on
for i = 1:numel(bin_centers)
    patch('XData', [bin_edges(i) bin_edges(i+1) bin_edges(i+1) bin_edges(i)], ...
          'YData', [0 0 Nhist(i) Nhist(i)], ...
          'FaceColor', cmap(i,:), 'EdgeColor', 'none');
end
set(gca, 'XScale', 'log', 'LineWidth', 2, 'FontSize', 14, ...
         'FontWeight', 'bold', 'TickLabelInterpreter', 'latex');
xlabel('Absolute Period Difference','FontSize',20,'Interpreter','latex')
ylabel('Fraction of Test Set','FontSize',20,'Interpreter','latex')
ylim([0 0.15])

nexttile;
hold on
for ii = 1:50:Norb
    states = squeeze(decoded_states_integrated(ii,:,:));
    val = T_diff(ii);
    idx = discretize(val,bin_edges);
    plot3([states(:,1); states(1,1)], [states(:,2); states(1,2)], [states(:,3); states(1,3)], ...
          'Color', cmap(idx,:), 'LineWidth', 0.5);
end
xlabel('X','FontSize',20,'Interpreter','latex')
ylabel('Y','FontSize',20,'Interpreter','latex')
zlabel('Z','FontSize',20,'Interpreter','latex')
axis equal
view([0 0])
set(gca, 'LineWidth', 2, 'FontSize', 20, 'FontWeight', 'bold', 'TickLabelInterpreter', 'latex','ColorScale','log');
colormap(parula)
cb = colorbar;
clim([min_val max_val])

fprintf("Median State Correction %.4e \n",median(total_correction))
fprintf("Median Steps Required %.2e \n",median(steps_required))
fprintf("Median Period Correction %.4e \n",median(T_diff))