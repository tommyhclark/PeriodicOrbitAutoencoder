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
config = readyaml("Halo2VerticalDoedel/config.yaml");

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

zdot_apolune = decoded_states(:,1,6);
 %% Plot 1 - Show Decoded Family - multiple views, northern and southern, latent
% clc
% z_apolune = decoded_states(:,1,3);
% cmap = jet(Norb);
% fig = figure('Color','w','Units','pixels');
% tileWidth = 300;
% tileHeight = tileWidth;
% fig.Position(3:4) = [3*tileWidth, 2*tileHeight];  % 3x3 grid
% 
% % --- View 1: 3D Side View (XZ plane, Y from back) ---
% ax1 = subplot(2,3,1);
% hold(ax1,'on');
% for ii = 1:10:Norb
%     states_out = squeeze(decoded_states(ii,:,:));
%     states = squeeze(decoded_states_integrated(ii,:,:));
%     orbitColor = cmap(ii,:);
%     scatter3(ax1, states_out(:,1), states_out(:,2), states_out(:,3), ...
%              5, "k", 'filled','MarkerFaceAlpha',0.1,'MarkerEdgeAlpha',0.1);
%     plot3(ax1, [states(:,1);states(1,1)], ...
%                [states(:,2);states(1,2)], ...
%                [states(:,3);states(1,3)], ...
%                'Color',[0 0 0 0.01],'LineWidth',1e-10);
% 
% end
% for ii = 3700:150:Norb
%     states_out = squeeze(decoded_states(ii,:,:));
%     states = squeeze(decoded_states_integrated(ii,:,:));
%     orbitColor = cmap(ii,:);
%     scatter3(ax1, states_out(:,1), states_out(:,2), states_out(:,3), ...
%              20, orbitColor, 'filled');
%     plot3(ax1, [states(:,1);states(1,1)], ...
%                [states(:,2);states(1,2)], ...
%                [states(:,3);states(1,3)], ...
%                'Color',orbitColor,'LineWidth',1e-10);
% end
% 
% 
% 
% scatter3(ax1, l2,0,0,'red','filled','diamond','DisplayName','L_2');
% moon_radius = const.moon_R;
% [Xm,Ym,Zm] = sphere(50);
% Xm = moon_radius*Xm + (1-mu);
% Ym = moon_radius*Ym;
% Zm = moon_radius*Zm;
% surf(ax1, Xm,Ym,Zm, 'FaceColor','k','EdgeColor','none','DisplayName','Moon');
% xlabel(ax1,'X','Interpreter','latex','FontSize',20);
% ylabel(ax1,'Y','Interpreter','latex','FontSize',20);
% zlabel(ax1,'Z','Interpreter','latex','FontSize',20);
% set(ax1, 'LineWidth', 2, 'FontSize', 20, 'FontWeight', 'bold', 'TickLabelInterpreter', 'latex');
% axis(ax1,'equal');
% xlim(ax1,[0.8 1.4]); ylim(ax1,[-0.3 0.3]); zlim(ax1,[-0.3 0.3]);
% view(ax1, [270 0]);
% daspect(ax1,[1 1 1]);
% 
% % --- Copy to View 2: XY Plane (Top View) ---
% ax1 = subplot(2,3,2);
% hold(ax1,'on');
% for ii = 2330:50:2900
%     states_out = squeeze(decoded_states(ii,:,:));
%     states = squeeze(decoded_states_integrated(ii,:,:));
%     orbitColor = cmap(ii,:);
%     scatter3(ax1, states_out(:,1), states_out(:,2), states_out(:,3), ...
%              20, orbitColor, 'filled',"");
%     plot3(ax1, [states(:,1);states(1,1)], ...
%                [states(:,2);states(1,2)], ...
%                [states(:,3);states(1,3)], ...
%                'Color',orbitColor,'LineWidth',1e-10);
% end
% 
% scatter3(ax1, l2,0,0,'red','filled','diamond','DisplayName','L_2');
% moon_radius = const.moon_R;
% [Xm,Ym,Zm] = sphere(50);
% Xm = moon_radius*Xm + (1-mu);
% Ym = moon_radius*Ym;
% Zm = moon_radius*Zm;
% surf(ax1, Xm,Ym,Zm, 'FaceColor','k','EdgeColor','none','DisplayName','Moon');
% xlabel(ax1,'X','Interpreter','latex','FontSize',20);
% ylabel(ax1,'Y','Interpreter','latex','FontSize',20);
% zlabel(ax1,'Z','Interpreter','latex','FontSize',20);
% set(ax1, 'LineWidth', 2, 'FontSize', 20, 'FontWeight', 'bold', 'TickLabelInterpreter', 'latex');
% axis(ax1,'equal');
% xlim(ax1,[0.9 1.25]); ylim(ax1,[-0.27 0.27]); zlim(ax1,[-0.15 0.15]);
% view(ax1, [0 90]);
% 
% 
% % --- Copy to View 3: XZ Plane (Side View from +Y) ---
% ax1 = subplot(2,3,3);
% hold(ax1,'on');
% % for ii = 3100:50:3600
% %     states_out = squeeze(decoded_states(ii,:,:));
% %     states = squeeze(decoded_states_integrated(ii,:,:));
% %     orbitColor = cmap(ii,:);
% %     scatter3(ax1, states_out(:,1), states_out(:,2), states_out(:,3), ...
% %              20, orbitColor, 'filled');
% %     plot3(ax1, [states(:,1);states(1,1)], ...
% %                [states(:,2);states(1,2)], ...
% %                [states(:,3);states(1,3)], ...
% %                'Color',orbitColor,'LineWidth',1e-10);
% % end
% 
% for ii = 1:100:2200
%     states_out = squeeze(decoded_states(ii,:,:));
%     states = squeeze(decoded_states_integrated(ii,:,:));
%     orbitColor = cmap(ii,:);
%     scatter3(ax1, states_out(:,1), states_out(:,2), states_out(:,3), ...
%              20, orbitColor, 'filled');
%     plot3(ax1, [states(:,1);states(1,1)], ...
%                [states(:,2);states(1,2)], ...
%                [states(:,3);states(1,3)], ...
%                'Color',orbitColor,'LineWidth',1e-10);
% end
% 
% scatter3(ax1, l2,0,0,'red','filled','diamond','DisplayName','L_2');
% moon_radius = const.moon_R;
% [Xm,Ym,Zm] = sphere(50);
% Xm = moon_radius*Xm + (1-mu);
% Ym = moon_radius*Ym;
% Zm = moon_radius*Zm;
% surf(ax1, Xm,Ym,Zm, 'FaceColor','k','EdgeColor','none','DisplayName','Moon');
% xlabel(ax1,'X','Interpreter','latex','FontSize',20);
% ylabel(ax1,'Y','Interpreter','latex','FontSize',20);
% zlabel(ax1,'Z','Interpreter','latex','FontSize',20);
% set(ax1, 'LineWidth', 2, 'FontSize', 20, 'FontWeight', 'bold', 'TickLabelInterpreter', 'latex');
% axis(ax1,'equal');
% xlim(ax1,[0.95 1.2]); ylim(ax1,[-0.15 0.15]); zlim(ax1,[-0.15 0.22]);
% view(ax1, [330 15]);
% 
% 
% %%%%%%%%% Bottom Row: 2D Latent Variable Plots %%%%%%%%%%%
% % --- Latent vs Period ---
% ax7 = subplot(2,3,4);
% scatter(ax7, latent_data, decoded_periods, 20, cmap(1:length(latent_data),:), 'filled');
% xlabel(ax7,'Latent Variable','FontSize',20,'Interpreter','latex');
% ylabel(ax7,'Orbit Period','FontSize',20,'Interpreter','latex');
% set(ax7, 'LineWidth', 2, 'FontSize', 20, 'FontWeight', 'bold', 'TickLabelInterpreter', 'latex');
% box(ax7, 'on');
% grid(ax7, 'on');
% 
% % --- Latent vs Apolune Z ---
% 
% ax8 = subplot(2,3,5);
% scatter(ax8, latent_data, jacobiConstant(decoded_states(:,1,1:6),mu), 20, cmap(1:length(latent_data),:), 'filled');
% xlabel(ax8,'Latent Variable','FontSize',20,'Interpreter','latex');
% ylabel(ax8,'Jacobi Constant','FontSize',20,'Interpreter','latex');
% set(ax8, 'LineWidth', 2, 'FontSize', 20, 'FontWeight', 'bold', 'TickLabelInterpreter', 'latex');
% box(ax8, 'on');
% grid(ax8, 'on');
% 
% % --- Latent vs Jacobi Constant ---
% ax9 = subplot(2,3,6);
% scatter(ax9, latent_data, zdot_apolune, 20, cmap(1:length(latent_data),:), 'filled');
% xlabel(ax9,'Latent Variable','FontSize',20,'Interpreter','latex');
% ylabel(ax9,'Apolune $\dot{Z}$ Value','FontSize',20,'Interpreter','latex');
% set(ax9, 'LineWidth', 2, 'FontSize', 20, 'FontWeight', 'bold', 'TickLabelInterpreter', 'latex');
% box(ax9, 'on');
% grid(ax9, 'on');


%% Plot 2 - Jacobi Difference histogram with point coloring
abscdiff = abs(cdiff);
% fig = figure;
% set(fig,"Position",[584   520   745   395]);
% tiledlayout(1,3);
% nexttile;
% min_val = 1e-6;%min(min(abscdiff));
% max_val = max(max(abscdiff));
% bin_edges = logspace(log10(min_val), log10(max_val), 31);
% bin_centers = sqrt(bin_edges(1:end-1) .* bin_edges(2:end)); % geometric centers for log bins
% [Nhist, ~] = histcounts(abscdiff, bin_edges, 'Normalization', 'probability');
% cmap = parula(numel(bin_centers));
% hold on
% for i = 1:numel(bin_centers)
%     patch('XData', [bin_edges(i) bin_edges(i+1) bin_edges(i+1) bin_edges(i)], ...
%           'YData', [0 0 Nhist(i) Nhist(i)], ...
%           'FaceColor', cmap(i,:), 'EdgeColor', 'none');
% end
% set(gca, 'XScale', 'log', 'LineWidth', 2, 'FontSize', 14, ...
%          'FontWeight', 'bold', 'TickLabelInterpreter', 'latex');
% xlabel('Jacobi Constant Variation','FontSize',20,'Interpreter','latex')
% ylabel('Fraction of Test Set','FontSize',20,'Interpreter','latex')
% ylim([0 0.15])
% xlim([1e-6 1e-2])
% clim([min_val max_val])
% 
% nexttile;
% hold on
% for ii = 1:50:Norb/2
%     states_out = squeeze(decoded_states(ii,:,:));
%     states = squeeze(decoded_states_integrated(ii,:,:));
% 
%     vals = squeeze(abscdiff(ii,:));
%     vals(vals<1e-6) = 1.1e-6;
%     idxs = discretize(vals, bin_edges);
% 
%     scatter3(states_out(:,1), states_out(:,2), states_out(:,3), ...
%              20, cmap(idxs,:), 'filled');
%     % 
%     % plot3([states(:,1); states(1,1)], [states(:,2); states(1,2)], [states(:,3); states(1,3)], ...
%     %       'Color', [0,0,0,0.2], 'LineWidth', 0.5);
% end
% scatter3(l2,0,0,'red','filled','diamond','DisplayName','L_2');
% moon_radius = const.moon_R;
% [Xm,Ym,Zm] = sphere(50);
% Xm = moon_radius*Xm + (1-mu);
% Ym = moon_radius*Ym;
% Zm = moon_radius*Zm;
% surf(Xm,Ym,Zm, 'FaceColor','k','EdgeColor','none','DisplayName','Moon');
% xlabel('X','FontSize',20,'Interpreter','latex')
% ylabel('Y','FontSize',20,'Interpreter','latex')
% zlabel('Z','FontSize',20,'Interpreter','latex')
% axis equal
% view([0 0])
% set(gca, 'LineWidth', 2, 'FontSize', 20, 'FontWeight', 'bold', 'TickLabelInterpreter', 'latex','ColorScale','log');
% xlim([0.95 1.25]); ylim([-0.15 0.15]); zlim([-0.08 0.23]);
% colormap(parula)
% % colorbar
% clim([min_val max_val])
% 
% nexttile;
% hold on
% for ii = 1:50:Norb/2
%     states_out = squeeze(decoded_states(ii,:,:));
%     states = squeeze(decoded_states_integrated(ii,:,:));
% 
%     vals = squeeze(abscdiff(ii,:));
%     vals(vals<1e-6) = 1.1e-6;
%     idxs = discretize(vals, bin_edges);
% 
%     scatter3(states_out(:,1), states_out(:,2), states_out(:,3), ...
%              20, cmap(idxs,:), 'filled');
%     % 
%     % plot3([states(:,1); states(1,1)], [states(:,2); states(1,2)], [states(:,3); states(1,3)], ...
%     %       'Color', [0,0,0,0.2], 'LineWidth', 0.5);
% end
% scatter3(l2,0,0,'red','filled','diamond','DisplayName','L_2');
% moon_radius = const.moon_R;
% [Xm,Ym,Zm] = sphere(50);
% Xm = moon_radius*Xm + (1-mu);
% Ym = moon_radius*Ym;
% Zm = moon_radius*Zm;
% surf(Xm,Ym,Zm, 'FaceColor','k','EdgeColor','none','DisplayName','Moon');
% xlabel('X','FontSize',20,'Interpreter','latex')
% ylabel('Y','FontSize',20,'Interpreter','latex')
% zlabel('Z','FontSize',20,'Interpreter','latex')
% axis equal
% view([-90 0])
% set(gca, 'LineWidth', 2, 'FontSize', 20, 'FontWeight', 'bold', 'TickLabelInterpreter', 'latex','ColorScale','log');
% xlim([0.95 1.25]); ylim([-0.15 0.15]); zlim([-0.08 0.23]);
% colormap(parula)
% colorbar
% clim([min_val max_val])

fprintf("Median Jacobi Constant Difference %.4e \n",median(abscdiff,'all'))

%% Plot 3 - Initial constraint violations histogram with orbit coloring
% fig = figure;
% set(fig,"Position",[584   520   745   395]);
% tiledlayout(1,2);
% nexttile;
% min_val = min(initial_constraint_violations(initial_constraint_violations > 0));
% max_val = max(initial_constraint_violations);
% bin_edges = logspace(log10(min_val), log10(max_val), 31);
% bin_centers = sqrt(bin_edges(1:end-1) .* bin_edges(2:end)); % geometric centers for log bins
% [Nhist, ~] = histcounts(initial_constraint_violations, bin_edges, 'Normalization', 'probability');
% cmap = parula(numel(bin_centers));
% hold on
% for i = 1:numel(bin_centers)
%     patch('XData', [bin_edges(i) bin_edges(i+1) bin_edges(i+1) bin_edges(i)], ...
%           'YData', [0 0 Nhist(i) Nhist(i)], ...
%           'FaceColor', cmap(i,:), 'EdgeColor', 'none');
% end
% set(gca, 'XScale', 'log', 'LineWidth', 2, 'FontSize', 14, ...
%          'FontWeight', 'bold', 'TickLabelInterpreter', 'latex');
% xlabel('Initial Constraint Violation','FontSize',20,'Interpreter','latex')
% ylabel('Fraction of Test Set','FontSize',20,'Interpreter','latex')
% ylim([0 0.17])
% clim([min_val max_val])
% 
% nexttile;
% hold on
% for ii = 1:50:Norb
%     states = squeeze(decoded_states_integrated(ii,:,:));
%     val = initial_constraint_violations(ii);
%     idx = discretize(val,bin_edges);
%     plot3([states(:,1); states(1,1)], [states(:,2); states(1,2)], [states(:,3); states(1,3)], ...
%           'Color', cmap(idx,:), 'LineWidth', 0.5);
% end
% xlabel('X','FontSize',20,'Interpreter','latex')
% ylabel('Y','FontSize',20,'Interpreter','latex')
% zlabel('Z','FontSize',20,'Interpreter','latex')
% axis equal
% view([0 0])
% set(gca, 'LineWidth', 2, 'FontSize', 20, 'FontWeight', 'bold', 'TickLabelInterpreter', 'latex','ColorScale','log');
% colormap(parula)
% colorbar
% clim([min_val max_val])

fprintf("Median Initial Constraint Violation %.4e \n",median(initial_constraint_violations))

%% Plot 4 - Appendix -  Test vs Decoded Corrected Periods histogram with orbit coloring (invertibility measure)
% clc
T_diff = abs(test_periods-decoded_period_corrected');
% fig = figure;
% set(fig,"Position",[584   520   745   395]);
% tiledlayout(1,2);
% nexttile;
% min_val = min(T_diff(T_diff > 0));
% max_val = max(T_diff);
% bin_edges = logspace(log10(min_val), log10(max_val), 31);
% bin_centers = sqrt(bin_edges(1:end-1) .* bin_edges(2:end)); % geometric centers for log bins
% [Nhist, ~] = histcounts(T_diff, bin_edges, 'Normalization', 'probability');
% cmap = parula(numel(bin_centers));
% hold on
% for i = 1:numel(bin_centers)
%     patch('XData', [bin_edges(i) bin_edges(i+1) bin_edges(i+1) bin_edges(i)], ...
%           'YData', [0 0 Nhist(i) Nhist(i)], ...
%           'FaceColor', cmap(i,:), 'EdgeColor', 'none');
% end
% set(gca, 'XScale', 'log', 'LineWidth', 2, 'FontSize', 14, ...
%          'FontWeight', 'bold', 'TickLabelInterpreter', 'latex');
% xlabel('Absolute Period Difference','FontSize',20,'Interpreter','latex')
% ylabel('Fraction of Test Set','FontSize',20,'Interpreter','latex')
% ylim([0 0.15])
% 
% nexttile;
% hold on
% for ii = 1:50:Norb
%     states = squeeze(decoded_states_integrated(ii,:,:));
%     val = T_diff(ii);
%     idx = discretize(val,bin_edges);
%     plot3([states(:,1); states(1,1)], [states(:,2); states(1,2)], [states(:,3); states(1,3)], ...
%           'Color', cmap(idx,:), 'LineWidth', 0.5);
% end
% xlabel('X','FontSize',20,'Interpreter','latex')
% ylabel('Y','FontSize',20,'Interpreter','latex')
% zlabel('Z','FontSize',20,'Interpreter','latex')
% axis equal
% view([0 0])
% set(gca, 'LineWidth', 2, 'FontSize', 20, 'FontWeight', 'bold', 'TickLabelInterpreter', 'latex','ColorScale','log');
% colormap(parula)
% cb = colorbar;
% clim([min_val max_val])

fprintf("Median Period Difference (Invertibility) %.4e \n",median(T_diff))


%% Information Used In Words
% Size of corrections (total)
% Correction Steps Required
% Period Correction Required

%%%%%%%%%%%%%% TOTAL CORRECTION %%%%%%%%%%%%%%%%%%%%%%%
total_correction = mean(vecnorm(decoded_states_corrected(:,:,:) - decoded_states(:,:,:),2,3),2);
% fig = figure;
% set(fig,"Position",[584   520   745   395]);
% tiledlayout(1,2);
% nexttile;
% min_val = min(min(total_correction(total_correction > 0)));
% max_val = max(max(total_correction));
% bin_edges = logspace(log10(min_val), log10(max_val), 31);
% bin_centers = sqrt(bin_edges(1:end-1) .* bin_edges(2:end)); % geometric centers for log bins
% [Nhist, ~] = histcounts(total_correction, bin_edges, 'Normalization', 'probability');
% cmap = parula(numel(bin_centers));
% hold on
% for i = 1:numel(bin_centers)
%     patch('XData', [bin_edges(i) bin_edges(i+1) bin_edges(i+1) bin_edges(i)], ...
%           'YData', [0 0 Nhist(i) Nhist(i)], ...
%           'FaceColor', cmap(i,:), 'EdgeColor', 'none');
% end
% set(gca, 'XScale', 'log', 'LineWidth', 2, 'FontSize', 14, ...
%          'FontWeight', 'bold', 'TickLabelInterpreter', 'latex');
% xlabel('Total Correction Size','FontSize',20,'Interpreter','latex')
% ylabel('Fraction of Test Set','FontSize',20,'Interpreter','latex')
% ylim([0 0.15])
% 
% nexttile;
% hold on
% for ii = 1:50:Norb
%     states = squeeze(decoded_states_integrated(ii,:,:));
%     val = total_correction(ii);
%     idx = discretize(val,bin_edges);
%     plot3([states(:,1); states(1,1)], [states(:,2); states(1,2)], [states(:,3); states(1,3)], ...
%           'Color', cmap(idx,:), 'LineWidth', 0.5);
% end
% xlabel('X','FontSize',20,'Interpreter','latex')
% ylabel('Y','FontSize',20,'Interpreter','latex')
% zlabel('Z','FontSize',20,'Interpreter','latex')
% axis equal
% view([0 0])
% set(gca, 'LineWidth', 2, 'FontSize', 20, 'FontWeight', 'bold', 'TickLabelInterpreter', 'latex','ColorScale','log');
% colormap(parula)
% colorbar
% clim([min_val max_val])

%%%%%%%%%%%%%%%% Correction Steps Required %%%%%%%%%%%%%%%%%%%%%%%
% fig = figure;
% set(fig,"Position",[584   520   745   395]);
% tiledlayout(1,2);
% nexttile;
% min_val = min(steps_required);
% max_val = max(steps_required);
% bin_edges = 2.5:1:11.5;
% bin_centers = sqrt(bin_edges(1:end-1) .* bin_edges(2:end)); % geometric centers for log bins
% [Nhist, ~] = histcounts(steps_required, bin_edges, 'Normalization', 'probability');
% cmap = parula(numel(bin_centers));
% norm_vals = ((bin_centers) - (min_val)) / ((max_val) - (min_val));
% ymin_log = 3e-4;
% hold on
% for i = 1:numel(bin_centers)
%     patch('XData', [bin_edges(i) bin_edges(i+1) bin_edges(i+1) bin_edges(i)], ...
%           'YData', [ymin_log ymin_log Nhist(i) Nhist(i)], ...
%           'FaceColor', cmap(i,:), 'EdgeColor', 'none');
% end
% set(gca, 'LineWidth', 2, 'FontSize', 14, ...
%          'FontWeight', 'bold', 'TickLabelInterpreter', 'latex');
% xlabel('Correction Steps Required','FontSize',20,'Interpreter','latex')
% ylabel('Fraction of Test Set','FontSize',20,'Interpreter','latex')
% ylim([4e-4 0.6])
% xlim([2.5,11.5]);
% 
% nexttile;
% hold on
% for ii = 1:50:Norb
%     states = squeeze(decoded_states_integrated(ii,:,:));
%     val = steps_required(ii);
%     idx = discretize(val, bin_edges);
%     plot3([states(:,1); states(1,1)], [states(:,2); states(1,2)], [states(:,3); states(1,3)], ...
%           'Color', cmap(idx,:), 'LineWidth', 0.5);
% end
% xlabel('X','FontSize',20,'Interpreter','latex')
% ylabel('Y','FontSize',20,'Interpreter','latex')
% zlabel('Z','FontSize',20,'Interpreter','latex')
% axis equal
% view([0 0])
% set(gca, 'LineWidth', 2, 'FontSize', 20, 'FontWeight', 'bold', 'TickLabelInterpreter', 'latex');
% colormap(parula)
% colorbar
% clim([min_val max_val])

%%%%%%%%%%%%%%%% PERIOD DIFFERENCE %%%%%%%%%%%%%%%%%%%%%%%
T_diff = abs(decoded_periods-decoded_period_corrected');
% fig = figure;
% set(fig,"Position",[584   520   745   395]);
% tiledlayout(1,2);
% nexttile;
% min_val = min(T_diff(T_diff > 0));
% max_val = max(T_diff);
% bin_edges = logspace(log10(min_val), log10(max_val), 31);
% bin_centers = sqrt(bin_edges(1:end-1) .* bin_edges(2:end)); % geometric centers for log bins
% [Nhist, ~] = histcounts(T_diff, bin_edges, 'Normalization', 'probability');
% cmap = parula(numel(bin_centers));
% hold on
% for i = 1:numel(bin_centers)
%     patch('XData', [bin_edges(i) bin_edges(i+1) bin_edges(i+1) bin_edges(i)], ...
%           'YData', [0 0 Nhist(i) Nhist(i)], ...
%           'FaceColor', cmap(i,:), 'EdgeColor', 'none');
% end
% set(gca, 'XScale', 'log', 'LineWidth', 2, 'FontSize', 14, ...
%          'FontWeight', 'bold', 'TickLabelInterpreter', 'latex');
% xlabel('Absolute Period Difference','FontSize',20,'Interpreter','latex')
% ylabel('Fraction of Test Set','FontSize',20,'Interpreter','latex')
% ylim([0 0.15])
% 
% nexttile;
% hold on
% for ii = 1:50:Norb
%     states = squeeze(decoded_states_integrated(ii,:,:));
%     val = T_diff(ii);
%     idx = discretize(val,bin_edges);
%     plot3([states(:,1); states(1,1)], [states(:,2); states(1,2)], [states(:,3); states(1,3)], ...
%           'Color', cmap(idx,:), 'LineWidth', 0.5);
% end
% xlabel('X','FontSize',20,'Interpreter','latex')
% ylabel('Y','FontSize',20,'Interpreter','latex')
% zlabel('Z','FontSize',20,'Interpreter','latex')
% axis equal
% view([0 0])
% set(gca, 'LineWidth', 2, 'FontSize', 20, 'FontWeight', 'bold', 'TickLabelInterpreter', 'latex','ColorScale','log');
% colormap(parula)
% cb = colorbar;
% clim([min_val max_val])

fprintf("Median State Correction %.4e \n",median(total_correction))
fprintf("Median Steps Required %.2e \n",median(steps_required))
fprintf("Median Period Correction %.4e \n",median(T_diff))

%%
cmap = zeros(Norb, 3);

% Define RGB triplets
blue   = [0.1, 0.1, 0.9];
red    = [0.9, 0.1, 0.1];
yellow = [0.9, 0.9, 0.05];
green  = [0.1, 0.9, 0.1];

% Assign colors to the specified ranges
% Blue for indices 1 to 2199
cmap(1:2140, :) = repmat(blue, 2140, 1);

% Red for indices 2200 to 2999
cmap(2141:3101, :) = repmat(red, 3101 - 2141 + 1, 1);

% Yellow for indices 3000 to 3600
cmap(3102:3650, :) = repmat(yellow, 3650 - 3102 + 1, 1);

% Green for indices 3700 to Norb
cmap(3651:Norb, :) = repmat(green, Norb - 3651 + 1, 1);
%% Plot 1 - Show Decoded Family - multiple views, northern and southern, latent
% clc
z_apolune = decoded_states(:,1,3);
% cmap = jet(Norb);
fig = figure('Color','w','Units','pixels');
tileWidth = 300;
tileHeight = tileWidth;
fig.Position(3:4) = [4*tileWidth, 2*tileHeight];  % 4x2 grid adjustment

axs_obj = gobjects(7,1);
axs_pos = [...
    0, 0.5, 0.25, 0.5; ...     % Top 1
    0.25, 0.5, 0.25, 0.5; ...  % Top 2 (inserted)
    0.5, 0.5, 0.25, 0.5; ...   % Top 3
    0.725, 0.5, 0.25, 0.5; ...  % Top 4
    0.05, 0, 0.25, 0.5; ...   % Bottom 1 (centered)
    0.3875, 0, 0.25, 0.5; ...   % Bottom 2 (centered)
    0.725, 0, 0.25, 0.5 ...    % Bottom 3 (centered)
];
for ii = 1:7
    axs_obj(ii) = axes(fig);
    set(axs_obj(ii),'OuterPosition',axs_pos(ii,:));
end

% --- View 1: 3D Side View (XZ plane, Y from back) ---
ax1 = axs_obj(1);
hold(ax1,'on');
for ii = 1:75:3300
    states_out = squeeze(decoded_states(ii,:,:));
    states = squeeze(decoded_states_integrated(ii,:,:));
    orbitColor = cmap(ii,:);
    scatter3(ax1, states_out(:,1), states_out(:,2), states_out(:,3), ...
             3, "k", 'filled','MarkerFaceAlpha',0.1,'MarkerEdgeAlpha',0.1);
    plot3(ax1, [states(:,1);states(1,1)], ...
               [states(:,2);states(1,2)], ...
               [states(:,3);states(1,3)], ...
               'Color',[0 0 0 0.05],'LineWidth',1e-10);

end
for ii = 3300:10:3699
    states_out = squeeze(decoded_states(ii,:,:));
    states = squeeze(decoded_states_integrated(ii,:,:));
    orbitColor = cmap(ii,:);
    scatter3(ax1, states_out(:,1), states_out(:,2), states_out(:,3), ...
             3, "k", 'filled','MarkerFaceAlpha',0.1,'MarkerEdgeAlpha',0.1);
    plot3(ax1, [states(:,1);states(1,1)], ...
               [states(:,2);states(1,2)], ...
               [states(:,3);states(1,3)], ...
               'Color',[0 0 0 0.05],'LineWidth',1e-10);

end
for ii = 3700:150:Norb
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
xlim(ax1,[0.8 1.4]); ylim(ax1,[-0.26 0.26]); zlim(ax1,[-0.26 0.26]);
view(ax1, [90 0]);
daspect(ax1,[1 1 1]);

% --- Inserted View (from commented code in original View 3) ---
ax_inserted = axs_obj(2);
hold(ax_inserted,'on');
for ii = 1:30:3099
    states_out = squeeze(decoded_states(ii,:,:));
    states = squeeze(decoded_states_integrated(ii,:,:));
    orbitColor = cmap(ii,:);
    scatter3(ax_inserted, states_out(:,1), states_out(:,2), states_out(:,3), ...
             3, "k", 'filled','MarkerFaceAlpha',0.1,'MarkerEdgeAlpha',0.1);
    plot3(ax_inserted, [states(:,1);states(1,1)], ...
               [states(:,2);states(1,2)], ...
               [states(:,3);states(1,3)], ...
               'Color',[0 0 0 0.05],'LineWidth',1e-10);

end
for ii = 3650:30:Norb
    states_out = squeeze(decoded_states(ii,:,:));
    states = squeeze(decoded_states_integrated(ii,:,:));
    orbitColor = cmap(ii,:);
    scatter3(ax_inserted, states_out(:,1), states_out(:,2), states_out(:,3), ...
             3, "k", 'filled','MarkerFaceAlpha',0.1,'MarkerEdgeAlpha',0.1);
    plot3(ax_inserted, [states(:,1);states(1,1)], ...
               [states(:,2);states(1,2)], ...
               [states(:,3);states(1,3)], ...
               'Color',[0 0 0 0.05],'LineWidth',1e-10);

end
for ii = 3105:50:3600
    states_out = squeeze(decoded_states(ii,:,:));
    states = squeeze(decoded_states_integrated(ii,:,:));
    orbitColor = cmap(ii,:);
    scatter3(ax_inserted, states_out(:,1), states_out(:,2), states_out(:,3), ...
             20, orbitColor, 'filled');
    plot3(ax_inserted, [states(:,1);states(1,1)], ...
               [states(:,2);states(1,2)], ...
               [states(:,3);states(1,3)], ...
               'Color',orbitColor,'LineWidth',1e-10);
end

scatter3(ax_inserted, l2,0,0,'red','filled','diamond','DisplayName','L_2');
moon_radius = const.moon_R;
[Xm,Ym,Zm] = sphere(50);
Xm = moon_radius*Xm + (1-mu);
Ym = moon_radius*Ym;
Zm = moon_radius*Zm;
surf(ax_inserted, Xm,Ym,Zm, 'FaceColor','k','EdgeColor','none','DisplayName','Moon');
xlabel(ax_inserted,'X','Interpreter','latex','FontSize',20);
ylabel(ax_inserted,'Y','Interpreter','latex','FontSize',20);
zlabel(ax_inserted,'Z','Interpreter','latex','FontSize',20);
set(ax_inserted, 'LineWidth', 2, 'FontSize', 20, 'FontWeight', 'bold', 'TickLabelInterpreter', 'latex');
axis(ax_inserted,'equal');
xlim(ax_inserted,[0.8 1.4]); ylim(ax_inserted,[-0.28 0.28]); zlim(ax_inserted,[-0.28 0.28]);  % Using original View 3 limits; adjust if needed
view(ax_inserted, [270 0]);
daspect(ax_inserted,[1 1 1]);

% --- Original View 2: XY Plane (Top View) ---
ax2 = axs_obj(3);
hold(ax2,'on');
for ii = 1:75:1990
    states_out = squeeze(decoded_states(ii,:,:));
    states = squeeze(decoded_states_integrated(ii,:,:));
    orbitColor = cmap(ii,:);
    scatter3(ax2, states_out(:,1), states_out(:,2), states_out(:,3), ...
             3, "k", 'filled','MarkerFaceAlpha',0.1,'MarkerEdgeAlpha',0.1);
    plot3(ax2, [states(:,1);states(1,1)], ...
               [states(:,2);states(1,2)], ...
               [states(:,3);states(1,3)], ...
               'Color',[0 0 0 0.05],'LineWidth',1e-10);

end
for ii = 3150:30:Norb
    states_out = squeeze(decoded_states(ii,:,:));
    states = squeeze(decoded_states_integrated(ii,:,:));
    orbitColor = cmap(ii,:);
    scatter3(ax2, states_out(:,1), states_out(:,2), states_out(:,3), ...
             3, "k", 'filled','MarkerFaceAlpha',0.1,'MarkerEdgeAlpha',0.1);
    plot3(ax2, [states(:,1);states(1,1)], ...
               [states(:,2);states(1,2)], ...
               [states(:,3);states(1,3)], ...
               'Color',[0 0 0 0.05],'LineWidth',1e-10);

end
for ii = 2200:50:2950
    states_out = squeeze(decoded_states(ii,:,:));
    states = squeeze(decoded_states_integrated(ii,:,:));
    orbitColor = cmap(ii,:);
    scatter3(ax2, states_out(:,1), states_out(:,2), states_out(:,3), ...
             20, orbitColor, 'filled');
    plot3(ax2, [states(:,1);states(1,1)], ...
               [states(:,2);states(1,2)], ...
               [states(:,3);states(1,3)], ...
               'Color',orbitColor,'LineWidth',1e-10);
end

scatter3(ax2, l2,0,0,'red','filled','diamond','DisplayName','L_2');
moon_radius = const.moon_R;
[Xm,Ym,Zm] = sphere(50);
Xm = moon_radius*Xm + (1-mu);
Ym = moon_radius*Ym;
Zm = moon_radius*Zm;
surf(ax2, Xm,Ym,Zm, 'FaceColor','k','EdgeColor','none','DisplayName','Moon');
xlabel(ax2,'X','Interpreter','latex','FontSize',20);
ylabel(ax2,'Y','Interpreter','latex','FontSize',20);
zlabel(ax2,'Z','Interpreter','latex','FontSize',20);
set(ax2, 'LineWidth', 2, 'FontSize', 20, 'FontWeight', 'bold', 'TickLabelInterpreter', 'latex');
axis(ax2,'equal');
xlim(ax2,[0.8 1.4]); ylim(ax2,[-0.3 0.3]); zlim(ax2,[-0.26 0.26]);
view(ax2, [90 90]);
daspect(ax2,[1 1 1]);

% --- Original View 3: XZ Plane (Side View from +Y), now with active loop ---
ax3 = axs_obj(4);
hold(ax3,'on');
for ii = 2200:30:Norb
    states_out = squeeze(decoded_states(ii,:,:));
    states = squeeze(decoded_states_integrated(ii,:,:));
    orbitColor = cmap(ii,:);
    scatter3(ax3, states_out(:,1), states_out(:,2), states_out(:,3), ...
             3, "k", 'filled','MarkerFaceAlpha',0.1,'MarkerEdgeAlpha',0.1);
    plot3(ax3, [states(:,1);states(1,1)], ...
               [states(:,2);states(1,2)], ...
               [states(:,3);states(1,3)], ...
               'Color',[0 0 0 0.05],'LineWidth',1e-10);

end
for ii = 1:100:1800
    states_out = squeeze(decoded_states(ii,:,:));
    states = squeeze(decoded_states_integrated(ii,:,:));
    orbitColor = cmap(ii,:);
    scatter3(ax3, states_out(:,1), states_out(:,2), states_out(:,3), ...
             20, orbitColor, 'filled');
    plot3(ax3, [states(:,1);states(1,1)], ...
               [states(:,2);states(1,2)], ...
               [states(:,3);states(1,3)], ...
               'Color',orbitColor,'LineWidth',1e-10);
end

for ii = 1817:25:2140
    states_out = squeeze(decoded_states(ii,:,:));
    states = squeeze(decoded_states_integrated(ii,:,:));
    orbitColor = cmap(ii,:);
    scatter3(ax3, states_out(:,1), states_out(:,2), states_out(:,3), ...
             20, orbitColor, 'filled');
    plot3(ax3, [states(:,1);states(1,1)], ...
               [states(:,2);states(1,2)], ...
               [states(:,3);states(1,3)], ...
               'Color',orbitColor,'LineWidth',1e-10);
end

scatter3(ax3, l2,0,0,'red','filled','diamond','DisplayName','L_2');
moon_radius = const.moon_R;
[Xm,Ym,Zm] = sphere(50);
Xm = moon_radius*Xm + (1-mu);
Ym = moon_radius*Ym;
Zm = moon_radius*Zm;
surf(ax3, Xm,Ym,Zm, 'FaceColor','k','EdgeColor','none','DisplayName','Moon');
xlabel(ax3,'X','Interpreter','latex','FontSize',20);
ylabel(ax3,'Y','Interpreter','latex','FontSize',20);
zlabel(ax3,'Z','Interpreter','latex','FontSize',20);
set(ax3, 'LineWidth', 2, 'FontSize', 20, 'FontWeight', 'bold', 'TickLabelInterpreter', 'latex');
axis(ax3,'equal');
xlim(ax3,[0.95 1.2]); ylim(ax3,[-0.15 0.15]); zlim(ax3,[-0.18 0.22]);
view(ax3, [330 0]);
daspect(ax3,[1 1 1]);

%%%%%%%%% Bottom Row: 2D Latent Variable Plots %%%%%%%%%%%
% --- Latent vs Period ---
ax7 = axs_obj(5);
scatter(ax7, latent_data, decoded_periods, 20, cmap(1:length(latent_data),:), 'filled');
xlabel(ax7,'Latent Variable','FontSize',20,'Interpreter','latex');
ylabel(ax7,'Orbit Period','FontSize',20,'Interpreter','latex');
set(ax7, 'LineWidth', 2, 'FontSize', 20, 'FontWeight', 'bold', 'TickLabelInterpreter', 'latex');
box(ax7, 'on');
grid(ax7, 'on');

% --- Latent vs Apolune Z ---
ax8 = axs_obj(6);
scatter(ax8, latent_data, jacobiConstant(decoded_states(:,1,1:6),mu), 20, cmap(1:length(latent_data),:), 'filled');
xlabel(ax8,'Latent Variable','FontSize',20,'Interpreter','latex');
ylabel(ax8,'Jacobi Constant','FontSize',20,'Interpreter','latex');
set(ax8, 'LineWidth', 2, 'FontSize', 20, 'FontWeight', 'bold', 'TickLabelInterpreter', 'latex');
box(ax8, 'on');
grid(ax8, 'on');

% --- Latent vs Jacobi Constant ---
ax9 = axs_obj(7);
scatter(ax9, latent_data, zdot_apolune, 20, cmap(1:length(latent_data),:), 'filled');
xlabel(ax9,'Latent Variable','FontSize',20,'Interpreter','latex');
ylabel(ax9,'Apolune $\dot{Z}$ Value','FontSize',20,'Interpreter','latex');
set(ax9, 'LineWidth', 2, 'FontSize', 20, 'FontWeight', 'bold', 'TickLabelInterpreter', 'latex');
box(ax9, 'on');
grid(ax9, 'on');

%%
cs = zeros(size(test_states,1));
for ii = 1:size(test_states,1)
    cs(ii) = mean(jacobiConstant(squeeze(test_states(ii,:,1:6)),mu));
end