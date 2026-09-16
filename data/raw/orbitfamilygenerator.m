%% Orbit Family Data Generator
% Generates Single Shooting Guesses for families via continuation.
clear all
clc
clear path
addpath("src/utils")
const = load('data/cr3bp_constants.mat','const').const;
mu = const.mu;
%% Beginning with Halo Orbit Family

nrho_initial_pt = [1.011072706185610   0.000000000000000   0.173186749582863   0.000000000000072  -0.078105372772181  -0.000000000000819]';
nrho_initial_t = 1.363739586209268;
[t, x, STMs, ~, ~, ~, DF] = differentialCorrectionMCF(nrho_initial_pt, nrho_initial_t, mu, false, 2); % z0 free method
nhat = null(DF)
%% Compute Halos
% Bifurcation stops at Lyapunov with stability tolerance 1e-6
moon_R = const.moon_R;
earth_R = const.earth_R;
bifur_tol = 1e-5;
[torbsHaloN,xorbsHaloN, MorbsHaloN] = pseudoArcLengthContinuation(x, t, mu, nhat, 10000, moon_R, earth_R, squeeze(STMs(end,:,:)),5*sqrt(2)*1e-4, bifur_tol);
%%
figure;
hold on
for ii = 1:100:length(xorbsHaloN)
    x = xorbsHaloN{ii};
    scatter3(x(:,1),x(:,2),x(:,3))
end
%%
% Halo Family N
halodata = NaN(length(xorbsHaloN),7);
for ii = 1:length(xorbsHaloN)
    x = xorbsHaloN{ii};
    halodata(ii,1:6) = x(1,1:6);
    halodata(ii,7) = torbsHaloN{ii}(end);
end
writematrix(halodata,"data/raw/L2_HaloN_Impact_Lyap_5sqrt2em4")

% Halo Family S
halodata = NaN(length(xorbsHaloN),7);
for ii = 1:length(xorbsHaloN)
    x = xorbsHaloN{ii};
    halodata(ii,1:6) = x(1,1:6);
    halodata(ii,3) = -halodata(ii,3);
    halodata(ii,6) = -halodata(ii,6);
    halodata(ii,7) = torbsHaloN{ii}(end);
end
writematrix(halodata,"data/raw/L2_HaloS_Impact_Lyap_5sqrt2em4")

%% Create Halo planar supplement
[t, x, STMs, ~, ~, ~, DF] = differentialCorrectionMCF(xorbsHaloN{end}(1,:)', torbsHaloN{end}(end), mu, false, 2); % z0 free method
nhat = null(DF);
nhat = nhat(:,1);
moon_R = const.moon_R;
bifur_tol = 1e-6;
[torbsHaloNsup,xorbsHaloNsup, MorbsHaloNsup] = pseudoArcLengthContinuation(x, t, mu, nhat, 2000, moon_R, earth_R, squeeze(STMs(end,:,:)),1e-5, bifur_tol);

%% Compute Lyapunovs (from Halo to Axial)
halo_lyap_bifur_guess = xorbsHaloN{end}(1,:)';
halo_lyap_bifur_guess_t = torbsHaloN{end}(end);
[t, x, STMs, ~, ~, ~, DF] = differentialCorrectionMCF(halo_lyap_bifur_guess, halo_lyap_bifur_guess_t, mu, false,1); % z0 = 0 method
nhat = null(DF)

% Bifurcation tolerance 1e-3 stops at axial
bifur_tol = 1e-3;
[torbsLyap,xorbsLyap, MorbsLyap] = pseudoArcLengthContinuation(x, t, mu, nhat, 4000, moon_R, earth_R, squeeze(STMs(end,:,:)), 5*sqrt(2)*1e-4, bifur_tol);
%%
% Lyap Family
lyapdata = NaN(length(xorbsLyap),7);
for ii = 1:length(xorbsLyap)
    x = xorbsLyap{ii};
    lyapdata(ii,1:6) = x(1,1:6);
    lyapdata(ii,7) = torbsLyap{ii}(end);
end
writematrix(lyapdata,"data/raw/L2_Lyap_Halo_Axial_5sqrt2em4")
%%
figure;
hold on
for ii = 1:50:length(xorbsLyap)
    x = xorbsLyap{ii};
    scatter3(x(:,1),x(:,2),x(:,3))
end

%% Compute Axials from (Lyapunov to Vertical)
clc
% Initial Guess from Lyapunov family
% init_guess = [1.0301513;0;0;0;0.7030025; 0.1552945];
init_guess = [1.2199770;0;0;0;-0.42749325;8.842848778e-4];
T = 4.3105100140285977;
[t, x, STMs, ~, ~, ~, DF] = differentialCorrectionMCF(init_guess, T, mu, false,0);
nhat = null(DF);
nhat = nhat(:,1);

% Bifurcation tolerance 1e-4 stops at vertical
bifur_tol = 1e-4;
[torbsAxial,xorbsAxial, MorbsAxial] = pseudoArcLengthContinuation(x, t, mu, nhat, 10000, 0, earth_R, squeeze(STMs(end,:,:)),5*sqrt(2)*1e-4, bifur_tol);


%%
figure;
hold on
for ii = 1:50:length(xorbsAxial)
    x = xorbsAxial{ii};
    scatter3(x(:,1),x(:,2),x(:,3))
end
%% Compute Vertical to small
clc
[V,D] = eig(MorbsAxial{end})
nhat = real(V(:,4))
init_guess = xorbsAxial{end}(1,:)'+nhat/norm(nhat)*5e-3;
T = torbsAxial{end}(end);
[t, x, STMs, ~, ~, ~, DF] = differentialCorrectionMCF(init_guess, T, mu, false,0);
nhat = null(DF);


bifur_tol = 1e-5;
[torbsVert,xorbsVert, MorbsVert] = pseudoArcLengthContinuation(x, t, mu, -nhat, 2000, moon_R, earth_R, squeeze(STMs(end,:,:)),5*sqrt(2)*1e-4, bifur_tol);

%% Compute Vertical to B2
clc
[V,D] = eig(MorbsAxial{end})
nhat = real(V(:,4))
init_guess = xorbsAxial{end}(1,:)'+nhat/norm(nhat)*5e-3;
T = torbsAxial{end}(end);
[t, x, STMs, ~, ~, ~, DF] = differentialCorrectionMCF(init_guess, T, mu, false,0);
nhat = null(DF);


bifur_tol = 1e-5;
[torbsVert,xorbsVert, MorbsVert]  = pseudoArcLengthContinuation(x, t, mu, nhat, 10000, moon_R, earth_R, squeeze(STMs(end,:,:)),5*sqrt(2)*1e-4,bifur_tol);

%%
figure;
hold on
for ii = 1:50:length(xorbsVert)
    x = xorbsVert{ii};
    scatter3(x(:,1),x(:,2),x(:,3))
end

%% Vert into B2 family
clc
[V,D] = eig(MorbsVert{end})
nhat = real(V(:,4))
init_guess = xorbsVert{end}(1,:)'+nhat/norm(nhat)*5e-3;
T = torbsVert{end}(end);
[t, x, STMs, ~, ~, ~, DF] = differentialCorrectionMCF(init_guess, T, mu, false);
nhat = null(DF);

[torbsB2,xorbsB2, MorbsB2] = pseudoArcLengthContinuation(x, t, mu, nhat, 1000, moon_R, earth_R, squeeze(STMs(end,:,:)),3e-3);

xorbs = xorbsB2;
colors = winter(length(xorbs));
figure;
hold on;
x = xorbs{1};
h1 = plot3(x(:,1), x(:,2), x(:,3), 'Color', colors(1,:), 'LineWidth', 1.5);
for i = 1:10:length(xorbs)
    x = xorbs{i};
    plot3(x(:,1), x(:,2), x(:,3), 'Color', colors(i,:), 'LineWidth', 1.2);
    scatter3(x(1,1), x(1,2), x(1,3),"red");
end
colorbar;
axis equal
%% Save data files

% Halo Family N
halodata = NaN(length(xorbsHaloN),7);
for ii = 1:length(xorbsHaloN)
    x = xorbsHaloN{ii};
    halodata(ii,1:6) = x(1,1:6);
    halodata(ii,7) = torbsHaloN{ii}(end);
end
writematrix(halodata,"L2_Halo_Impact_Lyap")


%%
% NAMING CONVENTION: TOP (+Z) lobe looking from Earth too moon Left or
% Right
% Axial Family
axialdata = NaN(length(xorbsAxial),7);
for ii = 1:length(xorbsAxial)
    x = xorbsAxial{ii};
    axialdata(ii,1:6) = x(1,1:6);
    axialdata(ii,7) = torbsAxial{ii}(end);
end
writematrix(axialdata,"L2_Axial_Lyap_Vert_Right_5sqrt2em4")
%%
% Vertical Family Small
verticaldata = NaN(length(xorbsVert),7);
for ii = 1:length(xorbsVert)
    x = xorbsVert{ii};
    verticaldata(ii,1:6) = x(1,1:6);
    verticaldata(ii,7) = torbsVert{ii}(end);
end
writematrix(verticaldata,"data/raw/L2_Vertical_Axial_Small_5sqrt2em4")

%%
% Vertical Family to B2
verticaldata = NaN(length(xorbsVert),7);
for ii = 1:length(xorbsVert)
    x = xorbsVert{ii};
    verticaldata(ii,1:6) = x(1,1:6);
    verticaldata(ii,7) = torbsVert{ii}(end);
end
writematrix(verticaldata,"data/raw/L2_Vertical_Axial_B2_5sqrt2em4")


%% Generate L1 Lyapunovs from JPL Periodic Database
% Begin with very small L1 Lyapunov
% Note JPL database is perilune states not apolune
clc

L1_lyapunov_initial_pt = [0.833238779822767,0,0,0,0.0316535811337714,0]';	
L1_lyapunov_initial_t = 2.6945800150396697;

[t, x, STMs, ~, ~, ~, DF] = differentialCorrectionMCF(L1_lyapunov_initial_pt, L1_lyapunov_initial_t, mu, false, 2); % z0 free method
nhat = null(DF);
nhat = nhat(:,1); % Starting on a bifurcation.

% Continue Lyapunov orbits
moon_R = const.moon_R;
earth_R = const.earth_R;
bifur_tol = 1e-5;
[torbsL1Lyap,xorbsL1Lyap, MorbsL1Lyap] = pseudoArcLengthContinuation(x, t, mu, nhat, 4000, moon_R, earth_R, squeeze(STMs(end,:,:)),sqrt(2)*1e-3,bifur_tol);

%% Check that red dots at apolune
xorbs = xorbsL1Lyap;
colors = winter(length(xorbs));
figure;
hold on;
x = xorbs{1};
plot3(x(:,1), x(:,2), x(:,3), 'Color', colors(1,:), 'LineWidth', 1.5);
for i = 1:100:length(xorbs)
    x = xorbs{i};
    plot3(x(:,1), x(:,2), x(:,3), 'Color', colors(i,:), 'LineWidth', 1.2);
    scatter3(x(1,1), x(1,2), x(1,3), 4,"red");
end
colorbar;
axis equal
%%
% Save L1 Lyapunov's
truncation = 0;
lyapdata = NaN(length(xorbsL1Lyap)-truncation,7);
for ii = 1:length(xorbsL1Lyap)-truncation
    x = xorbsL1Lyap{ii};
    lyapdata(ii,1:6) = x(1,1:6);
    lyapdata(ii,7) = torbsL1Lyap{ii}(end);
end
writematrix(lyapdata,"data/raw/L1_Lyap_Small_Big_sqrt2em3")


%% Generate L2 Lyapunov Family
clc
L2_lyapunov_initial_pt = [1.16534528066803,0.000312020007967348,0,0.000239720509282074,-0.0545526821366955,0]';	
L2_lyapunov_initial_t = 3.37768222603233;

[t, x, STMs, ~, ~, ~, DF] = differentialCorrectionMCF(L2_lyapunov_initial_pt, L2_lyapunov_initial_t, mu, false, 0); % z0 free method
nhat = null(DF);
nhat = nhat(:,1);

% Continue Lyapunov orbits
moon_R = const.moon_R;
earth_R = const.earth_R;
bifur_tol = 1e-5;
[torbsL2Lyap,xorbsL2Lyap, MorbsL2Lyap] = pseudoArcLengthContinuation(x, t, mu, nhat, 10000, moon_R, earth_R, squeeze(STMs(end,:,:)), sqrt(2)*1e-3, bifur_tol);
% Check that red dots at perilune
xorbs = xorbsL2Lyap;
colors = winter(length(xorbs));

figure;
hold on;
x = xorbs{1};
h1 = plot3(x(:,1), x(:,2), x(:,3), 'Color', colors(1,:), 'LineWidth', 1.5);
for i = 1:100:length(xorbs)
    x = xorbs{i};
    plot3(x(:,1), x(:,2), x(:,3), 'Color', colors(i,:), 'LineWidth', 1.2);
    scatter3(x(1,1), x(1,2), x(1,3), 4,"red");
end
colorbar;
axis equal

% Save L2 Lyapunov's
truncation = 0;
lyapdata = NaN(length(xorbsL2Lyap)-truncation,7);
for ii = 1:length(xorbsL2Lyap)-truncation
    x = xorbsL2Lyap{ii};
    lyapdata(ii,1:6) = x(1,1:6);
    lyapdata(ii,7) = torbsL2Lyap{ii}(end);
end
writematrix(lyapdata,"data/raw/L2_Lyap_Small_Moon_sqrt2em3")

%% Generate L2 Axial Family
clc

L2_axial_initial_pt = [1.2199775808312245,0,0,0,-4.2749410564244672e-1,-3.3990164101121321e-5]';	
L2_axial_initial_t = 4.3105091455663960;

[t, x, STMs, ~, ~, ~, DF] = differentialCorrectionMCF(L2_axial_initial_pt, L2_axial_initial_t, mu, false, 0); % z0 free method
nhat = null(DF);
nhat = -1*nhat(:,1); % -1 for right
%
% Continue Axial orbits
moon_R = const.moon_R;
earth_R = const.earth_R;
bifur_tol = 1e-6;
[torbsL2Axial,xorbsL2Axial, MorbsL2Axial] = pseudoArcLengthContinuation(x, t, mu, nhat, 10000, moon_R, earth_R, squeeze(STMs(end,:,:)),5*sqrt(2)*1e-4,bifur_tol);
% Check that red dots at perilune
xorbs = xorbsL2Axial;
colors = winter(length(xorbs));
%
figure;
hold on;
x = xorbs{1};
h1 = plot3(x(:,1), x(:,2), x(:,3), 'Color', colors(1,:), 'LineWidth', 1.5);
for i = 1:10:length(xorbs)
    x = xorbs{i};
    plot3(x(:,1), x(:,2), x(:,3), 'Color', colors(i,:), 'LineWidth', 1.2);
    scatter3(x(1,1), x(1,2), x(1,3), 4,"red");
end
colorbar;
axis equal

%% Save L2 Axial's
truncation = 0;
axialdata = NaN(length(xorbsL2Axial)-truncation,7);
for ii = 1:length(xorbsL2Axial)-truncation
    x = xorbsL2Axial{ii};
    axialdata(ii,1:6) = x(1,1:6);
    axialdata(ii,7) = torbsL2Axial{ii}(end);
end
writematrix(axialdata,"data/raw/L2_Axial_Lyap_Vert_Right_5sqrt2em4")


%% Generate L1 Vertical Family
clc
L1_vert_initial_pt = [9.2125368309130207e-1,0,0,0,-2.0272760533807448, -1.7568434743034661e-1]';	
L1_vert_initial_t = 6.2997987877752051;

[t, x, STMs, ~, ~, ~, DF] = differentialCorrectionMCF(L1_vert_initial_pt, L1_vert_initial_t, mu, false, 0); % z0 free method
nhat = null(DF);
nhat = -nhat(:,1);
%
% Continue Lyapunov orbits
moon_R = const.moon_R;
earth_R = const.earth_R;
[torbsL1Vert,xorbsL1Vert, MorbsL1Vert] = pseudoArcLengthContinuation(x, t, mu, nhat, 5000, moon_R, earth_R, squeeze(STMs(end,:,:)),1e-3);

%%
x_next = xorbsL1Vert2{200}(1,:)';
t_next = torbsL1Vert2{200}(end);

[t, x, STMs, ~, ~, ~, DF] = differentialCorrectionMCF(x_next, t_next, mu, false, 0); % z0 free method
nhat = null(DF)
nhat = -nhat(:,1)
%%
[torbsL1Vert2,xorbsL1Vert2, MorbsL1Vert2] = pseudoArcLengthContinuation(x, t, mu, nhat, 2000, moon_R, earth_R, squeeze(STMs(end,:,:)),1e-3);

%%
% Check that red dots at perilune
xorbs = xorbsL1Vert2;
colors = winter(length(xorbs));
figure;
hold on;
for i = 1:10:length(xorbs)
    x = xorbs{i};
    plot3(x(:,1), x(:,2), x(:,3), 'Color', colors(i,:), 'LineWidth', 1.2);
    scatter3(x(1,1), x(1,2), x(1,3), 4,"red");
end
colorbar;
axis equal
%%


%% Save L1 Verticals
truncation = 0;
axialdata = NaN(length(xorbsL1Vert)-truncation,7);
for ii = 1:length(xorbsL1Vert)-truncation
    x = xorbsL1Vert{ii};
    axialdata(ii,1:6) = x(1,1:6);
    axialdata(ii,7) = torbsL1Vert{ii}(end);
end
writematrix(axialdata,"data/raw/L1_Vert_Small_Big_5em4")


