% Lock apolune: y=0,xdot=0 and lock period (commenting out variable
% period)
function [x, T, collision] = singleShooting(x0, T0, mu, r_moon)
    tol = 1e-10;
    v = x0;
    % v(7) = T0;
    iter = 1;
    errors = zeros(1);
    options = odeset('RelTol', 3e-14, 'AbsTol',1e-16);
    collision = false; % Initialize collision flag

    while iter <= 100
        % [t, y] = ode113(@(t, state) cr3bp(state, mu), [0 v(7)], [v(1:6); reshape(eye(6), [], 1)], options);
        [t, y] = ode113(@(t, state) cr3bp(state, mu), [0 T0], [v(1:6); reshape(eye(6), [], 1)], options);
        if iter == 1
            yinit = y;
        end
        x_f = y(end, 1:6).';
        STM_f = reshape(y(end, 7:end), 6, 6);
        STMs = reshape(y(:, 7:end), length(t), 6, 6);
        
        % Check for Moon collision
        moon_pos = [1-mu; 0; 0]; % Moon's position in CR3BP
        distances = sqrt(sum((y(:, 1:3) - moon_pos').^2, 2)); % Distance to Moon's center
        if any(distances < r_moon)
            collision = true; % Flag if any point is within Moon's radius
        end
        
        DF = zeros(6,6);
        % DF = zeros(6, 7);
        DF(1:6, 1:6) = STM_f - eye(6);
        % DF(1:6, 7) = cr3bpStateOnly(x_f, mu);
        F = x_f - v(1:6);

        %Lock y,xdot
        F(7) = v(2);
        F(8) = v(4);
        DF(7,2) = 1;
        DF(8,4) = 1;

        errors(iter) = norm(F);
        if norm(F) < tol
            break
        else
            v = v - pinv(DF)*F;
        end
        iter = iter + 1;
    end

    if iter == 100 || norm(v-x0)>1e-1
        disp("Failed to Converge")
    end
    x = y(:, 1:6);
    % T = t(end);
    T = T0;
end