function [v, iter, diff, corrected_flag, initial_constraint_violation, initial_step_size, corrected_period] = multipleShootingVariablePeriod(x_init, T0, mu)
    
    max_iter = 20;
    tol = 1e-10;

    v(:,1:6) = x_init(:,1:6);
    iter = 1;
    errors = zeros(1,max_iter);
    options = odeset('RelTol', 3e-14, 'AbsTol',1e-16);
   
    nseg = size(v,1);
    dt = T0/nseg;
    v_init = v;
   
    while iter <= max_iter
        F = zeros(6*nseg+2,1);
        DF = zeros(6*nseg+2,6*nseg+1);
        % F = zeros(6*nseg+3,1);
        % DF = zeros(6*nseg+3,6*nseg+1);

        for ii = 1:nseg-1
            [~,y1] = ode113(@(t,state) cr3bp(state, mu), [0 dt], [v(ii,1:6)'; reshape(eye(6), [], 1)], options);
            F((ii-1)*6+1:ii*6) = squeeze(y1(end,1:6)-v(ii+1,1:6));
            stm = reshape(y1(end,7:end),6,6);
            DF((ii-1)*6+1:ii*6,(ii-1)*6+1:ii*6) = stm;
            DF((ii-1)*6+1:ii*6,(ii)*6+1:(ii+1)*6) = -1*eye(6);
            DF((ii-1)*6+1:ii*6,6*nseg+1) = cr3bpStateOnly(y1(end,1:6),mu);
        end
        [~,y1] = ode113(@(t,state) cr3bp(state, mu), [0 dt], [v(nseg,1:6)'; reshape(eye(6), [], 1)], options);
        F((nseg-1)*6+1:nseg*6) = squeeze(y1(end,1:6)-v(1,1:6));
        stm = reshape(y1(end,7:end),6,6);
        DF((nseg-1)*6+1:nseg*6,(nseg-1)*6+1:nseg*6) = stm;
        DF((nseg-1)*6+1:nseg*6,1:6) = -1*eye(6);
        DF((nseg-1)*6+1:nseg*6,6*nseg+1) = cr3bpStateOnly(y1(end,1:6),mu);

        if iter==1
            initial_constraint_violation = norm(F);
        end
        errors(iter) = norm(F);
        
        % Lock X axis
        F(6*nseg+1) = v(1,2);
        F(6*nseg+2) = v(1,4);
        % F(6*nseg+3) = v(1,6);
        DF(6*nseg+1,2) = 1;
        DF(6*nseg+2,4) = 1;
        % DF(6*nseg+3,6) = 1;

        diff = v - v_init;

        corrected_period = dt*nseg;

        if norm(F) < tol
            fprintf('  Iteration %i: |f| = %e\n  Converged. \n',iter,norm(F))
            corrected_flag = 1;
            break
        else
            corrected_flag = 0;
            full_correction = pinv(DF)*F;
            dt = dt - full_correction(end);
            dv = reshape(full_correction(1:end-1),6,nseg)';
            
            v = v - dv;
            if iter == 1
                initial_step_size = norm(dv);
            end
        end
        fprintf('  Iteration %i: |f| = %e | |D| = %e \n',iter,norm(F), norm(dv))
        if iter==max_iter
            fprintf("  DID NOT CONVERGE \n")
        end
        iter = iter + 1;
    end

end