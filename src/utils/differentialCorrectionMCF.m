function [t, x, STMs, errors, v, yinit, df] = differentialCorrectionMCF(init, T, mu, plots, opt)
    v = init;
    v(7) = T;
    iter = 1;
    errors = zeros(1);
    options = odeset('RelTol', 3e-14, 'AbsTol',1e-16);

    while iter<=20
      [t,y] = ode113(@(t,state) cr3bp(state,  mu), [0 v(7)], [v(1:6); reshape(eye(6), [], 1)], options);
      if iter==1
        yinit = y;
      end
      x_f = y(end,1:6).';
      STM_f = reshape(y(end,7:end), 6, 6);
      STMs = reshape(y(:,7:end), length(t), 6, 6);
      DF = zeros(6,7);
      
      %General Constraint Formulation
      DF(1:6, 1:6) = STM_f-eye(6);
      DF(1:6, 7) = cr3bpStateOnly(x_f,mu);
        
      if opt == 0
          % y0=0 -> periapsis pos.
          DF(5,:) = DF(6,:);
          DF(6,:) = [0,1,0,0,0,0,0];
          f = [x_f(1:4)-v(1:4); x_f(6)-v(6); v(2)];
      elseif opt == 1
          % y0=0, z0=0, xdot0=0 -> Planar periapsis
          DF(2,:) = [0,1,0,0,0,0,0];
          DF(3,:) = [0,0,1,0,0,0,0];
          DF(4,4) = 1;
          f = [x_f(1)-v(1); v(2); v(3);v(4);x_f(5:6)-v(5:6)];
      elseif opt == 2
          %%% y0 = 0, xdot0 = 0 -> periapsis
          DF(2,:) = [0,1,0,0,0,0,0];
          DF(4,:) = [0,0,0,1,0,0,0];
          f = [x_f(1)-v(1); v(2); x_f(3)-v(3);v(4);x_f(5:6)-v(5:6)];
      end

      errors(iter) = norm(f);
      if norm(f)<1e-10
        break
      else
        v = v - pinv(DF)*f;
      end
      iter=iter+1;
    end
    
    if iter >= 20
        disp("Failed to Converge")
    end

    if x_f(5)-v(5)>1e-10 || sign(x_f(5))~= sign(v(5))
        disp("Failed to Converge ydot")
    end

    df=DF;

    x = y(:, 1:6);

    if plots
        tiledlayout(1, 2);
    
        nexttile;
        semilogy(errors);
        title('Error Plot');
        xlabel('Index');
        ylabel('Error');
    
        nexttile;
        plot3(x(:,1), x(:,2), x(:,3));
        view(0, 90);
        title('Orbit');
        xlabel('X');
        ylabel('Y');
        zlabel('Z');
    end
end