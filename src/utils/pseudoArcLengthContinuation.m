function [torbs, xorbs, Morbs] = pseudoArcLengthContinuation(x, t, mu, nhat, max_orbs, R_Impact_moon, R_Impact_earth, Phi,deltaS, bifurcation_tolerance)
    % Set up cell to hold each orbit's trajectory
    orbs = 1;
    xorbs={x};
    torbs = {t};
    Morbs = {Phi};

    % Iteration Parameters
    max_iter = 15;

    vprev = [x(1,1:6)';t(end)];
    % Loop over family members
    while orbs<max_orbs
        disp(orbs)
        v = vprev+deltaS*nhat;
        options = odeset('RelTol', 3e-14, 'AbsTol',1e-16);
        iter = 1;
        while iter<=max_iter
            [t,x] = ode113(@(t,state) cr3bp(state,  mu), [0 v(7)], [v(1:6); reshape(eye(6), [], 1)], options);
            
            x_f = x(end,1:6).';
            STM_f = reshape(x(end,7:end), 6, 6);
            
            DH = zeros(7,7);
            DH(1:6, 1:6) = STM_f-eye(6);
            DH(1:6, 7) = cr3bpStateOnly(x_f,mu);
            DH(5,:) = DH(6,:);
            DH(6,:) = [0,1,0,0,0,0,0];
            DH(7,:) = nhat';


            H = [x_f(1:4)-v(1:4); x_f(6)-v(6); v(2); (v-vprev)'*nhat-deltaS];

            if norm(H)<1e-10
                break
            else
                v = v - pinv(DH)*H;
            end

            iter=iter+1;
        end

        % if iter>6
        %     deltaS = deltaS/10;
        %     disp("Reducing Step Size")
        % elseif iter<3
        %     deltaS = deltaS*10;
        %     disp("Increasing Step Size")
        % end
        
        min_alt_moon = min(sqrt((x(:,1)-1+mu).^2+x(:,2).^2+x(:,3).^2));

        min_alt_earth = min(sqrt((x(:,1)-mu).^2+x(:,2).^2+x(:,3).^2));

        if iter >= max_iter
            disp("Failed to Converge")
            break
        elseif min_alt_moon<R_Impact_moon
            disp("Lunar Impact")
            break
        elseif min_alt_earth<R_Impact_earth
            disp("Earth Impact")
            break
        else
            xorbs{orbs+1} = x(:,1:6);
            torbs{orbs+1} = t;
            Morbs{orbs+1} = STM_f;
            orbs = orbs+1;
            nhat_new = null(DH(1:6,:));

            ev = eig(STM_f);
            [~, idx] = sort(abs(real(ev) - 1), 'ascend');
            ev(idx(1:2))=[];
            stab_idx1 = ev(1) + 1/ev(1);
            if abs(ev(2)-1/ev(1))>1e-6
                stab_idx2 = ev(2) + 1/ev(2);
            else
                stab_idx2 = ev(4) + 1/ev(4);
            end
            stab_idx_bifur_tol = bifurcation_tolerance;
            stab_min = min(abs([abs(stab_idx1-2),abs(stab_idx2-2)]));
            if stab_min<stab_idx_bifur_tol
                if orbs>100 %&& size(nhat_new,2)>1%100
                    disp("Bifurcation")
                    disp(nhat_new)
                    % [~,ii] = max(abs(nhat'*nhat_new));
                    % nhat_new = nhat_new(:,ii);
                    break
                end
            end
            % Code to continue through bifurcation
            % if size(nhat_new,2)>1 
            %     disp("Bifurcation")
            %     [~,ii] = max(abs(nhat'*nhat_new));
            %     nhat_new = nhat_new(:,ii);
            % end
            if dot(nhat,nhat_new)<0
                nhat_new = -1*nhat_new;
            end
            nhat = nhat_new;
            vprev=v;
        end
    end
end