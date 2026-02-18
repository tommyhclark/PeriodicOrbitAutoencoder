function [l1,l2,l3,l4,l5] = lagrangePts(mu)
    options = optimset('FunValCheck', 'on');
    func = @(x) x - (1 - mu) * (x + mu) / (abs(x + mu))^3 - mu * (x - 1 + mu) / (abs(x - 1 + mu))^3;
    l1 = [fzero(func, 0.5,options);0;0];
    l2 = [fzero(func, 1.3,options);0;0];
    l3 = [fzero(func, -1-0.5*mu,options);0;0];
    l4 = [0.5-mu; sqrt(3)/2;0];
    l5 = [0.5-mu; -sqrt(3)/2;0];
end