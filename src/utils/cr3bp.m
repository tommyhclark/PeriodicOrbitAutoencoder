function c = cr3bp(x,mu)
    xPOS = x(1:6);
    grad = gradU(xPOS, mu);
    xPOS_dot = [xPOS(4); xPOS(5); xPOS(6); 2*xPOS(5)+grad(1); -2*xPOS(4)+grad(2); grad(3)];

    xSTM = reshape(x(7:end), 6, 6);
    dSTM = stmJacobian(x, mu);
    xSTM_dot = reshape(dSTM*xSTM, [], 1);

    c = [xPOS_dot; xSTM_dot];
end