function s = stmJacobian(x, mu)
    s = zeros(6,6);
    s(1:3,4:6) = eye(3);
    s(4:6,1:3) = uJacobian(x, mu);
    s(4,5) = 2;
    s(5,4) = -2;
end