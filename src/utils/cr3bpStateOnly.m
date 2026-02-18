function c = cr3bpStateOnly(x,mu)
    grad = gradU(x, mu);
    c = [x(4); x(5); x(6); 2*x(5)+grad(1); -2*x(4)+grad(2); grad(3)];
end