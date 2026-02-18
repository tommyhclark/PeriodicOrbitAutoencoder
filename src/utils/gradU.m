function grad = gradU(x, mu)
    grad = zeros(3,1);
    r1 = sqrt((x(1)+mu)^2+x(2)^2+x(3)^2);
    r2 = sqrt((x(1)-1+mu)^2+x(2)^2+x(3)^2);
    grad(1) = x(1) - (1-mu)*(x(1)+mu)/r1^3 - mu*(x(1)-1+mu)/r2^3;
    grad(2) = x(2) - (1-mu)*x(2)/r1^3 - mu*x(2)/r2^3;
    grad(3)= -(1-mu)*x(3)/r1^3 - mu*x(3)/r2^3;
end