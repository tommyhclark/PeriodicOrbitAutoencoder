function c = jacobiConstant(x, mu)
    r1 = sqrt((x(:,1)+mu).^2+x(:,2).^2+x(:,3).^2);
    r2 = sqrt((x(:,1)-1+mu).^2+x(:,2).^2+x(:,3).^2);
    c = x(:,1).^2+x(:,2).^2+2*(1-mu)./r1+2*mu./r2-x(:,4).^2-x(:,5).^2-x(:,6).^2;
end