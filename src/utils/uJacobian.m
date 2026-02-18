function u=uJacobian(state,mu)
    x = state(1,:);
    y = state(2,:);
    z = state(3,:);
    r1 = sqrt((x+mu).^2+y.^2+z.^2);
    r2 = sqrt((x+mu-1).^2+y.^2+z.^2);
    uxx = 1-(1-mu)./r1.^3-mu./r2.^3+3*(1-mu).*(x+mu).^2./r1.^5+3*mu*(x-1+mu).^2./r2.^5;
    uxy = 3*(1-mu).*(x+mu).*y./r1.^5+3*mu*(x-1+mu).*y./r2.^5;
    uxz = 3*(1-mu).*(x+mu).*z./r1.^5+3*mu*(x-1+mu).*z./r2.^5;
    uyy = 1-(1-mu)./r1.^3-mu./r2.^3+3*(1-mu).*y.^2./r1.^5+3*mu*y.^2./r2.^5;
    uyz = 3*(1-mu)*y*z./r1.^5+3*mu.*y.*z./r2.^5;
    uzz = -(1-mu)./r1.^3-mu./r2.^3+3*(1-mu).*z.^2./r1.^5+3*mu*z.^2./r2.^5;

    u = [uxx,uxy,uxz;
        uxy,uyy, uyz;
        uxz, uyz,uzz];
end