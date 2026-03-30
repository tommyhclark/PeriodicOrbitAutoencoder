%% Figure - Compression Plot
paramcount = [1146990,573750,287130,208794,182682,83640,73134,36618,32354];
DC = [7.3952e-5,1.0928e-4,1.3479e-4,1.6797e-4,2.7905e-4,2.6234e-4,4.1134e-4,5.1869e-4,5.5447e-4];
DI = [2.5783e-3,3.7740e-3,5.6424e-3,6.1137e-3,1.1601e-2,1.0618e-2,1.5824e-2,2.1920e-2,2.0403e-2];
MSE = [8.2023e-9,1.5868e-8,3.7338e-8,7.7567e-8,7.2927e-8,2.3552e-7,2.1400e-7,1.4907e-6,4.4819e-6];

totdata = 307*4280;
compressionrat = paramcount/totdata;
%%
figure;
hold on
grid on
yscale("log")

plot(compressionrat,DC,"-ok")
plot(compressionrat,DI,"-or")
plot(compressionrat,MSE,"-ob")
legend(["$\Delta C$","$\Delta I$","MSE"],"Interpreter","latex","FontSize",20)
xlabel("Compression Ratio","Interpreter","latex","FontSize",20)
set(gca,"TickLabelInterpreter","latex","FontSize",20)