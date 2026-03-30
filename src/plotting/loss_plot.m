%%
loss_history = readmatrix("HaloFamily/loss_history_1.txt",delimiter=",");
%%
epochs = 1:size(loss_history, 1);
train_raw = loss_history(:, 1);
val_raw = loss_history(:, 2);


train_smooth = movmean(train_raw, [10,0]);
val_smooth = movmean(val_raw, [10,0]);

%%
figure('Color', 'w', 'Units', 'inches', 'Position', [2, 2, 20, 40]);
hold on;
set(gca, 'YScale', 'log', 'FontSize', 30, 'TickLabelInterpreter','latex');

p1_raw = plot(epochs, train_raw, 'Color', [0.8, 0.8, 1], 'LineWidth', 0.5, 'HandleVisibility', 'off');
p2_raw = plot(epochs, val_raw, 'Color', [1, 0.8, 0.6], 'LineWidth', 0.5, 'HandleVisibility', 'off');

p1 = plot(epochs, train_smooth, 'Color', [0, 0.4470, 0.7410], 'LineWidth', 3, 'DisplayName', 'Training Loss');
p2 = plot(epochs, val_smooth, 'Color', [0.8500, 0.3250, 0.0980], 'LineWidth', 3, 'DisplayName', 'Validation Loss');

xline(400,"LineWidth",2,"Color","k","LineStyle", "--", 'DisplayName', 'Learning Rate Decay')
xline(800,"LineWidth",2,"Color","k","LineStyle", "--", 'HandleVisibility', 'off')

xlabel('Epoch', 'FontSize', 30,"Interpreter","latex");
ylabel('Mean Squared Error (MSE)', 'FontSize', 30,"Interpreter","latex");
legend('Location', 'northeast', 'FontSize', 30,"Interpreter","latex");

hold off;