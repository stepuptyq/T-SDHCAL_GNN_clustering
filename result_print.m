clear all;
clc;

curve_num = 3;

% 定义文件路径（注意MATLAB中使用正斜杠或双反斜杠）
file_path1 = 'D:\GNN\results\20250326_2\k_results_kmin_1_kmax_6_2k_time.txt';
if curve_num >= 2
    file_path2 = 'D:\GNN\results\20250326_2\k_results_kmin_1_kmax_6_2k_pos.txt';
end
if curve_num >= 3
    file_path3 = 'D:\GNN\results\20250326_2\k_results_kmin_1_kmax_6_2k_both.txt';
end

% 读取数据（显式指定逗号为分隔符）
data1 = dlmread(file_path1, ',');
if curve_num >= 2
    data2 = dlmread(file_path2, ',');
end
if curve_num >= 3
    data3 = dlmread(file_path3, ',');
end

% 定义颜色方案 (MATLAB默认配色 + 自定义紫色)
color1 = [0, 0.4470, 0.7410];       % 蓝色
if curve_num >= 2
    color2 = [0.8500, 0.3250, 0.0980];  % 橙色
end
if curve_num >= 3
    color3 = [0.4660, 0.6740, 0.1880];  % 绿色
end
% color3 = [0.4940, 0.1840, 0.5560];  % 紫色

% 共用参数
k_min = 1;
k_max = 6;
k_values = k_min:k_max;

% 创建画布
figure('Color','white','Position',[100, 100, 1200, 600]);
hold on;

% ===== 绘制数据集1 =====
means1 = mean(data1, 1);
stds1 = std(data1, 0, 1);
fill([k_values, fliplr(k_values)],...
    [means1 - stds1, fliplr(means1 + stds1)],...
    color1, 'FaceAlpha',0.15, 'EdgeColor','none');
line1 = plot(k_values, means1,...
    'Color', color1, 'LineWidth',2,...
    'Marker','o', 'MarkerSize',8, 'MarkerFaceColor','w');
errorbar(k_values, means1, stds1,...
    'Color', color1, 'LineStyle','none', 'CapSize',6);

if curve_num >= 2
    % ===== 绘制数据集2 =====
    means2 = mean(data2, 1);
    stds2 = std(data2, 0, 1);
    fill([k_values, fliplr(k_values)],...
        [means2 - stds2, fliplr(means2 + stds2)],...
        color2, 'FaceAlpha',0.15, 'EdgeColor','none');
    line2 = plot(k_values, means2,...
        'Color', color2, 'LineWidth',2,...
        'Marker','^', 'MarkerSize',8, 'MarkerFaceColor','w');
    errorbar(k_values, means2, stds2,...
        'Color', color2, 'LineStyle','none', 'CapSize',6);
end
if curve_num >= 3
    % ===== 绘制数据集3 ===== 
    means3 = mean(data3, 1);
    stds3 = std(data3, 0, 1);
    fill([k_values, fliplr(k_values)],...
        [means3 - stds3, fliplr(means3 + stds3)],...
        color3, 'FaceAlpha',0.15, 'EdgeColor','none');  % 更浅的透明度
    line3 = plot(k_values, means3,...
        'Color', color3, 'LineWidth',2,...
        'Marker','d', 'MarkerSize',8, 'MarkerFaceColor','w'); % 菱形标记
    errorbar(k_values, means3, stds3,...
        'Color', color3, 'LineStyle','none', 'CapSize',6);
end

% ===== 图形美化 =====
% 坐标轴设置
xlim([k_min-1, k_max+0.5]);
% ylim([0, 1]);  % 根据实际数据范围调整
xlabel('k Value', 'FontSize',14, 'FontWeight','bold');
ylabel('Accuracy', 'FontSize',14, 'FontWeight','bold');
if curve_num == 1
    title('Performance Trend with Error Range', 'FontSize', 16);
    % 图例与注释
    legend(line1,...
        {'Time'},...
        'Location', 'northeast', 'Box','off', 'FontSize',12);
elseif curve_num == 2
    title('Performance Comparison with Error Ranges', 'FontSize',16);
    % 图例与注释
    legend([line1, line2],...
        {'Time', 'Position'},...
        'Location', 'northeast', 'Box','off', 'FontSize',12);
elseif curve_num == 3
    title('Three-Way Performance Comparison', 'FontSize',16);
    % 图例与注释
    legend([line1, line2, line3],...
        {'Time', 'Position', 'Both'},...
        'Location', 'northeast', 'Box','off', 'FontSize',12);
end

[~,max_idx1] = max(means1);
text(k_values(max_idx1)-1, means1(max_idx1),...
    sprintf('Max: %.2f %%', means1(max_idx1) * 100),...
    'VerticalAlignment','bottom', 'Color',color1, 'FontSize',14);
if curve_num >= 2
    [~,max_idx2] = max(means2);
    text(k_values(max_idx2)-1, means2(max_idx2),...
        sprintf('Max: %.2f %%', means2(max_idx2) * 100),...
        'VerticalAlignment','bottom', 'Color',color2, 'FontSize',14);
end
if curve_num >= 3
    [~,max_idx3] = max(means3);
    text(k_values(max_idx3)-1, means3(max_idx3),...
        sprintf('Max: %.2f %%', means3(max_idx3) * 100),...
        'VerticalAlignment','bottom', 'Color',color3, 'FontSize',14);
end


set(gca, 'FontSize',12, 'XTick',k_values,...
    'GridAlpha',0.3, 'GridLineStyle','--');
grid on;

hold off;

