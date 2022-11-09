function [] = plot_fp(boxes, boundary, type, color_map)

%% 1. plot boundary
for i = 1:length(boundary)
    if i < length(boundary)
        p_0 = [boundary(i, 3); boundary(i + 1, 3)];
        p_1 = [boundary(i, 1); boundary(i + 1, 1)];
    else
        p_0 = [boundary(i, 3); boundary(1, 3)];
        p_1 = [boundary(i, 1); boundary(1, 1)];
    end
    plot(p_0, p_1, '-k', linewidth=2.5);
    grid on;
    hold on;
end


%% 2. plot furniture box
for i = 1:size(boxes, 1)
    maxr = boxes(i, 1);
    maxc = boxes(i, 2);
    minr = boxes(i, 3);
    minc = boxes(i, 4);
    x = [minc; maxc; maxc; minc; minc];
    y = [minr; minr; maxr; maxr; minr];
    plot(x, y, color=color_map(type(i) + 1, :), linewidth=1.5);
    hold on;
end

%% 3. plot furniture direction


%% 4. plot furniture function area

end