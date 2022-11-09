function [] = plot_scene(boxes, boundary, type, color_map)
    
%% 1. plot boundary
Y = boundary(:, 1);
X = boundary(:, 3);
Z = zeros(length(boundary)) - 0.01;
floor = fill3(X, Y, Z, [0.5, 0.5, 0.5]);
set(floor,'edgealpha',0,'facealpha',0.5);

hold on;

%% 2. plot furniture box
for i = 1:size(boxes, 1)
    maxr = boxes(i, 1);
    maxc = boxes(i, 2);
    minr = boxes(i, 3);
    minc = boxes(i, 4);
    z = boxes(i, 5);
    h = z + boxes(i, 8);
    x1 = [minc; maxc; maxc; minc; minc; minc; maxc; maxc; minc; minc;];
    y1 = [minr; minr; maxr; maxr; minr; minr; minr; maxr; maxr; minr;];
    z1 = [0; 0; 0; 0; 0; h; h; h; h; h];
    plot3(x1, y1, z1, color=color_map(type(i) + 1, :));
    hold on;

    f1 = fill3([minc; maxc; maxc; minc;], ...
        [minr; minr; maxr; maxr;], ...
        [0 ;0; 0; 0], color_map(type(i) + 1, :));

    set(f1,'edgealpha',0,'facealpha',0.5) 
    hold on;

    f2 = fill3([minc; maxc; maxc; minc;], ...
        [minr; minr; maxr; maxr;], ...
        [h ;h; h; h], color_map(type(i) + 1, :));

    set(f2,'edgealpha',0,'facealpha',0.5) 
    hold on;


    x2 = [maxc; maxc; maxc; maxc];
    y2 = [minr; minr; maxr; maxr];
    z2 = [0; h; h; 0];

    plot3(x2, y2, z2, color=color_map(type(i) + 1, :));
    hold on;

    f3 = fill3(x2, y2, z2, color_map(type(i) + 1, :));

    set(f3,'edgealpha',0,'facealpha',0.5) 
    hold on;

    x3 = [minc; minc; minc; minc];
    y3 = [maxr; maxr; minr; minr];
    z3 = [0; h; h; 0];

    plot3(x3, y3, z3, color=color_map(type(i) + 1, :));
    hold on;

    f4 = fill3(x3, y3, z3, color_map(type(i) + 1, :));

    set(f4,'edgealpha',0,'facealpha',0.5) 
    hold on;


    f5 = fill3([minc; minc; maxc; maxc;], ...
        [maxr; maxr; maxr; maxr;], ...
        [0; h; h; 0], color_map(type(i) + 1, :));

    set(f5,'edgealpha',0,'facealpha',0.5) 
    hold on;
    
    f6 = fill3([minc; minc; maxc; maxc;], ...
        [minr; minr; minr; minr;], ...
        [0; h; h; 0], color_map(type(i) + 1, :));

    set(f6,'edgealpha',0,'facealpha',0.5) 
    hold on;

end

end