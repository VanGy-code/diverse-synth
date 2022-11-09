function [newBox, type, edge] = align_fp(boundary, box, type, edge, threshold, color_map, drawResult)

% get entrance box if entrance is exist, but now we can not get the entrance

% move to centroid
[newBox] = align_centorid(boundary, box);

if drawResult
    figure(10)
    subplot(2,2,1)
    plot_fp(newBox, boundary, type, color_map);
    title('original');

    figure(11)
    subplot(2,2,1)
    plot_scene(newBox, boundary, type, color_map);
    title('original');
end

%% use greedy method: align 
% with boundary first and then neighbor

% 1. align with boundary after the neighbors have been aligned
[~, newBox, updated, closedSeg] = align_with_boundary(newBox, boundary, threshold, type);

if drawResult
    figure(10)
    subplot(2,2,2)
    plot_fp(newBox, boundary, type, color_map);
    title('Align with boundary');

    figure(11)
    subplot(2,2,2)
    plot_scene(newBox, boundary, type, color_map);
    title('original');
end

%% 2. adjacent each pair of furniture,
[~, newBox, ~] = align_neighbor(newBox, edge, updated, threshold, closedSeg);
if drawResult
    figure(10)
    subplot(2,2,3)
    plot_fp(newBox, boundary, type, color_map);
    title('Align with neighbors');

    figure(11)
    subplot(2,2,3)
    plot_scene(newBox, boundary, type, color_map);
    title('Align with neighbors');
end

%% 3. regularize fp, include box size reorganization and eliminate overlap 
[newBox, valid_index, temp_box] = regularize_fp(newBox, boundary, type);


newBox = newBox(valid_index,:);
type = type(valid_index);

if drawResult
    figure(10)
    subplot(2,2,4)
    % plot_fp(newBox(order,:), boundary, type(order), entranceBox);
    plot_fp(newBox, boundary, type, color_map);
    title('Regularize fp');

    figure(11)
    subplot(2,2,4)
    plot_scene(newBox, boundary, type, color_map);
    title('Regularize fp');
end