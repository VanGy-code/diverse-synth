function [constraint, box, updated] = align_neighbor(box, edge, updated, threshold, closed_boundary)

if isempty(updated)
    updated = false(size(box));
end

% disp(updated)

% preprocess
box(:, [1 2]) = box(:, [2 1]);
box(:, [3 4]) = box(:, [4 3]);

tempBox = box;

constraint = zeros(size(edge, 1)*3, 2);

iBegin = 1;

% check the start point of each edge
checked = false(size(edge, 1), 1);

% updatedCount = sum of the four direction of each furniture in edges
updatedCount = get_updated_count(updated, edge);

for i = 1:size(edge, 1)
    % find unchecked edge
    I = find(~checked);

    % find unchecked edge which has the most updated point
    % means we need to align which is easy to align first
    [~, t] = maxk(updatedCount(I), 1);

    % checked the edge
    checked(I(t)) = true;

    % get the furniture of current edge
    idx = edge(I(t),1:2) + 1;

    % MAIN: align and adjacent the furniture
    [newBox, newConstraint] = align_adjacent_furniture(box(idx, :), tempBox(idx, :), updated(idx,:), edge(I(t),3), threshold, closed_boundary(idx, :));

    for j = 1:length(idx)
        
        newConstraint(:, j) = (newConstraint(:,j)-1)*size(box,1) + double(idx(j));
        
    end

    % update box
    box(idx, :) = newBox;
    
    % get the chaged constraint rows length
    cNum = size(newConstraint, 1);
    
    % update contraint
    constraint(iBegin:iBegin+cNum-1, :) = newConstraint;

    % update iBegin
    iBegin = iBegin+cNum;
    
    % update count
    updatedCount = get_updated_count(updated, edge);
end

% squeeze constraint
constraint = constraint(1:iBegin-1, :);

box(:, [1 2]) = box(:, [2 1]);
box(:, [3 4]) = box(:, [4 3]);

function updatedCount = get_updated_count(updated, edge)
    updatedCount = zeros(size(edge, 1), 1);
    for k = 1:size(edge, 1)
        % find the index of updated furniture in each edge
        index = edge(k,1:2)+1;
        updatedCount(k) = sum(sum(updated(index,:)));
    end
end
end