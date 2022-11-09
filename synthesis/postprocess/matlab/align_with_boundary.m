function [constraint, tempBox, updated, closedSeg] = align_with_boundary(box, boundary, threshold, ~)

% disp("box:")
% disp(box)

% disp("size: ")
% disp(size(box, 1))

% disp("boundary: ")
% disp(boundary)

% preprocess
box(:, [1 2]) = box(:, [2 1]);
box(:, [3 4]) = box(:, [4 3]);

tempBox = box;
updated = false(size(box, 1), 4);
closedSeg = zeros(size(box, 1), 4);
distSeg = zeros(size(box, 1), 4);


for i = 1:size(box, 1)
    [closedSeg(i,:), distSeg(i,:)] = find_close_seg(box(i,:), boundary);
end

% disp("closedSeg:")
% disp(closedSeg)
% disp("distSeg:")
% disp(distSeg)
b = zeros(size(boundary, 1), 2);
for i = 1:size(boundary, 1)
    b(i, :) = boundary(i, [1 3]);
end

polyRoom = polyshape(b);
room_area = area(polyRoom);


direction = box(:, 9:11);

% move furniture to near boundary
for i = 1:size(distSeg, 1)
    distance = distSeg(i, :);;
    
    abs_distance = abs(distance);
    skip_types = [6, 7, 8, 9];
    polyFurniture = polyshape(box(i, [1 1 3 3]), box(i, [2 4 4 2]));
    furniture_area = area(polyFurniture);
    if (isempty(find(abs_distance <= threshold * 4, 1)) && furniture_area < room_area / 10) || ismember(box(i, 12), skip_types)
        continue
    end

    types = [2, 3, 4, 10, 14, 15, 16, 17, 18];

    vertical = 0;
    horizontal = 0;

    if(direction(i, 1) == -1 && (abs_distance(3) <= threshold * 3 || ismember(box(i, 12), types)))
        horizontal = -distance(3) - 0.02;
        updated(i, 3) = true;
        if ismember(box(i, 12), types)
            updated(i, 1) = true;
        end
    elseif(direction(i, 1) == 1 && (abs_distance(1) <= threshold * 3 || ismember(box(i, 12), types)))
        horizontal = -distance(1) + 0.02;
        updated(i, 1) = true;
        if ismember(box(i, 12), types)
            updated(i, 3) = true;
        end
    else
        if((abs_distance(1) < abs_distance(3) && abs_distance(1) <= threshold) || (box(i, 3) < closedSeg(i, 1)))
            horizontal = -distance(1) + 0.02;
            if box(i, 9) ~= 0
                box(i, 9) = 1;
            end
            updated(i, 1) = true;
        elseif((abs_distance(3) <= threshold) || (box(i, 1) > closedSeg(i, 3)))
            horizontal = -distance(3) - 0.02;
            if box(i, 9) ~= 0
                box(i, 9) = -1;
            end
            updated(i, 3) = true;
        end
    end


    if(direction(i, 2) == -1 && (abs_distance(2) <= threshold * 3 || ismember(box(i, 12), types)))
        vertical = -distance(2) - 0.02;
        updated(i, 2) = true;
        if ismember(box(i, 12), types)
            updated(i, 4) = true;
        end
    elseif(direction(i, 2) == 1 && (abs_distance(4) <= threshold * 3 || ismember(box(i, 12), types)))
        vertical = -distance(4) + 0.02;
        updated(i, 4) = true;
        if ismember(box(i, 12), types)
            updated(i, 2) = true;
        end
    else
        if((abs_distance(2) <= abs_distance(4) && abs_distance(2) <= threshold) || (box(i, 2) > closedSeg(i, 2)))
            vertical = -distance(2) - 0.02;
            if box(i, 10) ~= 0
                box(i, 10) = -1;
            end
            updated(i, 2) = true;  
        elseif((abs_distance(4) <= threshold) || (box(i, 4) < closedSeg(i, 4)))
            vertical = -distance(4) + 0.02;
            if box(i, 10) ~= 0
                box(i, 10) = 1;
            end
            updated(i, 4) = true;
        end
    end



%     disp("horizontal:")
%     disp(horizontal)
%     disp("vertical: ")
%     disp(vertical)
    tempBox(i, [1 3]) = box(i, [1 3]) + horizontal;
    tempBox(i, [2 4]) = box(i, [2 4]) + vertical;
end

% TODO: preprocess
tempBox(:, [1 2]) = tempBox(:, [2 1]);
tempBox(:, [3 4]) = tempBox(:, [4 3]);

% disp("tempBox:")
% disp(tempBox)
% box(distSeg <= threshold) = closedSeg(distSeg <= threshold);
% updated(distSeg <= threshold) = true;
idx = find(distSeg <= threshold);
constraint = [idx closedSeg(idx)];


%% check if any room box blocks the door
% entranceBox = get_entrance_space(boundary(1:2, 1:2), boundary(1,3), threshold);
% entrancePoly = polyshape(entranceBox([1 1 3 3]), entranceBox([2 4 4 2]));

% for i = 1:length(box) - 1
% 
%     if rType(i) ~= 10 && rType(i) ~= 0
%         roomPoly = polyshape(box(i, [1 1 3 3]), box(i, [2 4 4 2]));
%         if overlaps(entrancePoly, roomPoly)
%             box(i,:) = shrink_box(roomPoly, entrancePoly, boundary(1,3));
%             updated(i, box(i,:)==tempBox(i,:)) = false;
%             updated(i, box(i,:)~=tempBox(i,:)) = true;
%         end
%     end        
% end
