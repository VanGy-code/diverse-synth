function [box, valid_index, temp_box] = regularize_fp(box, boundary, type)

% preprocess
box(:, [1 2]) = box(:, [2 1]);
box(:, [3 4]) = box(:, [4 3]);


%% 0. check if there are boxes out of boundary
box_outrange_index = ones(size(box,1), 1);

for i = 1:size(box,1)
    [closedSeg, distSeg] = find_close_seg(box(i,:), boundary);
    if distSeg(1) > 0 && distSeg(2) < 0 && distSeg(3) < 0 && distSeg(4) > 0
        box_outrange_index(i) = 0;
    end
end

% disp(box_outrange_index)

%% 1. check if there is any overlapped region

% get iou
iou = zeros(size(box,1), size(box,1));
for i = 1:size(box,1)
    polyFurniture1 = polyshape(box(i, [1 1 3 3]), box(i, [2 4 4 2]));
    area1 = area(polyFurniture1);

    for j = i+1:size(box,1)
         polyFurniture2 = polyshape(box(j, [1 1 3 3]), box(j, [2 4 4 2]));
         area2 = area(polyFurniture2);
         inter = intersect(polyFurniture1, polyFurniture2);
         if inter.NumRegions >= 1
             inter_area = area(inter);
             iou(i, j) = inter_area / area1;
             iou(j, i) = inter_area / area2;
         end
    end
end

[overlap_x, overlap_y] = find(iou > 0);



%% 2. filter box 
% remove the box has maximum value of iou
checked = false(size(box));
temp_box = zeros(size(box, 1), size(box, 2) + 1);
valid_index = true(1, size(box, 1));

for i = 1:size(overlap_x)
    x = overlap_x(i);
    y = overlap_y(i);
    if any([~valid_index(x), ~valid_index(y)])
        continue;
    end

    if iou(x, y) > iou(y, x)
        temp_box(x, 1:12) = box(x, :);
        temp_box(x, 13) = 1;
        valid_index(x) = false;
    else
        temp_box(y, 1:12) = box(y, :);
        temp_box(y, 13) = 1;
        valid_index(y) = false;
    end

    checked(x, y) = true;
    checked(y, x) = true;

    
end

% disp(valid_index)

for i = 1:size(box_outrange_index)
    if(box_outrange_index(i) == 0)
        continue
    end
    if valid_index(i) == 0
        continue
    end
    temp_box(i, 1:12) = box(i, :);
    temp_box(i, 13) = 1;
    valid_index(i) = false;
end

% disp(valid_index)


%% 3. keep operate space
box_block_path = zeros(size(box,1), 1);
direction = box(:, 9:11);
types = [0, 9, 10, 11, 12, 10, 20];
for i = 1:size(box,1)
    if ismember(box(i, 12), types) || valid_index(i) == 0
        continue;
    end
    dir_i = direction(i, :);
    bbox_i = box(i, 1: 4);
    for j = 1:size(box, 1)
        bbox_j = box(j, 1: 4);
        if i == j || valid_index(j) == 0
            continue;
        end
        if dir_i(1) == -1 && abs(bbox_i(3) - bbox_j(1)) < 0.15 && (((bbox_i(4) < bbox_j(2)) &&  (bbox_j(2) < bbox_i(2))) || ((bbox_i(4) < bbox_j(4)) && (bbox_j(4) < bbox_i(2))))

            if box_block_path(i) ~= 1 && j > i
                box_block_path(j) = 1;
            elseif box_block_path(i) ~= 1 && j < i
                box_block_path(i) = 1;
            end
        elseif dir_i(1) == 1 && abs(bbox_i(1) - bbox_j(3)) < 0.15 && (((bbox_i(4) < bbox_j(2)) &&  (bbox_j(2) < bbox_i(2))) || ((bbox_i(4) < bbox_j(4)) && (bbox_j(4) < bbox_i(2))))

            if box_block_path(i) ~= 1 && j > i
                box_block_path(j) = 1;
            elseif box_block_path(i) ~= 1 && j < i
                box_block_path(i) = 1;
            end
        elseif dir_i(2) == -1 && abs(bbox_i(4) - bbox_j(2)) < 0.15 && (((bbox_i(3) < bbox_j(1)) &&  (bbox_j(1) < bbox_i(1))) || ((bbox_i(3) < bbox_j(3)) && (bbox_j(3) < bbox_i(1))))
            if box_block_path(i) ~= 1 && j > i
                box_block_path(j) = 1;
            elseif box_block_path(i) ~= 1 && j < i
                box_block_path(i) = 1;
            end
        elseif dir_i(2) == 1 && abs(bbox_i(2) - bbox_j(4)) < 0.15 && (((bbox_i(3) < bbox_j(1)) &&  (bbox_j(1) < bbox_i(1))) || ((bbox_i(3) < bbox_j(3)) && (bbox_j(3) < bbox_i(1))))
        
            if box_block_path(i) ~= 1 && j > i
                box_block_path(j) = 1;
            elseif box_block_path(i) ~= 1 && j < i
                box_block_path(i) = 1;
            end
        end
    end
end

for i = 1:size(box_block_path)
    if(box_block_path(i) == 0 || box(i, 12)  == 8 || box(i, 12)  == 9 || box(i, 12)  == 10 || box(i, 12)  == 11 || box(i, 12)  == 13)
        continue
    end
    if valid_index(i) == 0
        continue
    end
    temp_box(i, 1:12) = box(i, :);
    temp_box(i, 13) = 1;
    valid_index(i) = false;
end

% disp(valid_index)

% UNDER TEST: remove the box has collision with boundary
% for i = 1:size(box, 1)
%     % find closed boundary
%     [~, distanceBoundary] = find_close_seg(box(i,:), boundary);
%     
%     % remove box if it is out of boundary
%     if any(distanceBoundary([1 2]) < 0)|| any(distanceBoundary([3 4]) > 0)
%         valid_index(i) = false;
%         temp_box(i, 1:12) = box(i, 1:12);
%         temp_box(i, 13) = 1;
%     end
% end


%% 3. check if there are one free space inside the room can put furniture that has been filtered in step2
% except night_stand

% livingIdx = find(rType==0);
% for i = 1:size(box, 1)
%     if i ~= livingIdx
%        if box(i,1)==box(i,3) || box(i,2)==box(i,4)
%            disp('Empty box!!!');
%        else
%            polyRoom = polyshape(box(i, [1 1 3 3]), box(i, [2 4 4 2]));
%            polyBoundary = subtract(polyBoundary,polyRoom);
%        end
%        
%     end
% end
% livingPoly = polyshape(box(livingIdx, [1 1 3 3]), box(livingIdx, [2 4 4 2]));
% 
% gap = polyBoundary;
% if gap.NumRegions == 1
%     [xLimit, yLimit] = boundingbox(gap);
%     box(livingIdx,:) = [xLimit(1) yLimit(1) xLimit(2), yLimit(2)];
% else
%     rIdx = find(isnan(gap.Vertices(:,1)));
%     rIdx = [rIdx; size(gap.Vertices,1)+1];
%     
%     % for each region, check if it intersects with the living room, 
%     % otherwise get the room label and find the room that should cover 
%     % the region
%     
%     region = cell(length(rIdx), 1);
%     overlapArea = zeros(length(rIdx), 1);
%     closeRoomIdx = zeros(length(rIdx), 1);
%     idx = 1;
%     for k = 1:length(rIdx)
%         regionV = gap.Vertices(idx:rIdx(k)-1, :);
%         idx = rIdx(k) + 1;
%         region{k} = polyshape(regionV);
%         
%         if overlaps(region{k}, livingPoly)
%             iter = intersect(region{k}, livingPoly);
%             overlapArea(k) = area(iter);
%         end
%         
%         [x, y] = centroid(region{k});
%         center = [x, y];
%         
%         dist = 256;
%         bIdx = 0;
%         for i = 1:size(box, 1)
%             b = box(i, :);
%             bCenter = double([(b(:,1)+b(:,3))/2, (b(:,2)+b(:,4))/2]);
%             d = norm(bCenter-center);
%             if d<dist
%                 dist = d;
%                 bIdx = i;
%             end
%         end
%         closeRoomIdx(k) = bIdx;
%     end 
%     
%     [~, lIdx] = max(overlapArea);
%     for k = 1:length(closeRoomIdx)
%         if k == lIdx
%             [xLimit, yLimit] = boundingbox(region{k});
%             box(livingIdx,:) = [xLimit(1) yLimit(1) xLimit(2), yLimit(2)];
%         else
%             room = polyshape(box(closeRoomIdx(k), [1 1 3 3]), box(closeRoomIdx(k), [2 4 4 2]));
%             [xLimit, yLimit] = boundingbox(union(room, region{k}));
%             box(closeRoomIdx(k),:) = [xLimit(1) yLimit(1) xLimit(2), yLimit(2)];
%         end
%     end        
% end

box(:, [1 2]) = box(:, [2 1]);
box(:, [3 4]) = box(:, [4 3]);
    
