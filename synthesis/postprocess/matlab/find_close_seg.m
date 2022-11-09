function [closedSeg, distSeg, idx] = find_close_seg(box, boundary)

%% need to carefully select the closed wall seg for each box
% cannot introduce a hole inside the boundary

% disp("box:")
% disp(box)

% select the boundary which have not been process

%% get the ordered horizontal and vertical segments on the boundary
% line up the points into boundaries

bSeg = [boundary(:, [1 3]), boundary([2:end 1], [1 3]), boundary([2:end 1], 4)];

% find the vertical boundaries
vSeg = bSeg(mod(boundary(:,4), 2)==1, :);

% swap vertical boundaries (make sure the line always represent as
% low_point -> high_point )

vSeg(:, [1 2]) = vSeg(:, [2 1]);
vSeg(:, [3 4]) = vSeg(:, [4 3]);

vSeg(vSeg(:,5)==0, [2 4]) = vSeg(vSeg(:,5)==0, [4 2]);

% sort vertical boundaries (low -> high)
[~, I] = sort(vSeg(:,1));
vSeg = vSeg(I,:);

hSeg = bSeg(mod(boundary(:,4), 2)==0, :);


hSeg(:, [1 2]) = hSeg(:, [2 1]);
hSeg(:, [3 4]) = hSeg(:, [4 3]);

hSeg(hSeg(:,5)== 3, [1 3]) = hSeg(hSeg(:,5)== 3, [3 1]);

% sort horizontal boundaries (left -> right)
[~, I] = sort(hSeg(:,2));
hSeg = hSeg(I,:);

closedSeg = ones(1,4)*6;
distSeg = ones(1,4)*6; 
idx = zeros(1, 4);

% disp("vSeg:")
% disp(vSeg)
% disp("hSeg:")
% disp(hSeg)

% check vertical seg
for i = 1:size(vSeg,1)
    seg = vSeg(i, :);
    
    % the vertical distance of box and boundary

    if  box(2) < seg(4) ||  box(4) > seg(2)
        continue
    end

    % the horizontal distance of box and boundary
    hdist = box([1 3]) - seg(1);

    dist1 = double([hdist(1)]); 
    dist3 = double([hdist(2)]);

%     disp("dist1:")
%     disp(dist1)
%     disp("dist3:")
%     disp(dist3)

    % find the closed boundary
    if abs(dist3) < abs(distSeg(1)) && seg(5) == 0
        distSeg(1) = dist3;
        idx(1) = i;
        closedSeg(1) = seg(1);
    elseif abs(dist1) < abs(distSeg(3)) && seg(5) == 2
        distSeg(3) = dist1;
        idx(3) = i;
        closedSeg(3) = seg(3);
    end
end

% check horizontal seg
for i = 1:size(hSeg,1)
    seg = hSeg(i, :);

    if  box(1) < seg(1) ||  box(3) > seg(3)
        continue
    end
    
    vdist = box([4 2]) - seg(2);

    dist2 = double([vdist(1)]);
    dist4 = double([vdist(2)]);

%      disp("dist2:")
%      disp(dist2)
%      disp("dist4:")
%      disp(dist4)
    
    if seg(5) == 1 && abs(dist4) < abs(distSeg(2))
        % the boundary is top of box
        distSeg(2) = dist4;
        idx(2) = i;
        closedSeg(2) = seg(2);
    elseif seg(5) == 3 && abs(dist2) < abs(distSeg(4))
        distSeg(4) = dist2;
        idx(4) = i;
        closedSeg(4) = seg(4);
    end

end
% disp("closedSeg: ")
% disp(closedSeg)
% disp("distSeg: ")
% disp(distSeg)
% disp("=================================")