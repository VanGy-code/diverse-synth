function [newBox, constraint] = align_adjacent_furniture(box, tempBox, updated, type, threshold, closed_boundary)
% position of box1 relative to box2
% 0 left-above
% 1 left-below
% 2 left-of
% 3 above
% 4 inside
% 5 surrounding
% 6 below
% 7 right-of
% 8 right-above
% 9 right-below


newBox = box;

% disp("box:")
% disp(box)
% 
% disp("closed_boundary:")
% disp(closed_boundary)
% 
% disp("type:")
% disp(type)

% four point
constraint = zeros(4, 2);
idx = 1;

[strong_constraint_type, main_box, ref_box] = get_constraint(box, 23);
% disp("strong_constraint_type:")
% disp(strong_constraint_type)
% 
% disp("main_box:")
% disp(main_box)
% 
% disp("ref_box:")
% disp(ref_box)


if type == 0
    % box2 is left above of box1
    alignV(true, closed_boundary);
    alignH(true, closed_boundary);
elseif type == 1
    % box2 is left below of box1
    alignV(true, closed_boundary);
    alignH(false, closed_boundary);
elseif type == 2
    % box2 is left of box1
    % align box2's max_x to box1's min_x
    
    if strong_constraint_type == 1
        if box(main_box, 10) == -1
            align_anyway([main_box,2], [ref_box,2]);
            align_anyway([main_box, 3], [ref_box, 1]);
        elseif box(main_box, 10) == 1
            align_anyway([main_box, 4], [ref_box, 4]);
            align_anyway([main_box, 3], [ref_box, 1]);
        elseif box(main_box, 9) == -1
            align_anyway([main_box, 4], [ref_box, 2]);
            align_anyway([main_box, 1], [ref_box, 1]);
        elseif box(main_box, 9) == 1
            align_anyway([main_box, 2], [ref_box, 4]);
            align_anyway([main_box, 3], [ref_box, 3]);
        end
        newBox(ref_box, 9:11) = newBox(main_box, 9:11);
    elseif strong_constraint_type == 2 || strong_constraint_type == 3 || strong_constraint_type == 4
        align_anyway([main_box, 3], [ref_box, 1]);
        alignV(true, closed_boundary);
        newBox(ref_box, 9:11) = [-newBox(main_box, 9), newBox(main_box, 10), newBox(main_box, 11)];
    elseif strong_constraint_type == 5
        align_anyway([main_box, 1], [ref_box, 3]);
        newBox(ref_box, 9:11) = [-newBox(main_box, 9), newBox(main_box, 10), newBox(main_box, 11)];
    else
        if newBox(1, 4) < newBox(2, 2) && newBox(1, 2) > newBox(2, 4)
            align([2,1], [1,3], threshold, closed_boundary);
        end
        
    end

elseif type == 3
    % box2 is above of box1
    % align box2's min_y to box1's max_y
    if strong_constraint_type == 1
        if box(main_box, 9) == -1
            align_anyway([main_box, 2], [ref_box, 4]);
            align_anyway([main_box, 1], [ref_box, 1]);
        elseif box(main_box, 9) == 1
            align_anyway([main_box, 2], [ref_box, 4]);
            align_anyway([main_box, 3], [ref_box, 3]);
        elseif box(main_box, 10) == -1
            align_anyway([main_box, 2], [ref_box, 2]);
            align_anyway([main_box, 1], [ref_box, 3]);
        elseif box(main_box, 10) == 1
            align_anyway([main_box, 4], [ref_box, 4]);
            align_anyway([main_box, 3], [ref_box, 1]);
        end
        newBox(ref_box, 9:11) = newBox(main_box, 9:11);
   
    elseif strong_constraint_type == 2 || strong_constraint_type == 3 || strong_constraint_type == 4
        align_anyway([main_box,2], [ref_box,4]);
        alignH(true, closed_boundary);
        newBox(ref_box, 9:11) = [newBox(main_box, 9), -newBox(main_box, 10), newBox(main_box, 11)];
    elseif strong_constraint_type == 5
        align_anyway([main_box, 4], [ref_box, 2]);
        newBox(ref_box, 9:11) = [newBox(main_box, 9), -newBox(main_box, 10), newBox(main_box, 11)];
    else
        if newBox(1, 3) < newBox(2, 1) && newBox(1, 1) > newBox(2, 3)
            align([2,4], [1,2], threshold, closed_boundary);
        end
    end
elseif type == 4
    % box2 is inside of box1
elseif type == 5
    % box2 is surrounding by box1
elseif type == 6
    % box2 is below of box1
    % align box2's min_y to box1's max_y
    if strong_constraint_type == 1
        if box(main_box, 9) == -1
            align_anyway([main_box, 1], [ref_box, 1]);
            align_anyway([main_box, 4], [ref_box, 2]);
        elseif box(main_box, 9) == 1
            align_anyway([main_box, 3], [ref_box, 3]);
            align_anyway([main_box, 4], [ref_box, 2]);
        elseif box(main_box, 10) == -1
            align_anyway([main_box, 2], [ref_box, 2]);
            align_anyway([main_box, 3], [ref_box, 1]);
        elseif box(main_box, 10) == 1
            align_anyway([main_box, 4], [ref_box, 4]);
            align_anyway([main_box, 1], [ref_box, 3]);
        end
        newBox(ref_box, 9:11) = newBox(main_box, 9:11);
    elseif strong_constraint_type == 2 || strong_constraint_type == 3 || strong_constraint_type == 4
        align_anyway([main_box, 4], [ref_box, 2]);
        alignH(false, closed_boundary);
        newBox(ref_box, 9:11) = [newBox(main_box, 9), -newBox(main_box, 10), newBox(main_box, 11)];
    elseif strong_constraint_type == 5
        align_anyway([main_box, 2], [ref_box, 4]);
        newBox(ref_box, 9:11) = [newBox(main_box, 9), -newBox(main_box, 10), newBox(main_box, 11)];
    else
        if newBox(1, 3) < newBox(2, 1) && newBox(1, 1) > newBox(2, 3)
            align([2,4], [1,2], threshold, closed_boundary);
        end
    end

elseif type == 7
    % box2 is right of box1
    % align box2's min_x to box1's max_x
    
    if strong_constraint_type == 1
        newBox(ref_box, 9:11) = newBox(main_box, 9:11);
        if box(main_box, 10) == -1
            align_anyway([main_box, 1], [ref_box, 3]);
            align_anyway([main_box, 2], [ref_box, 2]);
        elseif box(main_box, 10) == 1
            align_anyway([main_box, 1], [ref_box, 3]);
            align_anyway([main_box, 4], [ref_box, 4]);
        elseif box(main_box, 9) == -1
            align_anyway([main_box, 2], [ref_box, 4]);
            align_anyway([main_box, 1], [ref_box, 1]);
        elseif box(main_box, 9) == 1
            align_anyway([main_box, 4], [ref_box, 2]);
            align_anyway([main_box, 3], [ref_box, 3]);
        end
    elseif strong_constraint_type == 2 || strong_constraint_type == 3 || strong_constraint_type == 4
        align_anyway([main_box, 1], [ref_box, 3]);
        alignV(false, closed_boundary);
        newBox(ref_box, 9:11) = [-newBox(main_box, 9), newBox(main_box, 10), newBox(main_box, 11)];
    elseif strong_constraint_type == 5
        align_anyway([main_box, 3], [ref_box, 1]);
        newBox(ref_box, 9:11) = [-newBox(main_box, 9), newBox(main_box, 10), newBox(main_box, 11)];
    else
        if newBox(1, 4) < newBox(2, 2) && newBox(1, 2) > newBox(2, 4)
            align([2,3], [1,1], threshold, closed_boundary);
        end
    end
    
elseif type == 8
    % box2 is right above of box1
    alignV(false, closed_boundary);
    alignH(true, closed_boundary);
elseif type == 9
    % box2 is right below of box1
    alignV(false, closed_boundary);
    alignH(false, closed_boundary);
end

constraint = constraint(1:idx-1, :);

% disp("new Box:")
% disp(newBox)
% disp("============================")

function alignV(isLeft, closed_boundary)
    if isLeft
        idx1 = 1;
        idx2 = 3;
    else
        idx1 = 3;
        idx2 = 1;
    end
    
    if abs(tempBox(2,idx1) - tempBox(1,idx2)) <= abs(tempBox(2,idx2) - tempBox(1,idx2))
        align([2,idx1], [1,idx2], threshold, closed_boundary)
    else
        align([2,idx2], [1,idx2], threshold, closed_boundary)
    end
end

function alignH(isAbove, closed_boundary)
    if isAbove
        idx1 = 2;
        idx2 = 4;
    else
        idx1 = 4;
        idx2 = 2;
    end
    
    if abs(tempBox(2,idx1) - tempBox(1,idx2)) <= abs(tempBox(2,idx2) - tempBox(1,idx2))
        align([2,idx1], [1,idx2], threshold, closed_boundary)
    else
        align([2,idx2], [1,idx2], threshold, closed_boundary)
    end
end

function align(idx1, idx2, threshold, ~)
    % furniture relation under strong confinement
%     disp("box:")
%     disp(box)

%     disp(closed_boundary)
    % if one border of box_1 is close to another border of box_2 then do align
    distance = box(idx1(1),idx1(2)) - tempBox(idx2(1), idx2(2));

%     disp("distance:")
%     disp(distance)
    
    if (distance > 0 && abs(distance) <= threshold) || (distance < 0 && abs(distance) <= threshold * 2)
%         disp("updated: ")
%         disp([updated(idx1(1), idx1(2)), updated(idx2(1), idx2(2))])

        % if box_1 is updated and box_2 is not
        if ~updated(idx2(1), idx2(2))
            % move box2
            % consider function space
            if mod(idx2(2), 2) == 0
                if box(10) ~= 0
                    newBox(idx2(1), [2 4]) = newBox(idx2(1), [2 4]) - distance - 0.25;
                elseif box(10) == 0 || (box(10) ~= 0 && distance < 0.25)
                    newBox(idx2(1), [2 4]) = newBox(idx2(1), [2 4]) - distance;
                end
            else
                if box(9) == 0
                    newBox(idx2(1), [1 3]) = newBox(idx2(1), [1 3]) + distance;
                elseif box(9) ~= 0 || (box(9) ~= 0 && distance < 0.25)
                    newBox(idx2(1), [1 3]) = newBox(idx2(1), [1 3]) + distance + 0.25;
                end
            end
        % if box_2 is updated and box1 is not
        elseif ~updated(idx1(1), idx1(2))
            if mod(idx1(2), 2) == 0
                if box(10) ~= 0
                    newBox(idx1(1), [2 4]) = newBox(idx1(1), [2 4]) + distance - 0.25;
                elseif box(10) == 0 || (box(10) ~= 0 && distance < 0.25)
                    newBox(idx1(1), [2 4]) = newBox(idx1(1), [2 4]) + distance;
                end
            else
                if box(9) == 0 || (box(9) ~= 0 && distance < 0.25)
                    newBox(idx1(1), [1 3]) = newBox(idx1(1), [1 3]) - distance;
                elseif box(9) ~= 0
                    newBox(idx1(1), [1 3]) = newBox(idx1(1), [1 3]) - distance + 0.25;
                end
            end

        end
        
        if idx1(1) == 1
            constraint(idx, :) = [idx1(2) idx2(2)];
        else
            constraint(idx, :) = [idx2(2) idx1(2)];
        end
        idx = idx + 1;
    end
end

function align_anyway(idx1, idx2)
    % furniture relation under strong confinement

    
    % if one border of box_1 is close to another border of box_2 then do align
    distance = box(idx1(1), idx1(2)) - tempBox(idx2(1), idx2(2));
%     disp("distance:")
%     disp(distance)
% 
%     disp("idx2:")
%     disp(idx2)
    
%     disp("updated: ")
%     disp([updated(idx1(1), idx1(2)), updated(idx2(1), idx2(2))])

    % move box_2
    if mod(idx2(2), 2) == 0
        newBox(idx2(1), [2 4]) = newBox(idx2(1), [2 4]) + distance;
    else
        newBox(idx2(1), [1 3]) = newBox(idx2(1), [1 3]) + distance;
    end

    newBox(idx2(1), 9:11) = newBox(idx1(1), 9:11);

    if idx1(1) == 1
        constraint(idx, :) = [idx1(2) idx2(2)];
    else
        constraint(idx, :) = [idx2(2) idx1(2)];
    end
    idx = idx + 1;
    end
end


