function [type, main_box, ref_box] = get_constraint(box, type_nums)
    type = 0;
    main_box = -1;
    ref_box = -1;
    patterns = [];
    % 1. nightstand and end_table must near bedroom
    pattern1 = false(type_nums, 1);
    pattern1([2, 14]) = true;

    pattern2 = false(type_nums, 1);
    pattern2(3) = true;
   
    patterns = [patterns, [pattern1, pattern2]];

    % 2. chair must near desk
    pattern1 = false(type_nums, 1);
    pattern1(7) = true;

    pattern2 = false(type_nums, 1);
    pattern2(1) = true;

    patterns = [patterns, [pattern1, pattern2]];

    % 3. dining chair must near dining table
    pattern1 = false(type_nums, 1);
    pattern1(9) = true;

    pattern2 = false(type_nums, 1);
    pattern2(12) = true;

    patterns = [patterns, [pattern1, pattern2]];

    % 4. dressing chair must near dressing table
    pattern1 = false(type_nums, 1);
    pattern1(10) = true;

    pattern2 = false(type_nums, 1);
    pattern2(11) = true;

    patterns = [patterns, [pattern1, pattern2]];
    
    % 5. tea/coffee table must near sofa
%     pattern1 = false(type_nums, 1);
%     pattern1(21) = true;
% 
%     pattern2 = false(type_nums, 1);
%     pattern2(13) = true;
% 
%     patterns = [patterns, [pattern1, pattern2]];
    
    patterns = patterns';

    cnt = 1;
    for i = 1:2:size(patterns, 1)
        if (patterns(i, box(1, 12) + 1) && patterns(i + 1, box(2, 12)+ 1))
            type = cnt;
            main_box = 2;
            ref_box = 1;
        elseif(patterns(i + 1, box(1, 12)+ 1) && patterns(i, box(2, 12)+ 1))
            type = cnt;
            main_box = 1;
            ref_box = 2;
        end
        cnt = cnt + 1;
    end
end