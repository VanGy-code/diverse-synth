function [newBox] = align_centorid(boundary, boxes)
    newBox = boxes;
    bpoint = boundary(:, [1, 3]);
    polyRoom = polyshape(bpoint);
    [y, x] = centroid(polyRoom);

    newBox(:, [1, 3]) = newBox(:, [1, 3]) + y;
    newBox(:, [2, 4]) = newBox(:, [2, 4]) + x;

end