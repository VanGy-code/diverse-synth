clear;clc;close;

data = load('21.mat');

[newBox, types, edge] = align_fp(data.boundary, data.boxes, data.type, data.edges, 0.35, data.colors,true);

