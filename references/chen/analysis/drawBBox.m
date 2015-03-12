function showCroppedImage(id)
    global chenDataset;
    global boxCols;
    global boxColIdx;

    % Load dataset if not already loaded
    if ~numel(chenDataset)
        chenDataset = load('../500_image_dataset.mat');
    end

    % Declare box colours
    if ~numel(boxCols)
        boxCols = [
            255, 255, 255; % White
            0,   0,   0;   % Black
            255, 0,   0;   % Bright red
            0,   255, 0;   % Bright green
            0,   0,   255; % Bright blue
            0,   255, 255; % Cyan
            255, 255, 0;   % Yellow
            255, 0,   255; % Magenta
            255, 165, 0;   % Orange
            0,   100, 0;   % Dark green
        ];
    end

    % Start from index 1
    if ~numel(boxColIdx)
        boxColIdx = 1;
    end

    % Select a random image if id not given
    if ~exist('id')
        id = randi(numel(chenDataset.img_gt));
    end

    % Load selected image
    imgData = chenDataset.img_gt(id);
    I = imread(['../image/', imgData.filename]);
    for j = 1:size(imgData.bbox, 1)
        I = drawBox(I, imgData.bbox(j, :));

        % Increment colour id
        boxColIdx = boxColIdx + 1;
        if boxColIdx > size(boxCols, 1)
            boxColIdx = 1;
        end
    end

    % Write image with cropping boxes
    if ~exist('./images_withbbox/', 'dir')
        mkdir('./images_withbbox/');
    end
    imwrite(I, ['./images_withbbox/' imgData.filename]);
end

function img = drawBox(img, bbox)
    global boxCols;
    global boxColIdx;

    hmin = bbox(1); wmin = bbox(2); hmax = bbox(3); wmax = bbox(4);
    [wmin, hmin] = getFixedCoord(img, wmin, hmin);
    [wmax, hmax] = getFixedCoord(img, wmax, hmax);

    col = boxCols(boxColIdx, :);
    nw = wmax-wmin+1; nh = hmax-hmin+1;
    img(hmin, wmin:wmax, :) = repmat(col, nw, 1);
    img(hmax, wmin:wmax, :) = repmat(col, nw, 1);
    img(hmin:hmax, wmin, :) = repmat(col, nh, 1);
    img(hmin:hmax, wmax, :) = repmat(col, nh, 1);
end

function [c1, c2] = getFixedCoord(img, c1, c2)
    [h, w, ~] = size(img);
    if c1 < 1; c1 = 1; end
    if c2 < 1; c2 = 1; end
    if c1 > w; c1 = w; end
    if c2 > h; c2 = h; end
end
