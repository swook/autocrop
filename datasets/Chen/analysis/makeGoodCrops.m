function makeGoodCrops()
    global chenDataset;
    chenDataset = load('../500_image_dataset.mat');
    if ~exist('./good_crops/', 'dir')
        mkdir('./good_crops/');
    end

    for i = 1:numel(chenDataset.img_gt)
        imgData = chenDataset.img_gt(i);
        fn = imgData.filename;
        [~, name, ext] = fileparts(fn);

        fprintf('Processing file: %s\n', fn);
        I = imread(['../image/' fn]);

        for j = 1:size(imgData.bbox, 1)
            bbox = imgData.bbox(j, :);
            hmin = bbox(1); wmin = bbox(2); hmax = bbox(3); wmax = bbox(4);
            [wmin, hmin] = getFixedCoord(I, wmin, hmin);
            [wmax, hmax] = getFixedCoord(I, wmax, hmax);

            if (hmax-hmin) < 1 || (wmax-wmin) < 1
                continue;
            end

            cfn = [name, '_', num2str(j), ext];
            imwrite(I(hmin:hmax, wmin:wmax), ['good_crops/', cfn]);
        end
    end
end

function [c1, c2] = getFixedCoord(img, c1, c2)
    [h, w, ~] = size(img);
    if c1 < 1; c1 = 1; end
    if c2 < 1; c2 = 1; end
    if c1 > w; c1 = w; end
    if c2 > h; c2 = h; end
end
