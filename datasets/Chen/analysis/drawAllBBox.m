function drawAllBBox()
    global chenDataset;
    chenDataset = load('../500_image_dataset.mat');
    for i = 1:numel(chenDataset.img_gt)
        drawBBox(i);
        fprintf('Drew bounding boxes for id: %d, filename: %s\n', i, ...
            chenDataset.img_gt(i).filename);
    end
end
