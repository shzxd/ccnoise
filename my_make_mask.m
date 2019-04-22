function M = my_make_mask(img_dir, t, img_mean)
g = fspecial('gaussian', [13, 13], 3);
b = imfilter(rgb2gray(img_mean), g);
b = b - mean(mean(b));

M = zeros(size(b, 1), size(b, 2));
    f = imfilter(rgb2gray(im2double(imread(img_dir))), g);
    f = f - mean(mean(f));
    M = or(M, abs(f - b) > t);

for i = 1:8:size(M, 1)
    for j = 1:8:size(M, 2)
        imin = min(i+7, size(M, 1));
        jmin = min(j+7, size(M, 2));
        if sum(sum(M(i:imin, j:jmin))) > 0
            M(i:imin, j:jmin) = 1;
        end
    end
end