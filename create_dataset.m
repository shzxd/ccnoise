function create_dataset(row, col, prefix, start_idx, end_idx, postfix, savepath)
% 求n幅图像的均值图像
img_mean = make_mean(row, col, prefix, start_idx, end_idx, postfix);
% 求n幅图像的协方差矩阵的平均协方差矩阵
img_cov = make_cov(prefix, start_idx, end_idx, postfix, img_mean);
img_mask = make_mask(prefix, start_idx, end_idx, postfix, 0.05, img_mean);
img_mean = im2uint8(img_mean);
img_cov = single(img_cov);
img_mask = logical(img_mask);
save(savepath, 'img_mean', 'img_cov', 'img_mask');
