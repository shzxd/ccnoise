% 图像数量
n = 50;
% 自建重压缩图像
predict_prefix = 'Twitter-data/Predict/P';
predict_postfix = '.jpg';
% Tweet图像
tweet_prefix = 'Twitter-data/Tweet/T';
tweet_postfix = '.jpg_large';
% 保存路径
savepath_prefix = 'Twitter-data/stats';
savepath_postfix = '.mat';

for i = 1:n
    % 准备路径
    pre_img_dir = strcat(predict_prefix, num2str(i), predict_postfix);
    mean_img_dir = strcat(tweet_prefix, num2str(i), tweet_postfix);
    savepath = strcat(savepath_prefix, num2str(i), savepath_postfix)
    % 开始计算
    img_mean = im2double(imread(mean_img_dir)) ./ 1;
    img_cov = my_make_cov(pre_img_dir, img_mean);
    img_mask = my_make_mask(pre_img_dir, 0.05, img_mean);

    img_mean = im2uint8(img_mean);
    img_cov = single(img_cov);
    img_mask = logical(img_mask);
    save(savepath, 'img_mean', 'img_cov', 'img_mask');
end
