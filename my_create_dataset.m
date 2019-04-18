% 自建图像
predict_prefix = 'Real/predict/';
predict_name = 'gnu_pre_DPP0022';
predict_postfix = '.jpg';
pre_img_dir = strcat(predict_prefix, predict_name, predict_postfix);
% Tweet图像
tweet_prefix = 'Real/tweet/';
tweet_name = 'tweet_DPP0022';
tweet_postfix = '.jpg';
mean_img_dir = strcat(tweet_prefix, tweet_name, tweet_postfix);
% 保存路径
savepath = 'Real/stats.mat';

% 开始计算
img_mean = im2double(imread(mean_img_dir)) ./ 1;
img_cov = my_make_cov(pre_img_dir, img_mean);
img_mask = my_make_mask(pre_img_dir, 0.05, img_mean);

img_mean = im2uint8(img_mean);
img_cov = single(img_cov);
img_mask = logical(img_mask);
save(savepath, 'img_mean', 'img_cov', 'img_mask');