function generate_LR_noise()
%% matlab code to genetate images with gaussian noise.

%% set parameters
% comment the unnecessary line
input_folder = '../../datasets/DIV2K800/DIV2K800_sub';
save_noise_img_folder = '../../datasets/DIV2K800/DIV2K800_sub_noise15';

noise_sigma = 15;

if exist('save_noise_img_folder', 'var')
    if exist(save_noise_img_folder, 'dir')
        disp(['It will cover ', save_noise_img_folder]);
    else
        mkdir(save_noise_img_folder);
    end
end

randn('seed', 0);

idx = 0;
filepaths = dir(fullfile(input_folder,'*.*'));
for i = 1 : length(filepaths)
    [paths,imname,ext] = fileparts(filepaths(i).name);
    if isempty(imname)
        disp('Ignore . folder.');
    elseif strcmp(imname, '.')
        disp('Ignore .. folder.');
    else
        idx = idx + 1;
        str_rlt = sprintf('%d\t%s.\n', idx, imname);
        fprintf(str_rlt);
        % read image
        img = imread(fullfile(input_folder, [imname, ext]));
        img = im2double(img);

        noise = noise_sigma/255.*randn(size(img));
        img_noise = single(img + noise);
        img_noise = im2uint8(img_noise);

        if exist('save_noise_img_folder', 'var')
            imwrite(img_noise, fullfile(save_noise_img_folder, [imname, '_noise', num2str(noise_sigma), '.png']));
        end
    end
end
end

