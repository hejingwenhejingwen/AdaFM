function generate_LR_JPEG()
%% matlab code to genetate JPEG compressed images.

%% set parameters
% comment the unnecessary line
input_folder = '../../datasets/DIV2K800/DIV2K800_sub';
save_JPEG_folder = '../../datasets/DIV2K800/DIV2K800_sub_q80';

jpeg_quality = 80;

if exist('save_JPEG_folder', 'var')
    if exist(save_JPEG_folder, 'dir')
        disp(['It will cover ', save_JPEG_folder]);
    else
        mkdir(save_JPEG_folder);
    end
end

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

        if exist('save_JPEG_folder', 'var')
            imwrite(img, fullfile(save_JPEG_folder, [imname, '.jpg']), 'jpg', 'Quality', jpeg_quality);
        end
    end
end
end

