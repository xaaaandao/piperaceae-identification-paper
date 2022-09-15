
function rgb2gray_resize()
    path = "../images/rgb/";
    list_dir = sort_dir_name(dir(path));
    
    %{
        quando lista o diretorio os dois primeiros valores sao '.' e '..'
        e o terceiro valor eh o nome das pastas ou arquivo
    %}
    for i=1:height(list_dir)
        list_file = dir(append(path, list_dir.name(i)));
        list_dir_files(path, list_dir.name(i), list_file);
    end
end

function [dir] = sort_dir_name(list_dir)
    list_dir = [list_dir(3:length(list_dir)-1)];    
    list_dir = struct2table(list_dir);
    dir = sortrows(list_dir,'name');
end

function [path] = path_to_img(path, dirname, filename)
    path = append(path, append(dirname, append('/', filename)));
end

function [] = list_dir_files(path, dirname, list_file)
    new_size = 400;
    for i=3:length(list_file)
        img = imread(path_to_img(path, dirname, list_file(i).name));
        img2gray = rgb2gray(img);
        img_resize = imresize(img2gray, [new_size, new_size]);
        
        path_out = "../images/";
        create_dir_if_not_exist(append(path_out, 'rgb2gray'));      
        new_dir = append('rgb2gray/', dirname);
        create_dir_if_not_exist(append(path_out, new_dir));              
        
        imwrite(img_resize, path_to_img(path_out, new_dir, list_file(i).name));
        
    end
end

function [] = create_dir_if_not_exist(path)
    if ~exist(path, 'dir')
        mkdir(path);
    end
end
