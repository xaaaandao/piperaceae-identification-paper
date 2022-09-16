function rgb2gray_resize()
    path = "./OUT";
    new_size = 256;
        
    list_imgs = sort_dir_name(dir(path));
    %{
        quando lista o diretorio os dois primeiros valores sao '.' e '..'
        e o terceiro valor eh o nome das pastas ou arquivo
    %}
    for i=1:height(list_imgs)
        path = append(string(list_imgs.folder(i)), "/");
        path_img_input = append(path, append(list_imgs.name(i)));
        img_input = imread(path_img_input);
        % img2gray = rgb2gray(img_input);
        img_resize = imresize(img_input, [new_size, new_size]);
    
        imwrite(img_resize, append("./rgb2gray/", list_imgs.name(i)));
    end
end

function [dir] = sort_dir_name(list_dir)
    list_dir = [list_dir(3:length(list_dir))];    
    list_dir = struct2table(list_dir);
    dir = sortrows(list_dir,'name');
end