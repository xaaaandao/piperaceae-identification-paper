function get_features()
    path = "../images/rgb2gray/";
    delete_file_exists()
    list_dir = sort_by_name(dir(path));
    
    %{
        quando lista o diretorio os dois primeiros valores sao '.' e '..'
        e o terceiro valor eh o nome das pastas ou arquivo
    %}
    for i=1:height(list_dir)
        list_file = dir(append(path, list_dir.name(i)));
        list_dir_files(path, list_dir.name(i), list_file);
    end
end

function [sort_list_by_name] = sort_by_name(list)
    list = [list(3:length(list))];    
    list = struct2table(list);
    sort_list_by_name = sortrows(list,'name');
end

function [path] = path_to_img(path, dirname, filename)
    path = append(path, append(dirname, append('/', filename)));
end

function [] = list_dir_files(path, dirname, list_filename)
    list_filename = sort_by_name(list_filename);
    for i=1:height(list_filename)
        path_img = path_to_img(path, dirname, list_filename.name(i));
        img = imread(path_img);
        label = get_label(dirname);
        
        addpath(genpath('fileout'));
        fprintf('[%s-%d/%d] extraindo...\n', string(dirname), i, height(list_filename));
%         addpath(genpath('lbp')); % tipo include e sem ele nao funfa
%         feature = lbp(img);
%         fileout('lbp.txt', feature, label);
% 
%         features = extractLBPFeatures(img);
%         fileout("lbp2.txt", features, label);
%         
%         addpath(genpath('surf'));
%         feature = surf(img, 64);
%         fileout('surf64.txt', feature, label);
%         
%         addpath(genpath('surf'));
%         feature = surf(img, 128);
%         fileout('surf128.txt', feature, label);
%         [featureVector, hogVisualization] = extractHOGFeatures(img, 'BlockSize', [2 2]);
%         fileout(['hog_blocksize_2_2.txt'], featureVector, label);
        addpath(genpath('sift'));
        feature = sift(img);
        fileout(['sift3.txt'], feature, label);

    end
end

function delete_file_exists()
    fprintf('apagando arquivos...\n');
    files = ['surf.txt', 'surf1289.txt', 'lbp.txt', 'lbp2.txt'];
    for i=1:length(files)
         if isfile(files(i))
             fprintf('apagou o arquivo: %s\n', files(i))
             delete(files(i));
         end
    end
end

function [label] = get_label(dirname)
    label = extractAfter(string(dirname(1)),"f");
end

function fileout(filename, feature, label)    
    file = fopen(filename, 'a');
    for i=1:length(feature)
        fprintf(file, "%s ", num2str(feature(i)));
    end
    fprintf(file, "%s \n", label);
    fclose(file);
end

