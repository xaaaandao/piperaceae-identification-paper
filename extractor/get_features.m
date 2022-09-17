function get_features()
    path = "./images/RGB/512/";
    delete_file_exists();
    list_dir = sort_by_name(dir(path));
    
    %{
        quando lista o diretorio os dois primeiros valores sao '.' e '..'
        e o terceiro valor eh o nome das pastas ou arquivo
    %}
    for i=1:height(list_dir)
        path_img = append(list_dir.folder(i), "/", list_dir.name(i));
        img = imread(path_img);
        label = get_label(list_dir.name(i));
        
        addpath(genpath("lbp")); % tipo include e sem ele nao funfa
        feature = lbp(img);
        fileout("lbp.txt", feature, string(label));

        addpath(genpath("surf"));
        feature = surf(img, 64);
        fileout("surf64.txt", feature, label);
        
        addpath(genpath("surf"));
        feature = surf(img, 128);
        fileout("surf128.txt", feature, label);
    end
end

function [sort_list_by_name] = sort_by_name(list)
    list = [list(3:length(list))];    
    list = struct2table(list);
    sort_list_by_name = sortrows(list, "name");
end

function [path] = path_to_img(path, dirname, filename)
    path = append(path, append(dirname, append("/", filename)));
end

function delete_file_exists()
    fprintf('apagando arquivos...\n');
    for i=["surf.txt", "surf128.txt", "lbp.txt", "lbp2.txt"]
        delete(i);
    end
end

function i = get_label(filename)
    list_label = ["manekia", "ottonia", "peperomia", "piper", "pothomorphe"];
    for i=1:length(list_label)
        if contains(filename, list_label(i), "IgnoreCase", true)
            return;
        end
    end
end

function fileout(filename, feature, label)    
    file = fopen(filename, 'a');
    for i=1:length(feature)
        fprintf(file, "%s ", num2str(feature(i)));
    end
    fprintf(file, "%s \n", label);
    fclose(file);
end

