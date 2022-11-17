function get_features()
    path_in = "/home/xandao/Documentos/GitHub/dataset_gimp/imagens_george/imagens/grayscale/genus/256/2/matlab";
    path_out = "/home/xandao/Documentos/GitHub/dataset_gimp/imagens_george/features/grayscale/segmented_unet/256/patch=1/genus/2";
%     path_in = "/home/xandao/Documentos/GitHub/dataset_gimp/imagens_sp/imagens/grayscale/segmented_manual/400/matlab";
%     path_out = "/home/xandao/Documentos/GitHub/dataset_gimp/imagens_sp/features/grayscale/segmented_manual/400/patch=1/";
    delete_file_exists();
    list_dir = sort_by_name(dir(path_in));
    
    %{
        quando lista o diretorio os dois primeiros valores sao '.' e '..'
        e o terceiro valor eh o nome das pastas ou arquivo
    %}
    for i=1:height(list_dir)
        filename = list_dir.name(i);
        if contains(list_dir.name(i), "jpeg", "IgnoreCase", true) && filename{1}(1) ~= "_"
            path_img = append(list_dir.folder(i), "/", filename);
            disp(path_img);            
            img = imread(path_img, "jpeg");
            label = get_label(filename);
            
            feature = lbp(img);
            filename_lbp = append(path_out, "/" , "lbp.txt");
            fileout(filename_lbp, feature, string(label));
    
            feature = surf(img, 64);
            filename_surf = append(path_out, "/" , "surf64.txt");
            fileout(filename_surf, feature, string(label));
            
            feature = surf(img, 128);
            filename_surf = append(path_out, "/", "surf128.txt");
            fileout(filename_surf, feature, string(label));
        end
    end
end

function [sort_list_by_name] = sort_by_name(list)
    list = [list(3:length(list))];    
    list = struct2table(list);
    sort_list_by_name = sortrows(list, "name");
end

function delete_file_exists()
    fprintf('apagando arquivos...\n');
    for i=["surf.txt", "surf128.txt", "lbp.txt", "lbp2.txt"]
        delete(i);
    end
end

function i = get_label(filename)
%     dataset iwssip
    list_label=["manekia", "ottonia", "peperomia", "piper", "pothomorphe"];
%     dataset george
%     > 2       
%     list_label=["Peperomia","Piper"]
%     > 5
%     list_label=["nitida", "pereskiaefolia", "hydrocotyloides", "pseudoestrellensis", "xylosteoides", "umbellatum", "urocarpa", "gaudichaudianum", "miquelianum", "amalago", "tetraphylla", "arifolia", "mosenii", "caldense", "blanda", "arboreum", "dilatatum", "hayneanum", "glabratum", "caldasianum", "viminifolium", "rhombea", "malacophyllum", "galioides", "obtusa", "martiana", "mollicomum", "alata", "glabella", "rupestris", "rotundifolia", "hispidula", "aduncum", "catharinae", "mikanianum", "cernuum", "crassinervium", "hatschbachii", "diospyrifolium", "circinnata", "lhotzkianum", "barbarana", "regnellii", "hispidum", "trineura", "reitzii", "subretusa", "hilariana", "corcovadensis", "pereskiifolia", "macedoi", "emarginella", "solmsianum", "trineuroides", "pereirae"];
%     > 10
%     list_label = ["cernuum", "glabella", "reitzii", "barbarana", "rotundifolia", "catharinae", "amalago", "miquelianum", "viminifolium", "caldense", "nitida", "arboreum", "gaudichaudianum", "pseudoestrellensis", "dilatatum", "blanda", "mollicomum", "malacophyllum", "hispidula", "trineura", "crassinervium", "mikanianum", "aduncum", "glabratum", "martiana", "xylosteoides", "alata", "urocarpa", "corcovadensis", "hispidum", "solmsianum", "pereskiaefolia", "regnellii", "tetraphylla"];
%     > 20
%     list_label = ["miquelianum", "tetraphylla", "glabratum", "caldense", "urocarpa", "blanda", "glabella", "xylosteoides", "gaudichaudianum", "catharinae", "mikanianum", "corcovadensis", "crassinervium", "aduncum", "solmsianum", "martiana", "rotundifolia", "hispidum", "amalago", "arboreum"];
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

function feature = lbp(image)
    lbpFeatures = extractLBPFeatures(image);
    numNeighbors = 8;
    numBins = numNeighbors*(numNeighbors-1)+3;
    lbpCellHists = reshape(lbpFeatures, numBins, []);
    feature = reshape(lbpCellHists, 1, []);
end

function [featVector] = surf(I, SURFSize)

    points = detectSURFFeatures( I );
    [histograma, valid_points] = extractFeatures(I, points, 'SURFSize', SURFSize); 


    % escreve QTDE. DESCRITORES na tela
    vHist =  size(histograma, 1);

    % media
    vetorAux = mean(histograma, 1);
    media =  vetorAux(1:size(vetorAux, 2));

    % desvio padrao
    vetorAux = std(histograma, 0, 1);
    desvPad =  vetorAux(1:size(vetorAux, 2));

    % Obliquidade
    vetorAux = skewness(histograma, 0, 1);
    obliq =  vetorAux(1:size(vetorAux, 2));

    % Curtose
    vetorAux = kurtosis(histograma, 0, 1);
    curt = vetorAux(1:size(vetorAux, 2));

    featVector = [vHist, media, desvPad, obliq, curt] ;

end