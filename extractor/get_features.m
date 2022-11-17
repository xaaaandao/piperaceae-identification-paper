function get_features()
    path_in = "/home/xandao/Documentos/GitHub/dataset_gimp/imagens_george/imagens/grayscale/segmented_unet/512/matlab/specific_epithet/todos";
    path_out = "/home/xandao/Documentos/GitHub/dataset_gimp/imagens_george/features/grayscale/segmented_unet/512/patch=1/specific_epithet/5";
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
%     iwssip
%     list_label = ["manekia", "ottonia", "peperomia", "piper", "pothomorphe"];
%     genus/peperomia-piper
%     list_label = ["peperomia", "piper"];
%     genus/todos
%     list_label = ["manekia", "peperomia", "piper", "pothomorphe", "sarcorhachis"];
%     especies/todos
    list_label = ["abutiloides", "aduncum", "aequale", "alata", "alnoides", "amalago", "amplum", "arboreum", "arifolia", "balansana", "barbarana", "blanda", "brasiliensis", "caldasianum", "caldense", "callosum", "calophylla", "catharinae", "caulibarbis", "cernuum", "circinnata", "clivicola", "concinnatoris", "corcovadensis", "crassinervium", "crinicaulis", "delicatula", "diaphanodies", "diaphanoides", "dilatatum", "diospyrifolium", "elongata", "emarginella", "flavicans", "fuligineum", "galioides", "gaudichaudianum", "glabella", "glabratum", "glaziovi", "glaziovii", "gracilicaulis", "hatschbachii", "hayneanum", "hemmandorfii", "hemmendorffii", "hemmendorfii", "hernandiifolia", "hilariana", "hispidula", "hispidum", "hydrocotyloides", "ibiramana", "lanceolat", "lanceolatopeltata", "lepturum", "leucaenum", "leucanthum", "lhotzkianum", "lhotzkyanum", "lindbergii", "lucaeanum", "lyma", "macedoi", "magnoliifolia", "malacophyllum", "mandiocana", "mandioccana", "martiana", "michelianum", "mikanianium", "mikanianum", "miquelianum", "mollicomum", "mosenii", "nitida", "nudifolia", "obtusa", "obtusifolia", "ouabianae", "ovatum", "pellucida", "pereirae", "pereskiaefolia", "pereskiifolia", "perlongicaulis", "permucronatum", "piritubanum", "pseudoestrellensis", "pseudolanceolatum", "psilostachya", "punicea", "quadrifolia", "radicosa", "reflexa", "regenelli", "regnellii", "reitzii", "renifolia", "retivenulosa", "rhombea", "rivinoides", "rizzinii", "rotundifolia", "rubricaulis", "rupestris", "sandersii", "schwackei", "solmsianum", "stroemfeltii", "subcinereum", "subemarginata", "subretusa", "subrubrispica", "subternifolia", "tenuissima", "tetraphylla", "trichocarpa", "trineura", "trineuroides", "tuberculatum", "umbellata", "umbellatum", "urocarpa", "vicosanum", "viminifolium", "warmingii", "xylosteoides", "xylosteroides"];
%     acima-5 
%     list_label=["aduncum", "alata", "amalago", "arboreum", "arifolia", "barbarana", "blanda", "caldasianum", "caldense", "catharinae", "cernuum", "circinnata", "corcovadensis", "crassinervium", "dilatatum", "diospyrifolium", "emarginella", "galioides", "gaudichaudianum", "glabella", "glabratum", "hatschbachii", "hayneanum", "hilariana", "hispidula", "hispidum", "hydrocotyloides", "lhotzkianum", "macedoi", "malacophyllum", "martiana", "mikanianum", "miquelianum", "mollicomum", "mosenii", "nitida", "obtusa", "pereirae", "pereskiaefolia", "pereskiifolia", "pseudoestrellensis", "regnellii", "reitzii", "rhombea", "rotundifolia", "rupestris", "solmsianum", "subretusa", "tetraphylla", "trineura", "trineuroides", "umbellatum", "urocarpa", "viminifolium", "xylosteoides"];
%     acima-10
%     list_label=["aduncum", "alata", "amalago", "arboreum", "barbarana", "blanda", "caldense", "catharinae", "cernuum", "corcovadensis", "crassinervium", "dilatatum", "gaudichaudianum", "glabella", "glabratum", "hispidula", "hispidum", "malacophyllum", "martiana", "mikanianum", "miquelianum", "mollicomum", "nitida", "pereskiaefolia", "pseudoestrellensis", "regnellii", "reitzii", "rotundifolia", "solmsianum", "tetraphylla", "trineura", "urocarpa", "viminifolium", "xylosteoides"];
%     acima-20
%     list_label=["aduncum", "amalago", "arboreum", "blanda", "caldense", "catharinae", "corcovadensis", "crassinervium", "gaudichaudianum", "glabella", "glabratum", "hispidum", "martiana", "mikanianum", "miquelianum", "rubricaulis", "stroemfeltii", "trichocarpa", "vicosanum", "xylosteroides"];
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