function [featVector] = sift(I)

    %I = single( I );		

    points = detectSIFTFeatures( I );
    [histograma, valid_points] = extractFeatures(I, points); 


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