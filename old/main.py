import classificadores
import extrator


def main():
    cv = 5
    tam_treinamento = 0.8
    tam_teste = 0.2
    indices_cv = classificadores.retorna_indices_cv('./dataset/surf64.txt', cv, tam_treinamento, tam_teste)

    # arquivos avulsos
    # extrator.testa('./dataset/hog_cellsize_2_2.txt', cv, indices_cv, pca=True)
    # extrator.testa('./dataset/surf64.txt', cv, indices_cv, pca=False)

    # RODAR
    extrator.testa('./dataset/surf128.txt', cv, indices_cv, pca=True)
    extrator.testa('./dataset/lbp.txt', cv, indices_cv, pca=False)
    extrator.testa('./dataset/mobilenetv2-patch=1', cv, indices_cv, pca=False)
    extrator.testa('./dataset/resnet50v2-patch=1', cv, indices_cv, pca=True)
    extrator.testa('./dataset/vgg16-patch=1', cv, indices_cv, pca=True)

    # dados da rede neural
    n_patch = [3]
    rede_neural = ['mobilenetv2', 'vgg16', 'resnet50v2']
    tipo_patch = ['horizontal']
    for tp in tipo_patch:
        for n in n_patch:
            for rede in rede_neural:
                extrator.testa(f'./dataset/patch_{tp}/patch={n}/{rede}', cv, indices_cv, pca=True)

    # NAO RODAR
    # outros extratores
    # n_bloco = ['1x1', '3x3', '5x5']
    # extratores = ['BSIF', 'LBP', 'LPQ', 'OBIF2', 'OBIF', 'SURF64', 'SURF128']
    # for b in n_bloco:
    #     for e in extratores:
    #         extrator.testa(f'dataset/diego/{b}/features_{e}_{b}.txt', cv, indices_cv, pca=False)


if __name__ == '__main__':
    main()


