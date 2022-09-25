import os
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from log import arquivo_log, console_log


def retorna_titulo(acuracia, n_amostras, n_features, n_patch, nome_classificador, nome_extrator, tipo_calculo):
    # titulo = f'ext: {nome_extrator}, n_amostras: {n_amostras}, n_feat: {n_features},\n class: {nome_classificador}, acc: {round(acuracia, 4)}, tc: {tipo_calculo}'
    # return f'{titulo}, n_patch: {n_patch}' if n_patch else titulo
    titulo = f'Confusion Matrix\n({nome_extrator}+{nome_classificador}, Accuracy: {format(round(acuracia, 4) * 100, ".2f")}%)'
    return f'{titulo}, Patch: {n_patch}' if n_patch and n_patch > 1 else titulo


def retorna_diretorio_final(extrator):
    nome_extrator_diretorio_saida = extrator.nome.replace('.txt', '').replace('.npy', '')
    if extrator.n_patch:
        return f'./out/{nome_extrator_diretorio_saida}/ft={extrator.n_features}/pt={extrator.n_patch}'
    return f'./out/{nome_extrator_diretorio_saida}/ft={extrator.n_features}'


def configura(matriz_confusao, titulo):
    matriz_confusao.plot(cmap='Reds')
    plt.title(titulo, pad=20)
    plt.ylabel('y_test', fontsize=12)
    plt.xlabel('y_pred', fontsize=12)
    plt.gcf().subplots_adjust(bottom=0.15, left=0.25)


def imprime_matriz_confusao(diretorio, it, matriz_confusao, extrator, n_amostras, n_features, nome_classificador, acuracia, tipo_calculo):
    rotulos = ['$\it{Manekia}$', '$\it{Ottonia}$', '$\it{Peperomia}$', '$\it{Piper}$', '$\it{Pothomorphe}$']
    try:
        titulo = retorna_titulo(acuracia, n_amostras, n_features, extrator.n_patches, nome_classificador, extrator.nome, tipo_calculo)
        configura(ConfusionMatrixDisplay(matriz_confusao, display_labels=rotulos).plot(), titulo)

        nome_arquivo_final = os.path.join(diretorio, f'it{it}.png')

        plt.savefig(nome_arquivo_final)
    except Exception as e:
        arquivo_log.error(f'excecao extrator {extrator.nome}, class {nome_classificador}, erro {e}')
        console_log.info(f'excecao extrator {extrator.nome}, class {nome_classificador}, erro {e}')
