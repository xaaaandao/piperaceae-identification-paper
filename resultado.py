import os
import time
from enum import Enum

import numpy
from sklearn.metrics import accuracy_score, confusion_matrix

from log import escreve_log_arquivo_console
from matriz_confusao import imprime_matriz_confusao


def cria_arquivo_resultado(dados, extrator, lista_classificadores, n_features):
    diretorio_base = f'./out/{extrator.nome}/pt={extrator.n_patches}/ft={n_features}'

    for classificador in lista_classificadores:
        diretorio_classificador = os.path.join(diretorio_base, f'{classificador.nome}')

        lista_tipo_calculo = [TipoCalculo.MAIOR, TipoCalculo.SOMA, TipoCalculo.MULT]
        for tc in lista_tipo_calculo:
            lista = retorna_todas_resultado_por_calculo_tipo(classificador.todos_resultados, tc)
            if len(lista) > 0:
                diretorio_tipo_calculo = os.path.join(diretorio_classificador, f'{tc}')

                if not os.path.isdir(diretorio_tipo_calculo):
                    os.makedirs(diretorio_tipo_calculo)

                arquivo = open(os.path.join(diretorio_tipo_calculo, f'{classificador.nome}.txt'), 'w')
                arquivo.write(f'saida do arquivo: {extrator.nome}\n')
                arquivo.write(f'=======================================================================\n')
                arquivo.write(f'classificador {classificador.__getattribute__("nome")}\n')
                arquivo.write(f'melhor_params: {classificador.__getattribute__("melhores_params")}\n')
                arquivo.write(
                    f'acuracia media: {classificador.acuracia_media}, desvio_padrao: {classificador.desvio_padrao}\n')
                arquivo.write(f'tipo calculo: {classificador.__getattribute__("tipo_calculo")}\n')
                arquivo.write(f'tempo ms: {classificador.__getattribute__("tempo_execucao")} microsegundos\n')
                arquivo.write(
                    f'tempo ss: {round(classificador.__getattribute__("tempo_execucao") / 1000, 4)} segundos\n')
                arquivo.write(f'=======================================================================\n')
                media = []
                media_tempo = []
                for resultado in sorted(lista, key=lambda x: x.indice_cv):
                    arquivo.write(f'indice_cv: {resultado.indice_cv}, tipo_calculo: {resultado.tipo_calculo}\n')
                    arquivo.write(f'accuracia: {round(resultado.acuracia, 4)}\n')
                    media.append(resultado.acuracia)
                    media_tempo.append(resultado.tempo_execucao)
                    imprime_matriz_confusao(diretorio_tipo_calculo, resultado.indice_cv, resultado.matriz_confusao,
                                            extrator, dados.n_amostras, dados.n_features, classificador.nome,
                                            resultado.acuracia, resultado.tipo_calculo)
                if len(media) > 0:
                    arquivo.write(f'n_amostras: {dados.n_amostras}, n_features: {dados.n_features}\n')
                    arquivo.write(f'media acuracia: {round(numpy.mean(numpy.array(media)), 4)}\n')
                    arquivo.write(f'desvio padrao: {round(numpy.std(numpy.array(media)), 4)}\n')
                    arquivo.write(f'media tempo (ms): {round(numpy.mean(numpy.array(media_tempo)), 4)}\n')
                    arquivo.write(f'media tempo (s): {round(numpy.mean(numpy.array(media_tempo))/1000, 4)}\n')
                    arquivo.write(f'=======================================================================\n')
    arquivo.close()


class TipoCalculo(Enum):
    SOMA = 1
    MAIOR = 2
    MULT = 3


class Resultado:

    def __init__(self, indice_cv, tipo_calculo, y_pred_antigo, y_pred_novo, y_teste, tempo, n_amostras,
            n_features) -> None:
        super().__init__()
        self.indice_cv = indice_cv
        self.tipo_calculo = tipo_calculo
        self.y_pred_antigo = y_pred_antigo
        self.y_pred_novo = y_pred_novo
        self.y_teste = y_teste
        self.acuracia = accuracy_score(y_true=self.y_teste, y_pred=self.y_pred_novo)
        self.matriz_confusao = confusion_matrix(y_true=self.y_teste, y_pred=self.y_pred_novo)
        self.tempo_execucao = self.calcula_tempo(tempo)
        self.n_amostras = n_amostras
        self.n_features = n_features

    def calcula_tempo(self, tempo_inicio):
        return (time.time() - tempo_inicio) * 1000000


def retorna_todas_resultado_por_calculo_tipo(lista_resultados, tipo_calculo):
    return list(filter(lambda x: (x.tipo_calculo == tipo_calculo), lista_resultados))


def todos_valores_label_iguais(y_test):
    return numpy.min(y_test) == numpy.max(y_test)


def novo_y_pred_prob(indice_cv, y_pred_prob, y_test, n_patch, tempo_inicio, n_amostras, n_features):
    novo_y_pred_soma = numpy.empty((0,))
    novo_y_pred_maior = numpy.empty((0,))
    novo_y_pred_mult = numpy.empty((0,))
    novo_y_test = numpy.empty((0,))
    y_pred_maior = []

    for i in range(0, y_test.shape[0], n_patch):
        start = i
        end = start + n_patch

        if not todos_valores_label_iguais(y_test[start:end]):  # garanti que todas os valores são iguais
            escreve_log_arquivo_console('erro nem todas as labels são iguais')

        novo_y_pred_soma = regra_da_soma(y_pred_prob[start:end], novo_y_pred_soma)
        novo_y_pred_mult = regra_do_produto(y_pred_prob[start:end], novo_y_pred_mult)
        novo_y_test = numpy.append(novo_y_test, y_test[start:end][0])
        y_pred = y_pred_prob[start:end]
        novo_y_pred_maior = regra_do_maior(novo_y_pred_maior, y_pred, y_pred_maior)

    resultado_soma = Resultado(indice_cv, TipoCalculo.SOMA, y_pred_prob, novo_y_pred_soma, novo_y_test, tempo_inicio,
                               n_amostras, n_features)
    resultado_mult = Resultado(indice_cv, TipoCalculo.MULT, y_pred_prob, novo_y_pred_mult, novo_y_test, tempo_inicio,
                               n_amostras, n_features)
    resultado_maior = Resultado(indice_cv, TipoCalculo.MAIOR, numpy.array(y_pred_maior), novo_y_pred_maior, novo_y_test,
                                tempo_inicio, n_amostras, n_features)
    return [resultado_soma, resultado_mult, resultado_maior]


def regra_do_produto(y_pred, novo_y_pred):
    return numpy.append(novo_y_pred, numpy.argmax(y_pred.prod(axis=0)) + 1)


def regra_da_soma(y_pred, novo_y_pred):
    return numpy.append(novo_y_pred, numpy.argmax(y_pred.sum(axis=0)) + 1)


def novo_y_pred_prob_sem_patch(y_pred_prob):
    y_pred = numpy.empty((0,))
    for y in y_pred_prob:
        y_pred = numpy.append(y_pred, (numpy.argmax(y) + 1))
    return y_pred


def gera_resultado(indice_cv, y_pred_prob, y_teste, n_patch, tempo_inicio, n_amostras, n_features):
    if n_patch:
        return novo_y_pred_prob(indice_cv, y_pred_prob, y_teste, n_patch, tempo_inicio, n_amostras, n_features)
    return Resultado(indice_cv, TipoCalculo.MAIOR, y_pred_prob, novo_y_pred_prob_sem_patch(y_pred_prob), y_teste,
                     tempo_inicio, n_amostras, n_features)


def retorna_todos_resultados_por_indice_cv(lista_classificadores, indice_cv):
    lista_por_indice_cv = []
    for classificador in lista_classificadores:
        lista_todos_tipo_calculo_maior = retorna_todas_resultado_por_calculo_tipo(
            classificador.__getattribute__('todos_resultados'), TipoCalculo.MAIOR)
        lista_todos_com_i_indice_cv = list(filter(lambda x: (x.indice_cv == indice_cv), lista_todos_tipo_calculo_maior))
        [lista_por_indice_cv.append(j) for j in lista_todos_com_i_indice_cv]
    return lista_por_indice_cv


def retorna_todos_resultados_y_pred_y_test(lista_por_indice_cv):
    todos_y_pred = []
    y_test = []
    for resultado in lista_por_indice_cv:
        todos_y_pred.append(resultado.y_pred_antigo)
        y_test = resultado.y_teste
    return todos_y_pred, y_test


def regra_do_maior(novo_y_pred_maior, y_pred, y_pred_maior):
    maior_valor = numpy.max(y_pred)
    # pos_um -> linha
    # pos_dois -> coluna -> representa a label
    pos_um, pos_dois = numpy.where(maior_valor == y_pred)  # retorna a posicao do maior valor
    novo_y_pred_maior = verifica_maior_valor(novo_y_pred_maior, pos_dois, pos_um, y_pred, y_pred_maior)
    return novo_y_pred_maior


def verifica_maior_valor(novo_y_pred_maior, pos_dois, pos_um, y_pred, y_pred_maior):
    if len(pos_um) > 1 and len(pos_dois) > 1:  # o maior valor eh repetido
        novo_y_pred_maior = tem_mais_de_um_maior_valor(novo_y_pred_maior, pos_dois, pos_um, y_pred, y_pred_maior)
    else:
        novo_y_pred_maior = tem_apenas_um_maior_valor(novo_y_pred_maior, pos_dois, pos_um, y_pred, y_pred_maior)
    return novo_y_pred_maior


def tem_apenas_um_maior_valor(novo_y_pred_maior, pos_dois, pos_um, y_pred, y_pred_maior):
    novo_y_pred_maior = numpy.append(novo_y_pred_maior, pos_dois[0] + 1)
    y_pred_maior.append(y_pred[pos_um[0]])
    return novo_y_pred_maior


def tem_mais_de_um_maior_valor(novo_y_pred_maior, pos_dois, pos_um, y_pred, y_pred_maior):
    # os maiores valores dizem que são labels iguais
    if numpy.min(pos_dois) == numpy.max(pos_dois):  # deu empate, pois os dois valores sao maiores
        novo_y_pred_maior = valores_maiores_label_igual(novo_y_pred_maior, pos_dois, pos_um, y_pred, y_pred_maior)
    else:
        # os maiores valores pertencem a labels diferentes
        novo_y_pred_maior = valores_maiores_label_diferentes(novo_y_pred_maior, y_pred, y_pred_maior)
    return novo_y_pred_maior


def valores_maiores_label_diferentes(novo_y_pred_maior, y_pred, y_pred_maior):
    soma = y_pred.sum(axis=0)  # somo as probabilidades
    novo_y_pred_maior = numpy.append(novo_y_pred_maior, numpy.argmax(soma) + 1)
    y_pred_maior.append(soma)
    return novo_y_pred_maior


def valores_maiores_label_igual(novo_y_pred_maior, pos_dois, pos_um, y_pred, y_pred_maior):
    novo_y_pred_maior = numpy.append(novo_y_pred_maior, pos_dois[0] + 1)
    y_pred_maior.append(y_pred[pos_um[0]])
    return novo_y_pred_maior
