import joblib
import numpy
import time
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import dados
from log import console_log, escreve_log_arquivo_console
from resultado import TipoCalculo, Resultado, cria_arquivo_resultado, regra_do_maior, gera_resultado, \
    retorna_todos_resultados_por_indice_cv, retorna_todos_resultados_y_pred_y_test, \
    retorna_todas_resultado_por_calculo_tipo, regra_da_soma, regra_do_produto


def retorna_nome_classificador(classificador):
    if isinstance(classificador, SVC):
        return 'SVM'
    elif isinstance(classificador, KNeighborsClassifier):
        return 'k-NN'
    elif isinstance(classificador, RandomForestClassifier):
        return 'Random Forest'
    elif isinstance(classificador, MLPClassifier):
        return 'MLP'
    elif isinstance(classificador, StackingClassifier):
        return 'stack'
    return 'Decision Tree'


def retorna_hiperparams(classificador):
    if isinstance(classificador, SVC):
        return retorna_hiperparams_svm()
    elif isinstance(classificador, KNeighborsClassifier):
        return retorna_hiperparams_knn()
    elif isinstance(classificador, RandomForestClassifier):
        return retorna_hiperparams_floresta_aleatoria()
    elif isinstance(classificador, MLPClassifier):
        return retorna_hiperparams_mlp()
    elif isinstance(classificador, StackingClassifier):
        return retorna_hiperparams_stack()
    return retorna_hiperparams_arvore_decisao()


def retorna_melhor_classificador(classificador, dados):
    console_log.info(f'BUSCANDO MELHORES HIPERPARAMS {retorna_nome_classificador(classificador)}')
    modelo = GridSearchCV(classificador, retorna_hiperparams(classificador), scoring='accuracy', cv=5, verbose=42, n_jobs=-1)
    with joblib.parallel_backend("threading", n_jobs=-1):
        modelo.fit(dados['x'], dados['y'])
    return modelo.best_estimator_, modelo.best_params_


def retorna_hiperparams_stack():
    return {
        'stack_method': ['auto', 'predict_proba']
    }


def retorna_hiperparams_arvore_decisao():
    return {
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'max_depth': [10, 100, 1000]
    }


def retorna_hiperparams_svm():
    return {
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    }


def retorna_hiperparams_mlp():
    return {
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'solver': ['adam', 'sgd'],
        'learning_rate_init': [0.01, 0.001, 0.0001],
        'momentum': [0.9, 0.4, 0.1]
    }


def retorna_hiperparams_knn():
    return {
        'n_neighbors': [2, 4, 6, 8, 10],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }


def retorna_hiperparams_floresta_aleatoria():
    return {
        'n_estimators': [200, 400, 600, 800, 1000],
        'max_features': ['sqrt', 'log2'],
        'criterion': ['gini', 'entropy'],
        'max_depth': [10, 100, 1000]
    }


def retorna_arvore_decisao():
    return DecisionTreeClassifier(random_state=1)


def retorna_svm():
    return SVC(random_state=1, probability=True)


def retorna_mlp():
    return MLPClassifier(random_state=1)


def retorna_knn():
    return KNeighborsClassifier(n_jobs=-1)


def retorna_floresta_aleatoria():
    return RandomForestClassifier(random_state=1, n_jobs=-1)


def retorna_todos_classificadores():
    # return [retorna_arvore_decisao(), retorna_floresta_aleatoria(), retorna_knn(), retorna_mlp(), retorna_svm()]
    return [retorna_svm()]


def treina_testa(modelo, x_treinamento, y_treinamento, x_teste):
    modelo.fit(x_treinamento, y_treinamento)
    return modelo.predict_proba(x_teste)


def retorna_indices_cv(nome_arquivo, n_splits, tam_treinamento, tam_teste):
    console_log.info(f'separando dados a partir do arquivo: {nome_arquivo}')
    x, y = dados.retorna_features_e_labels(numpy.loadtxt(nome_arquivo))
    indice_cv = StratifiedShuffleSplit(n_splits=n_splits, train_size=tam_treinamento, test_size=tam_teste, random_state=1)
    return [[indice_treinamento, indice_teste] for indice_treinamento, indice_teste in
            indice_cv.split(x, y)]


def retorna_dados_treinamento_teste(indice, x, y, n_patch):
    if n_patch:
        x_treinamento, y_treinamento = dados.retorna_dados_imgs_patch(x, y, indice[0], n_patch)
        x_teste, y_teste = dados.retorna_dados_imgs_patch(x, y, indice[1], n_patch)
        return x_treinamento, x_teste, y_treinamento, y_teste
    return x[indice[0]], x[indice[1]], y[indice[0]], y[indice[1]]


def cv(classificador, indices_cv, dados, n_patch, n_amostras, n_features):
    lista_resultados = []
    for indice, indice_dados in enumerate(indices_cv):
        x_treinamento, x_teste, y_treinamento, y_teste = retorna_dados_treinamento_teste(indice_dados, dados.x, dados.y,
                                                                                         n_patch)
        tempo_inicio = time.time()
        y_pred_prob = treina_testa(classificador, x_treinamento, y_treinamento, x_teste)
        resultados = gera_resultado(indice, y_pred_prob, y_teste, n_patch, tempo_inicio, n_amostras, n_features)
        if isinstance(resultados, list):
            lista_resultados = lista_resultados + resultados
        else:
            lista_resultados.append(gera_resultado(indice, y_pred_prob, y_teste, n_patch, tempo_inicio, n_amostras, n_features))
    return lista_resultados


def imprime_lista_classificadores(lista_classificadores):
    console_log.info('=========================================')
    console_log.info('RESULTADO')
    for classificador in lista_classificadores:
        console_log.info(f'classificador: {classificador.__getattribute__("nome")}, melhor_params: {classificador.__getattribute__("melhores_params")}')
        console_log.info(f'MAIOR acuracia media: {classificador.acuracia_media}, desvio_padrao: {classificador.desvio_padrao}, ')
        console_log.info(f'tipo calculo: {classificador.__getattribute__("tipo_calculo")}')
        console_log.info(f'tempo: {classificador.__getattribute__("tempo_execucao")} milisegundos')
        console_log.info(f'tempo: {round(classificador.__getattribute__("tempo_execucao")/1000)} segundos')
        console_log.info('=========================================')
        lista_tipo_calculo = [TipoCalculo.MAIOR, TipoCalculo.SOMA, TipoCalculo.MULT]
        for tc in lista_tipo_calculo:
            lista_todos_tipo_calculo_maior = retorna_todas_resultado_por_calculo_tipo(classificador.__getattribute__('todos_resultados'), tc)
            media = []
            for resultado in lista_todos_tipo_calculo_maior:
                console_log.info(f'indice_cv {resultado.indice_cv}, tipo_calculo: {resultado.tipo_calculo} e accuracia {round(resultado.acuracia, 4)}')
                media.append(resultado.acuracia)
            if len(media) > 0:
                console_log.info(f'n_amostras: {lista_todos_tipo_calculo_maior[0].n_amostras}, n_features: {lista_todos_tipo_calculo_maior[0].n_features}')
                console_log.info(f'media acuracia: {round(numpy.mean(numpy.array(media)), 4)}')
                console_log.info('=========================================')


def remove_lista_classificadores(lista_classificadores):
    return list(filter(lambda x: 'arvore' in x.nome.replace(' ', ''), lista_classificadores))


def combinador(extrator, dados, lista_classificadores):
    if len(lista_classificadores) < 2:
        escreve_log_arquivo_console('deve ter pelo menos dois classificadores')


    todos_resultados = []
    for i in range(0, dados.cv):
        tempo_inicio = time.time()
        lista_por_indice_cv = retorna_todos_resultados_por_indice_cv(lista_classificadores, i)

        '''
            img sem patch -> regra do maior
            img com patch -> tem as três regras, mas aqui eu SÓ utiliza os que tem a regra do MAIOR 
        '''
        todos_y_pred, y_test = retorna_todos_resultados_y_pred_y_test(lista_por_indice_cv)

        novo_y_pred_maior = numpy.empty((0,))
        y_pred_maior = []
        for x in zip(*todos_y_pred):
            novo_y_pred_maior = regra_do_maior(novo_y_pred_maior, numpy.array(x), y_pred_maior)

        todos_resultados.append(Resultado(i, TipoCalculo.MAIOR, None, novo_y_pred_maior, y_test, tempo_inicio, dados.n_amostras, dados.n_features))

    cria_arquivo_resultado(dados, extrator, [Classificador("combinador", None, None, todos_resultados)], dados.n_features)


def classifica(extrator, indices_cv):
    lista_classificadores = []
    for dados in extrator.lista_dados:
        console_log.info(f'n_amostras: {dados.n_amostras}, n_features: {dados.n_features}')
        console_log.info(f'+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        for classificador in retorna_todos_classificadores():
            console_log.info(f'utilizando o classsificador {retorna_nome_classificador(classificador)}')
            melhor_classificador, melhor_params = retorna_melhor_classificador(classificador, dados)
            lista_resultados = cv(melhor_classificador, indices_cv, dados, extrator.n_patches, dados.n_amostras, dados.n_features)
            lista_classificadores.append(Classificador(retorna_nome_classificador(classificador), classificador, melhor_params, lista_resultados))
        dados.lista_classificadores = lista_classificadores
        # imprime_lista_classificadores(lista_classificadores)
        # combinador(extrator, dados, lista_classificadores)
        cria_arquivo_resultado(dados, extrator, lista_classificadores, dados.n_features)


class Classificador():

    def __init__(self, nome, modelo, melhores_params, todos_resultados) -> None:
        super().__init__()
        self.nome = nome
        self.modelo = modelo
        self.melhores_params = melhores_params
        self.todos_resultados = todos_resultados
        self.tipo_calculo = None
        self.acuracia_media = self.define_melhor_acuracia_media()
        self.desvio_padrao = self.calcula_desvio_padrao()
        self.tempo_execucao = self.calcula_tempo()

    def define_tipo_calculo(self, acuracia_media_soma, acuracia_media_maior, acuracia_media_mult):
        maior_acuracia = max(acuracia_media_soma, acuracia_media_maior, acuracia_media_mult)
        if maior_acuracia == acuracia_media_maior:
            self.__setattr__('tipo_calculo', TipoCalculo.MAIOR)
        elif maior_acuracia == acuracia_media_mult:
            self.__setattr__('tipo_calculo', TipoCalculo.MULT)
        else:
            self.__setattr__('tipo_calculo', TipoCalculo.SOMA)

    def define_melhor_acuracia_media(self):
        acuracia_media_soma = self.calcula_acuracia(TipoCalculo.SOMA)
        acuracia_media_maior = self.calcula_acuracia(TipoCalculo.MAIOR)
        acuracia_media_mult = self.calcula_acuracia(TipoCalculo.MULT)
        self.define_tipo_calculo(acuracia_media_soma, acuracia_media_maior, acuracia_media_mult)
        return max(acuracia_media_soma, acuracia_media_maior, acuracia_media_mult)

    def calcula_acuracia(self, tipo_calculo):
        lista_acuracia_por_calculo = self.retorna_todas_acuracia_por_calculo_tipo(tipo_calculo)
        if len(lista_acuracia_por_calculo) > 0:
            return round(sum(lista_acuracia_por_calculo)/len(lista_acuracia_por_calculo), 4)
        return -1

    def retorna_todas_resultado_por_calculo_tipo(self, tipo_caclulo):
        return list(filter(lambda x: (x.tipo_calculo == tipo_caclulo), self.__getattribute__('todos_resultados')))

    def retorna_todas_acuracia_por_calculo_tipo(self, tipo_calculo):
        return [resultado.acuracia for resultado in self.retorna_todas_resultado_por_calculo_tipo(tipo_calculo)]

    def calcula_desvio_padrao(self):
        return round(numpy.std(numpy.array(self.retorna_todas_acuracia_por_calculo_tipo(self.tipo_calculo))), 4)

    def retorna_todos_tempos_por_calculo_tipo(self, tipo_calculo):
        return [resultado.tempo_execucao for resultado in self.retorna_todas_resultado_por_calculo_tipo(tipo_calculo)]

    def calcula_tempo(self):
        lista_tempo_por_calculo = self.retorna_todos_tempos_por_calculo_tipo(self.tipo_calculo)
        if len(lista_tempo_por_calculo) > 0:
            return round(sum(lista_tempo_por_calculo) / len(lista_tempo_por_calculo), 4)
