import logging
import os
from datetime import datetime


def retorna_nome_arquivo():
    return str(datetime.now().strftime('%d-%m-%y'))


def configura_handler(diretorio_base, eh_arquivo):
    if eh_arquivo:
        nome_arquivo = os.path.join(diretorio_base, f'{retorna_nome_arquivo()}.log')
        return logging.FileHandler(filename=nome_arquivo)
    return logging.StreamHandler()


def configura_level_log(log, eh_arquivo):
    if eh_arquivo:
        log.setLevel(logging.ERROR)
    else:
        log.setLevel(logging.INFO)


def configura(nome, eh_arquivo=False):
    diretorio_base = 'logs'
    if not os.path.isdir(diretorio_base):
        os.makedirs(diretorio_base)

    handler = configura_handler(diretorio_base, eh_arquivo)
    handler.setFormatter(logging.Formatter('[%(asctime)s] %(message)s'))
    log = logging.getLogger(nome)
    configura_level_log(log, eh_arquivo)
    log.addHandler(handler)

    return log


def escreve_log_arquivo_console(mensagem):
    arquivo_log.error(mensagem)
    console_log.info(mensagem)
    exit(1)


arquivo_log = configura('arquivo', eh_arquivo=True)
console_log = configura('log')