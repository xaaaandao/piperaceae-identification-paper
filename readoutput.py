import pathlib


def main():
    lista_media = []
    for path_arquivo in pathlib.Path('out').rglob('*.txt'):
        arquivo = open(path_arquivo)
        linhas = arquivo.readlines()
        lista_media.append((linhas[20].replace('\n', '').replace('media acuracia: ', ''), path_arquivo))

    print(*sorted([l for l in lista_media], reverse=True), sep='\n')
    arquivo.close()


if __name__ == '__main__':
    main()