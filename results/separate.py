import os
import pathlib
import re
import shutil

path = '/media/xandao/c2f58d30-ff2c-47f7-95af-91ad6fd69760/resultados/regioes/novos'
for regioes in ['Norte', 'Sul', 'Sudeste', 'Nordeste', 'Centro-Oeste']:
    list_cmd = []
    for p in pathlib.Path(os.path.join(path)).rglob('info.csv'):
        pathlib.Path(os.path.join('/media/xandao/c2f58d30-ff2c-47f7-95af-91ad6fd69760/resultados/regioes', regioes)).mkdir(exist_ok=True, parents=True)
        with open(p, 'r', encoding='latin1', errors='ignore') as file:
            if any(regioes in lines for lines in file.readlines()):
                _, _, _, _, _, _, _, date, _, _, _, _, _, _, _, _, _, _ = re.split('[./]', str(p))
                src = os.path.join(path, date)
                dst = path.replace('novos', regioes)
                dst = os.path.join(dst, date)
                # dst = os.path.join(regioes, date)
                print('src: %s' % src)
                print('dst: %s\n' % dst)
                # print(os.path.exists('./%s' % regioes))
                cmd = 'mv %s %s' % (src, dst)
                list_cmd.append(cmd)

            file.close()

    import numpy as np
    for cmd in np.unique(list_cmd):
        print(cmd)
        os.system(cmd)
