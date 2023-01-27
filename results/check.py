import datetime
import pandas
import pathlib

def is_date(string):
    try:
        datetime.datetime.strptime(string, '%d-%m-%Y-%H-%M-%S')
        return True
    except ValueError:
        return False


path = '/media/xandao/c2f58d30-ff2c-47f7-95af-91ad6fd69760/resultados/george/specific_epithet/novos'
l=[]
for p in pathlib.Path(path).rglob('info.csv'):
	sheet = pandas.read_csv(p.resolve(), header=None, sep=';', index_col=0)
	# print(p, sheet.loc['dir_input'][1])
	# print(str(p)[str(p).index('unet'):len(str(p))])
	date = None
	for pp in str(p).split('/'):
		if is_date(pp):
			date = pp
	l.append({
		'c': str(p)[str(p).index('unet'):len(str(p))],
		'd': sheet.loc['dir_input'][1],
		'date': date
	})
print(l, sep='\n')
print()

lll=[]
for ll in l:
	for cc in l:
		if ll['c'] == cc['c'] and ll['d'] == cc['d'] and ll['date'] != cc['date']:
			print(ll['date'], cc['date'])
			lll.append({'date_one': ll['date'], 'date_two': cc['date']})

import numpy as np

print(np.unique(lll))
