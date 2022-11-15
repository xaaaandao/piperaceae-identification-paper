#!/bin/bash

py=~/miniconda3/bin/python

for threshold in 2; do
	mkdir -p sheets_${threshold}
	$py read_result.py -i ~/Documentos/resultados_gimp/identificacao_george/genero/RGB/$threshold -o sheets_${threshold} -c RGB
done
