#!/bin/bash

py=~/miniconda3/bin/python

for threshold in 5 10 20; do
	mkdir -p sheets_${threshold}
	$py read_result.py -i ~/Documentos/resultados_gimp/identificacao_george/especie/$threshold -o sheets_${threshold} -c RGB
done
