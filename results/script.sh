#!/bin/bash

py=~/miniconda3/bin/python

for threshold in 5; do
	mkdir -p sheets_${threshold}
	$py read_result.py -i ~/Documentos/resultados_gimp/br --threshold ${threshold} -o sheets_${threshold} -c rgb --taxon specific_epithet
done
