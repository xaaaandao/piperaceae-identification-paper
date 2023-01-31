#!/bin/bash

py=~/miniconda3/bin/python

$py separate.py

for threshold in 5 10 20; do
	for regioes in Norte Centro-Oeste Nordeste Sul Sudeste; do
		for color in rgb grayscale; do
			mkdir -p /media/xandao/c2f58d30-ff2c-47f7-95af-91ad6fd69760/resultados/regioes/${regioes}
			mkdir -p /media/xandao/c2f58d30-ff2c-47f7-95af-91ad6fd69760/resultados/regioes/${regioes}/resultados/${color}/sheets_${threshold}
			$py read_result.py -i /media/xandao/c2f58d30-ff2c-47f7-95af-91ad6fd69760/resultados/regioes/${regioes} --threshold ${threshold} -o /media/xandao/c2f58d30-ff2c-47f7-95af-91ad6fd69760/resultados/regioes/${regioes}/resultados/${color}/sheets_${threshold} -c ${color} --taxon specific_epithet	
		done
	done
done
