#!/bin/bash

py=~/miniconda3/bin/python

mkdir -p /media/xandao/c2f58d30-ff2c-47f7-95af-91ad6fd69760/resultados/iwssip/resultados/grayscale
mkdir -p /media/xandao/c2f58d30-ff2c-47f7-95af-91ad6fd69760/resultados/iwssip/resultados/rgb
$py read_result_sp.py -i /media/xandao/c2f58d30-ff2c-47f7-95af-91ad6fd69760/resultados/iwssip/novo --threshold 0 -o /media/xandao/c2f58d30-ff2c-47f7-95af-91ad6fd69760/resultados/iwssip/resultados/grayscale -c grayscale --taxon specific_epithet
$py read_result_sp.py -i /media/xandao/c2f58d30-ff2c-47f7-95af-91ad6fd69760/resultados/iwssip/novo --threshold 0 -o /media/xandao/c2f58d30-ff2c-47f7-95af-91ad6fd69760/resultados/iwssip/resultados/rgb -c rgb --taxon specific_epithet	

