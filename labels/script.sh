#!/bin/bash

py=~/miniconda3/bin/python

for res in 256 400 512; do
	for threshold in 2; do
		for taxon in genus; do
			$py labels.py -i ~/Documentos/GitHub/dataset_gimp/imagens_george/imagens/RGB/${taxon}/${res}/${threshold}/ -f ~/Documentos/GitHub/dataset_gimp/imagens_george/genus-and-specific_epithet-by-image.csv -t ${taxon}
		done
	done
done