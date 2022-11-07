#!/bin/bash

py=~/miniconda3/bin/python

for res in 256 400 512; do
	for threshold in 5 10 20; do
		for taxon in specific_epithet genus; do
			$py file_labels.py -i ~/Documentos/GitHub/dataset_gimp/imagens_george/imagens/grayscale/specific_epithet/${res}/${threshold}/ -f ~/Documentos/GitHub/dataset_gimp/imagens_george/genus-and-specific_epithet-by-image.csv -t ${taxon}
		done
	done
done