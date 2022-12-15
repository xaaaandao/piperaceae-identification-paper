

#!/bin/bash
PY=~/miniconda3/bin/python
TAXON=specific_epithet
METRIC=f1_weighted
DIR_FEATURES=../dataset_gimp/imagens_regioes/features
DIR_LABEL=../dataset_gimp/imagens_regioes/imagens

for img_size in 256 400 512; do
	for cnn in mobilenetv2 resnet50v2 vgg16; do
		for color in RGB grayscale;	do
		    for patch in 1 3; do
		        for threshold in 5 10 20; do
                    for regiao in Norte Nordeste Centro-Oeste Sudeste Sul; do
                        echo ${cnn} ${img_size} ${threshold} ${color} ${METRIC}
                        ${PY} main.py -i ${DIR_FEATURES}/${color}/segmented_unet/${img_size}/patch=${patch}/${regiao}/${TAXON}/${threshold}/${cnn}/horizontal -l ${DIR_LABEL}/${color}/${img_size}/${regiao}/${threshold}/label2.txt -m ${METRIC}
                    done
                done
		    done
		done
	done
done