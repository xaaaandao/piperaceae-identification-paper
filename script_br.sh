#!/bin/bash
PY=~/miniconda3/bin/python
TAXON=specific_epithet
METRIC=f1_weighted
DIR_FEATURES=../dataset_gimp/imagens_br/features
DIR_LABEL=../dataset_gimp/imagens_br/imagens

for img_size in 256 400 512; do
	for cnn in mobilenetv2 resnet50v2 vgg16; do
		for color in RGB grayscale;	do
			for patch in 1 3; do
				for threshold in 5 10 20; do
					echo ${cnn} ${img_size} ${threshold} ${color} ${METRIC}
					${PY} main.py -i ${DIR_FEATURES}/${color}/segmented_unet/${img_size}/patch=${patch}/${TAXON}/${threshold}/${cnn}/horizontal -l ${DIR_LABEL}/${color}/${TAXON}/${img_size}/${threshold}/label2.txt -m ${METRIC}
				done
			done
		done
	done
done

for img_size in 256 400 512; do
	for file in lbp surf surf64; do
		for threshold in 5 10 20; do
			color=grayscale
			segmented=unet
			${PY} main.py -i ${DIR_FEATURES}/${color}/segmented_${segmented}/${img_size}/patch=1/${TAXON}/${threshold}/${file}.txt -l ${DIR_LABEL}/${color}/segmented_${segmented}/${img_size}/${threshold}/label2.txt -m ${METRIC}
		done
	done
done