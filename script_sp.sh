#!/bin/bash
PY=~/miniconda3/bin/python
METRIC=accuracy
DIR_FEATURES=../dataset_gimp/imagens_sp/features
DIR_LABEL=../dataset_gimp/imagens_sp/imagens

for img_size in 256 400 512; do
	for segmented in manual unet; do
		for cnn in mobilenetv2 resnet50v2 vgg16; do
			for color in RGB grayscale; do
				for patch in 1 3; do
					echo ${img_size} ${cnn} ${color} ${segmented}
					${PY} main.py -i ${DIR_FEATURES}/${color}/segmented_${segmented}/${img_size}/patch=${patch}/${cnn}/horizontal -l ${DIR_LABEL}/${color}/segmented_${segmented}/${img_size}/label2.txt -m ${METRIC}
				done
			done
		done
		
	done
done

for img_size in 256 400 512; do
	for segmented in manual unet; do
		for file in lbp surf surf64; do
			color=grayscale
			${PY} main.py -i ${DIR_FEATURES}/${color}/segmented_${segmented}/${img_size}/patch=1/${file}.txt -l ${DIR_LABEL}/${color}/segmented_${segmented}/${img_size}/label2.txt -m ${METRIC}
		done
	done
done
