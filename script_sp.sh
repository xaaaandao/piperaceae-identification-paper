#!/bin/bash

py=~/miniconda3/bin/python
metric=accuracy

for res in 256 400 512; do
	for e in lbp surf128 surf64; do
		echo ${e} ${res} ${metric}
		${py} main.py -i ../dataset_gimp/imagens_sp/features/grayscale/segmented_unet/${res}/patch\=1/${e}.txt -l ../dataset_gimp/imagens_sp/imagens/grayscale/segmented_unet/${res}/label2.txt -m ${metric}
	done
done

for res in 256 400 512; do
	for e in mobilenetv2 resnet50v2 vgg16; do
		for color in RGB grayscale; do
			echo ${e} ${res} ${metric}
			${py} main.py -i ../dataset_gimp/imagens_sp/features/${color}/segmented_unet/${res}/patch\=3/${e}/horizontal -l ../dataset_gimp/imagens_sp/imagens/${color}/segmented_unet/${res}/label2.txt -m ${metric}
		done
	done
done
