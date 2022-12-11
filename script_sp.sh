#!/bin/bash
py=~/miniconda3/bin/python
metric=accuracy
for res in 256 400 512
do
	for e in mobilenetv2 resnet50v2 vgg16
	do
		for color in RGB grayscale
		do
			for segmented in manual unet
			do
			    echo ${res} ${e} ${color} ${segmented}
				${py} main.py -i ../dataset_gimp/imagens_sp/features/${color}/segmented_${segmented}/${res}/patch\=3/${e}/horizontal -l ../dataset_gimp/imagens_sp/imagens/${color}/segmented_${segmented}/${res}/label2.txt -m ${metric}
			done
		done
	done
done
