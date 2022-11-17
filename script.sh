#!/bin/bash

py=~/miniconda3/bin/python
colormode=RGB
taxon=genus
threshold=2
metric=accuracy

for res in 256 400 512; do
	for e in mobilenetv2 resnet50v2 vgg16; do
		echo ${e} ${res} ${threshold} ${colormode} ${metric}
		#		${py} main.py -i ../dataset_gimp/imagens_george/features/${colormode}/segmented_unet/${res}/patch\=3/${taxon}/${threshold}/${e}/horizontal -l ../dataset_gimp/imagens_george/imagens/${colormode}/${taxon}/${res}/${threshold}/label2.txt -m ${metric}
		${py} main.py -i ../dataset_gimp/imagens_sp/features/${colormode}/segmented_unet/${res}/patch\=3/${e}/horizontal -l ../dataset_gimp/imagens_sp/imagens/${colormode}/segmented_unet/${res}/label2.txt -m ${metric}
	done
done
