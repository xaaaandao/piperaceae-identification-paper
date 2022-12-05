#!/bin/bash
py=~/miniconda3/bin/python
taxon=specific_epithet
metric=f1_weighted
for img_size in 512
do
	for extractor in mobilenetv2 resnet50v2 vgg16
	do
		for color in RGB
		do
			for threshold in 5
			do
				echo ${extractor} ${img_size} ${threshold} ${color} ${metric}
				${py} main.py -i ../dataset_gimp/imagens_br/features/${color}/segmented_unet/${img_size}/patch\=3/${taxon}/${threshold}/${extractor}/horizontal -l ../dataset_gimp/imagens_br/imagens/${color}/${taxon}/${img_size}/${threshold}/label2.txt -m ${metric}
			done
		done
	done
done