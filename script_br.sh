#!/bin/bash
py=~/miniconda3/bin/python
taxon=specific_epithet
metric=f1_weighted

for img_size in 256 400 512;
do
	for extractor in vgg16;
	do
		for color in RGB;
		do
			for threshold in 20;
			do
			    for regiao in "Sul";
			    do
                    echo ${extractor} ${img_size} ${threshold} ${color} ${metric}
                    ${py} main.py -i ../dataset_gimp/imagens_regioes/features/${color}/segmented_unet/${img_size}/patch\=3/${regiao}/${taxon}/${threshold}/${extractor}/horizontal -l ../dataset_gimp/imagens_regioes/imagens/${color}/${img_size}/${regiao}/${threshold}/label2.txt -m ${metric}
                done
			done
		done
	done
done