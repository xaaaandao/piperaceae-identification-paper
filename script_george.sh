#!/bin/bash
TAXON=specific_epithet
METRIC=f1_weighted
DIR_FEATURES=../dataset_gimp/imagens_george/features
DIR_LABEL=../dataset_gimp/imagens_george/imagens

for img_size in 400; do
	for cnn in resnet50v2; do
		for color in RGB grayscale; do
			for patch in 3; do
				for threshold in 5 10; do
					echo ${cnn} ${img_size} ${threshold} ${color} ${patch} ${METRIC}
					python main.py -i ${DIR_FEATURES}/${color}/segmented_unet/${img_size}/patch=${patch}/${TAXON}/${threshold}/${cnn}/horizontal -l ${DIR_LABEL}/${color}/${TAXON}/${img_size}/${threshold}/label2.txt -m ${METRIC} --pca
				done
			done
		done
	done
done

# for img_size in 400; do
# 	for file in surf64; do
# 		for threshold in 5; do
# 			color=grayscale
# 			segmented=unet
# 			python main.py -i ${DIR_FEATURES}/${color}/segmented_${segmented}/${img_size}/patch=1/${TAXON}/${threshold}/${file}.txt -l ${DIR_LABEL}/${color}/${TAXON}/${img_size}/${threshold}/label2.txt -m ${METRIC} --pca
# 		done
# 	done
# done
