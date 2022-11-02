py=~/miniconda3/bin/python
colormode=RGB
threshold=5
taxon=specific_epithet
metric=f1_weighted

for res in 400; do
	for e in mobilenetv2; do
		${py} main.py -i ../dataset_gimp/imagens_george/features/${colormode}/segmented_unet/${res}/patch\=3/${taxon}/20/${e}/horizontal -l ../dataset_gimp/imagens_george/imagens/${colormode}/${taxon}/${res}/20/label.txt -m ${metric}
	done
done

for res in 256 400 512; do
	for e in resnet50v2 vgg16; do
		${py} main.py -i ../dataset_gimp/imagens_george/features/${colormode}/segmented_unet/${res}/patch\=3/${taxon}/${threshold}/${e}/horizontal -l ../dataset_gimp/imagens_george/imagens/${colormode}/${taxon}/${res}/${threshold}/label.txt -m ${metric}
	done
done
