py=~/miniconda3/bin/python
colormode=RGB
threshold=20
taxon=specific_epithet
metric=f1_weighted

${py} main.py -i ../dataset_gimp/imagens_george/features/${colormode}/segmented_unet/512/patch\=3/${taxon}/20/mobilenetv2/horizontal -l ../dataset_gimp/imagens_george/imagens/${colormode}/${taxon}/512/20/label.txt -m ${metric}

for res in 256 400 512; do
	for e in mobilenetv2 resnet50v2 vgg16; do
		${py} main.py -i ../dataset_gimp/imagens_george/features/${colormode}/segmented_unet/${res}/patch\=3/${taxon}/${threshold}/${e}/horizontal -l ../dataset_gimp/imagens_george/imagens/${colormode}/${taxon}/${res}/${threshold}/label.txt -m ${metric}
	done
done
