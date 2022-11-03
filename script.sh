py=~/miniconda3/bin/python
colormode=RGB
taxon=specific_epithet
metric=f1_weighted

for res in 512; do
	for e in mobilenetv2 resnet50v2 vgg16; do
		for th in 20 10 5; do
			${py} main.py -i ../dataset_gimp/imagens_george/features/${colormode}/segmented_unet/${res}/patch\=3/${taxon}/${th}/${e}/horizontal -l ../dataset_gimp/imagens_george/imagens/${colormode}/${taxon}/${res}/${th}/label.txt -m ${metric}
		done
	done
done

