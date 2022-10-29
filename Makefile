py=~/miniconda3/bin/python
colormode=RGB
threshold=10
extractor=mobilenetv2
taxon=specific_epithet
metric=f1_weighted

all: main.py
	for res in 256 400 512; do \
  		for e in mobilnetv2 resnet50v2 vgg16; do \
  		  $(py) main.py -i ../dataset_gimp/imagens_george/features/$(color_mode)/segmented_unet/$res/patch\=3/$(taxon)/$(threshold)/$e/horizontal -l /home/xandao/Documentos/dataset_gimp/imagens_george/imagens/$(color_mode)/$(taxon)/$res/$(threshold)/label.txt -m $(metric) \
  		done \
  	done
