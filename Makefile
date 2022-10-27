py=~/miniconda3/bin/python
threshold=acima-10
extractor=mobilenetv2

all: main.py
	$(py) main.py -i ../dataset_gimp/imagens_george/features/RGB/segmented_unet/256/patch\=3/specific_epithet/$(threshold)/$(extractor)/horizontal -l ./txt/$(threshold).txt
	$(py) main.py -i ../dataset_gimp/imagens_george/features/RGB/segmented_unet/400/patch\=3/specific_epithet/$(threshold)/$(extractor)/horizontal -l ./txt/$(threshold).txt
	$(py) main.py -i ../dataset_gimp/imagens_george/features/RGB/segmented_unet/512/patch\=3/specific_epithet/$(threshold)/$(extractor)/horizontal -l ./txt/$(threshold).txt