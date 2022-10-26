threshold=acima-20
extractor=resnet50v2

all:
	python main.py -i ../dataset_gimp/imagens_george/features/RGB/segmented_unet/256/patch\=3/specific_epithet/$(threshold)/mobilenetv2/horizontal -l ./txt/$(threshold).txt
	python main.py -i ../dataset_gimp/imagens_george/features/RGB/segmented_unet/512/patch\=3/specific_epithet/$(threshold)/mobilenetv2/horizontal -l ./txt/$(threshold).txt
	python main.py -i ../dataset_gimp/imagens_george/features/RGB/segmented_unet/400/patch\=3/specific_epithet/$(threshold)/mobilenetv2/horizontal -l ./txt/$(threshold).txt
	python main.py -i ../dataset_gimp/imagens_george/features/RGB/segmented_unet/256/patch\=3/specific_epithet/$(threshold)/$(extractor)/horizontal -l ./txt/$(threshold).txt
	python main.py -i ../dataset_gimp/imagens_george/features/RGB/segmented_unet/512/patch\=3/specific_epithet/$(threshold)/$(extractor)/horizontal -l ./txt/$(threshold).txt
	python main.py -i ../dataset_gimp/imagens_george/features/RGB/segmented_unet/400/patch\=3/specific_epithet/$(threshold)/$(extractor)/horizontal -l ./txt/$(threshold).txt
