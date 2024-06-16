#!/usr/bin/env bash

python setup build_ext --inplace


for minimum in 5; do # 10 20; do
  for model in vgg16; # do mobilenetv2 resnet50v2; do
    for dataset in pr_dataset; do
      for size in 256; do # 400 512; do
        for color in GRAYSCALE; do # RGB; do
          for classifier in DecisionTreeClassifier; do # RandomForestClassifier KNeighborsClassifier MLPClassifier SVC; do
            python main.py -i /home/xandao/dataset/${dataset}+${minimum}/${color}/${size}/${model} -c ${classifier}
          done
        done
      done
    done
  done
done

#DATASET=regions_dataset
#for minimum in 5 10 20; do
#  for model in resnet50v2 vgg16 mobilenetv2; do
#    for region in North Northeast Middlewest South Southeast; do
#      for size in 256 400 512; do
#        python main.py --contrast 1.2 --formats npz --height ${size} -i ./datasetv2/new/formatted/${DATASET}+${region}+${minimum}/${color}/${size}/ -o /media/xandao/eec07521-c36a-4d2b-9047-0110e7749eae/datasetv2/new/features/${DATASET}+${region}+${minimum}/${color}/${size}/${model} --patches 3 --width ${size} --save_images --model ${model} --orientation horizontal
#      done
#    done
#  done
#done
