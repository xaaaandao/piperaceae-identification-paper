#!/usr/bin/env bash

python setup.py build_ext --inplace

for dataset in pr; do
  for augmentation in Affine; do
    for color in RGB; do
      for min in "20"; do
        for size in "512"; do
          for model in "vgg16"; do
            for clf in "MLPClassifier"; do
              python main.py -i "/mnt/eec07521-c36a-4d2b-9047-0110e7749eae/Nextcloud/Dropbox import/datasets/2025/uem/original/features/pr_dataset+${min}/features/${color}/${size}/${model}" -d "/mnt/eec07521-c36a-4d2b-9047-0110e7749eae/Nextcloud/Dropbox import/datasets/2025/uem/data-augmentation/features/pr_dataset+${min}+data-augmentation/${augmentation}/features/${color}/${size}/${model}" -c ${clf} -o "/mnt/eec07521-c36a-4d2b-9047-0110e7749eae/Nextcloud/Dropbox import/datasets/2025/uem/resultados/pr_dataset+${min}+${augmentation}+${color}+${size}+${model}+${clf}"
            done
          done
        done
      done
    done
  done
done
