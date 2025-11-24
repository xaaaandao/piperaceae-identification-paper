#!/usr/bin/env bash

INPUT_DIR="/mnt/eec07521-c36a-4d2b-9047-0110e7749eae/Nextcloud/Dropbox import/datasets/2025/features/multimedia"
INPUT_DIR_AUG="/mnt/eec07521-c36a-4d2b-9047-0110e7749eae/Nextcloud/Dropbox import/datasets/2025/features/uem/patch=1/pr_dataset+20"
OUTPUT_DIR="/mnt/eec07521-c36a-4d2b-9047-0110e7749eae/Nextcloud/Dropbox import/datasets/2025/resultados/uem/classification/data-augmentation"

# for dataset in pr; do
#   for min_aug in "20" "30" "40" "50" "60" "70" "80" "100"; do
#     for color in RGB; do
#       for min in "20"; do
#         for size in "256"; do
#           for model in "vgg16"; do
#             for clf in "MLPClassifier"; do
#               FOLDER_NAME="Rotate+90+Rotate+180+Rotate+270+Transpose+VerticalFlip+HorizontalFlip"
#               python ./src/main.py -i "${INPUT_DIR}/pr_dataset/pr_dataset+${min}/${color}/${size}/${model}" -d "${INPUT_DIR_AUG}/pr_dataset+20+HorizontalFlip/features/${color}/${size}/${model}" -d "${INPUT_DIR_AUG}/pr_dataset+20+Rotate+90/features/${color}/${size}/${model}" -d "${INPUT_DIR_AUG}/pr_dataset+20+Rotate+180/features/${color}/${size}/${model}" -d "${INPUT_DIR_AUG}/pr_dataset+20+Rotate+270/features/${color}/${size}/${model}" -d "${INPUT_DIR_AUG}/pr_dataset+20+Transpose/features/${color}/${size}/${model}" -d "${INPUT_DIR_AUG}/pr_dataset+20+VerticalFlip/features/${color}/${size}/${model}" -c ${clf} -o "${OUTPUT_DIR}/pr_dataset+${min}+${FOLDER_NAME}+${model}+${clf}+min=${min_aug}" --min_data_aug ${min_aug}
#             done
#           done
#         done
#       done
#     done
#   done
# done


for dataset in pr; do
  for min_aug in "20" "30" "40" "50"; do
    for color in RGB; do
      for min in "20"; do
        for size in "256"; do
          for model in "vgg16"; do
            for clf in "MLPClassifier"; do
              for aug in "Rotate+90" "Rotate+180" "Rotate+270" "Transpose" "VerticalFlip" "HorizontalFlip"; do
                python ./src/main.py -i "${INPUT_DIR}/pr_dataset/pr_dataset+${min}/${color}/${size}/${model}" -d "${INPUT_DIR_AUG}/pr_dataset+20+${aug}/features/${color}/${size}/${model}" -c ${clf} -o "${OUTPUT_DIR}/pr_dataset+${min}+${aug}+${model}+${clf}+min=${min_aug}" --min_data_aug ${min_aug} 
              done
            done
          done
        done
      done
    done
  done
done

