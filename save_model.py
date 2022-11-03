import joblib
import os
import pathlib
import pickle
import tarfile

def save_best_model(classifier, fold, path):
    pathlib.Path(os.path.join(path, str(fold))).mkdir(parents=True, exist_ok=True)
    filename = os.path.join(path, str(fold), 'best_model.pkl')
    print(f'save {filename}')

    try:
        with open(filename, 'wb') as file:
            joblib.dump(classifier, file)
        file.close()
    except FileExistsError:
        print(f'{filename} exists')

    compress_file_model(filename, fold, path)
    os.remove(filename)


def compress_file_model(filename_model, fold, path):
    filename = os.path.join(path, str(fold), 'best_model.tar.gz')
    print(f'compress file {filename}')
    try:
        with tarfile.open(filename, 'w:gz') as tar_file:
            tar_file.add(filename_model)
        tar_file.close()
    except FileExistsError:
        print(f'{filename} exists')
