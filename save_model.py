import os
import pathlib
import pickle
import tarfile


def save_best_model(classifier, fold, path):
    print(f'save {os.path.join(path, "best_model.pkl")}')

    pathlib.Path(os.path.join(path, str(fold))).mkdir(parents=True, exist_ok=True)
    filename_model = os.path.join(path, str(fold), 'best_model.pkl')

    with open(filename_model, 'wb') as file:
        pickle.dump(classifier, file)
    file.close()

    compress_file_model(filename_model, fold, path)

    os.remove(filename_model)


def compress_file_model(filename_model, fold, path):
    tar_filename = os.path.join(path, str(fold), 'best_model.tar.gz')
    with tarfile.open(tar_filename, 'w:gz') as tar_file:
        tar_file.add(filename_model)
    tar_file.close()
