import csv
import os
import tarfile


def save_csv(df, filename, path, header=True, index=True):
    filename = os.path.join(path, '%s.csv' % filename)
    print('[CSV] save %s' % filename)
    df.to_csv(filename, sep=';', na_rep='', header=header, quoting=csv.QUOTE_ALL, index=index)


def save_xlsx(df, filename, path, header=True, index=True):
    filename = os.path.join(path, '%s.xlsx' % filename)
    print('[XLSX] save %s' % filename)
    df.to_excel(filename, na_rep='', engine='xlsxwriter', header=header, index=index)


def save_df(df, filename, path, header=True, index=True):
    save_csv(df, filename, path, header=header, index=index)
    save_xlsx(df, filename, path, header=header, index=index)


def compress_folder(filename_compress, foldername):
    try:
        with tarfile.open(filename_compress, 'w:gz') as tar_file:
            tar_file.add(foldername)
        tar_file.close()
    except FileExistsError:
        print('%s exists' % filename_compress)
