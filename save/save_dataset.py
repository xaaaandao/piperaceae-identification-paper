import pandas as pd

from save.save_files import save_df


def save_info_dataset(data, metric, path):
    index = get_index_sheet_dataset()
    values = get_values_sheet_dataset(data, metric)
    df = pd.DataFrame(values, index)
    filename = 'info'
    save_df(df, filename, path)


def get_values_sheet_dataset(data, metric):
    return [data['color_mode'], data['n_features'], data['n_samples'], data['dataset'], data['image_size'],
            data['dir'], data['extractor'], data['n_patch'], data['slice_patch'], metric]


def get_index_sheet_dataset():
    return ['color_mode', 'data_n_features', 'data_n_samples', 'dataset', 'dim_image', 'dir_input', 'extractor',
            'n_patch', 'slice', 'metric']
