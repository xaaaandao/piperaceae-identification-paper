import numpy


def get_samples_with_patch(x, y, list_index, n_patch):
    new_x = numpy.zeros(shape=(0, x.shape[1]))
    new_y = numpy.zeros(shape=(0,))

    for index in list_index:
        start = (index * n_patch)
        end = start + n_patch
        new_x = numpy.concatenate([new_x, x[start:end]])
        new_y = numpy.concatenate([new_y, y[start:end]])

    return new_x, new_y

