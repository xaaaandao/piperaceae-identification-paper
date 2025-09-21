import logging
import pandas as pd

def save_csv_transpose(data, filename: str):
    df = pd.DataFrame(data, columns=list(data.keys()))
    df = df.transpose()
    df.to_csv(filename, sep=';', quoting=2, quotechar='"', encoding="utf-8", index=True, header=False)
    logging.info("saving %s" % filename)
