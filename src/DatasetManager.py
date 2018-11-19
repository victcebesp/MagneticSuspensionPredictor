import pandas as pd


def read_dataset(dataset_url):
    return pd.read_excel(dataset_url)

def prepare_dataset(dataset_url, sample_size):

    dataframe = pd.DataFrame()

    input_dataframe = read_dataset(dataset_url)

    for index in range(0, len(input_dataframe) - sample_size):

        chop = input_dataframe.loc[index:index + sample_size - 1]

        row = chop['i(t)'].tolist() + chop['y(t)'].tolist()
        row.append(input_dataframe.loc[index + sample_size]['y(t)'])

        dataframe = dataframe.append([row], ignore_index=True)

    return dataframe