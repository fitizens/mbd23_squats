import pandas as pd


def merge_data(data: list):
    """
    This function merge all data and creates the dataframe to build ML models.
    :param data: training data
    :return: dataframe
    """
    time_series = list()
    timedelta = pd.Timedelta(milliseconds=0)
    for info in data:
        df = info["series"]
        df.index = df.index + timedelta
        time_series.append(df)
        timedelta = pd.Timedelta(seconds=df.index.max().timestamp()) + pd.Timedelta(
            milliseconds=1000
        )
    return pd.concat(time_series)
