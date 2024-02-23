import uuid
import zipfile
import json
import pandas as pd
import numpy as np
from custom_libraries.find_peaks import find_peaks_cust


def load_data_for_label(
    filelist: list,
    signals: list,
    target_signal: str = "linAccZ",
    time_column_name: str = "time",
    is_peak_minima: bool = False,
    smoothing_algorithm: str = "ZP-MA",
    smoothing_duration: int = 200,
    target_frequency: int = None
) -> list:
    
    """
    Load data from a list of zip files and process it to include smoothed signals, magnitudes and resampling to add a label.

    The format of the resulting data is a list of dictionaries representing a workout. It contains the following information
    - data: a Pandas dataframe containing the processed data
    - sampling_rate: the sampling rate of the data
    - filename: the name of the file containing the data

    Parameters
    ----------
    filelist : list of str
        List of zip files to load data from. The zip files must contain a json file with the data in the format of the output of the labeling tool
    signals : list of str
        List of signals to include in the dataframe. The signals must be present in the input data
    target_signal : str, optional
        Signal to use as target for the peak detection algorithm. Defaults to "linAccZ".
    time_column_name : str, optional
        Name of the column containing the timestamp
    smoothing_algorithm : str, optional
        Smoothing algorithm to use. Supported values are [None,"MA","ZP-MA"]. Defaults to None.
        - None: No smoothing
        - MA: Moving average filter
        - ZP-MA: Zero phase moving average filter
    smoothing_duration : int, optional
        Duration of the smoothing window in milliseconds. Defaults to None.
    target_frequency : int, optional
        Target frequency to resample the data. Defaults to None.

    Returns
    -------
    list of dict
        List of dictionaries containing the processed dataframes, the sampling rate and the filename
    """

    wodata = []
    for zipf in filelist:
        with zipfile.ZipFile(zipf) as z:
            for filename in sorted(z.namelist()):
                with z.open(filename) as f:
                    jsonfile = json.load(f)
                    wo = pd.DataFrame.from_records(jsonfile["data"])
                    wo = wo[[time_column_name] + signals]

                wo[time_column_name] = pd.to_datetime(wo[time_column_name])
                wo = wo.set_index(time_column_name)
                wo = wo.sort_index()

                origin_frequency = int(jsonfile["sampling_rate"])
                if target_frequency is None:
                    target_frequency = origin_frequency

                # smooth data according to the selected algorithm
                if smoothing_algorithm and (smoothing_duration > 0):
                    smoothing_samples = int(
                        smoothing_duration * origin_frequency / 1000
                    )

                    if smoothing_algorithm == "ZP-MA":
                        smoothed = wo[signals].rolling(smoothing_samples).mean()
                        smoothed = smoothed[::-1].rolling(smoothing_samples).mean()
                    else:
                        raise ValueError("Smoothing algorithm not supported")

                    smoothed = smoothed.fillna(method="bfill").fillna(method="ffill")

                    # calculate the smoothed signals magnitudes
                    magnitudes = smoothed**2

                    # include the smoothed signals and magnitudes in the dataframe
                    wo = wo.rename(
                        columns={signal: signal + "_orig" for signal in signals}
                    )

                    # substitute original signals with smoothed signals
                    wo = pd.concat([wo, smoothed], axis=1)

                    magnitudes = magnitudes.add_suffix("_mod")
                    wo = pd.concat([wo, magnitudes], axis=1)
                else:
                    # calculate the signals magnitudes over original signals
                    magnitudes = wo[signals] ** 2
                    magnitudes = magnitudes.add_suffix("_mod")
                    wo = pd.concat([wo, magnitudes], axis=1)

                # resample data to target frequency if greater from origin frequency
                if origin_frequency > target_frequency:
                    divisor = int(origin_frequency / target_frequency)
                    wo = wo.iloc[::divisor, :]

                wodata.append(
                    {
                        "data": wo,
                        "sampling_rate": target_frequency,
                        "filename": filename,
                        "id": uuid.uuid4(),
                    }

                )
    # apply algorithm to find peaks
    for i, workout in enumerate(wodata):
        workout = find_peaks_cust(
            workout=workout,
            is_peak_minima=is_peak_minima,
            target_signal=target_signal
        )
        wodata[i] = workout



    return wodata














