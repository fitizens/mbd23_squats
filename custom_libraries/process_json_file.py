import uuid
import json
import pandas as pd
from custom_libraries.find_peaks import find_peaks_cust

def process_json_files(
        filelist: list,
        signals: list,
        target_signal: str = "linAccZ",
        time_column_name: str = "time",
        is_peak_minima: bool = False,
        smoothing_algorithm: str = "ZP-MA",
        smoothing_duration: int = 200,
        target_frequency: int = None
) -> list:

    wodata = []

    for json_file_path in filelist:
        with open(json_file_path, 'r') as f:
            jsonfile = json.load(f)

            wo = pd.DataFrame(jsonfile["workout"])
            wo = wo[[time_column_name] + signals]

            wo[time_column_name] = pd.to_datetime(wo[time_column_name],unit='ms')
            wo = wo.set_index(time_column_name)
            wo = wo.sort_index()
            origin_frequency = 104
            if target_frequency is None:
                target_frequency = origin_frequency

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
                magnitudes = smoothed ** 2
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
                    "filename": json_file_path,
                    "id": uuid.uuid4(),
                }
            )
    for i, workout in enumerate(wodata):
        workout = find_peaks_cust(
            workout=workout,
            is_peak_minima=is_peak_minima,
            target_signal=target_signal
        )
        wodata[i] = workout

    return wodata