import pandas as pd
from scipy.signal import find_peaks

def _find_closest_last_sample(row, last_samples):
    if row.first_sample_closest_peak:
        diff = (last_samples[last_samples.notna()].index - row.name).total_seconds()
        diff = pd.Series(diff, index=last_samples[last_samples.notna()].index)
        diff = diff[diff > 0]
        idx = diff.idxmin()
        return idx
    else:
        return None


def _find_closest_peak(row, column, peaks):
    if row[column]:
        diff = abs((row.name - peaks[peaks.notna()].index).total_seconds())
        idx = pd.Series(diff, index=peaks[peaks.notna()].index).idxmin()
        return idx
    else:
        return None


def _find_closest_last_sample(row, last_samples):
    if row.first_sample_closest_peak:
        diff = (last_samples[last_samples.notna()].index - row.name).total_seconds()
        diff = pd.Series(diff, index=last_samples[last_samples.notna()].index)
        diff = diff[diff > 0]
        idx = diff.idxmin()
        return idx
    else:
        return None


def adjust_labels_to_peaks(
    workout,
    is_peak_minima=False,
    target_signal="linAccZ",
    exercise_column_name="exercise",
    no_exercise_label="NO_EXERCISE",
):
    signal = workout["data"]

    sign = 1
    if is_peak_minima:
        sign = -1

    # find real peaks in signal
    peaks, height = find_peaks(sign * signal[target_signal], height=-100, distance=1)
    peaks = pd.Series(
        sign * height["peak_heights"], index=signal[target_signal].index[peaks]
    )
    signal = signal.join(peaks.rename("peaks"))

    # check if the workout is a junk workout
    if workout["exercise"] != no_exercise_label:
        a = """ # find periods where the user is exercising
        is_exercising = signal[exercise_column_name] != no_exercise_label
        is_exercising = is_exercising.values

        # calculate exercising periods given the label
        counter = 1
        exercising_periods = np.zeros(len(is_exercising), dtype=int)
        for i in range(1, len(is_exercising)):
            if is_exercising[i] == 1:
                exercising_periods[i] = counter
            else:
                if is_exercising[i - 1] == 1:
                    counter += 1 """

        exercising_periods = signal["exercising_periods"].values

        # group exercising periods and extract repetitions
        first_sample = []
        last_sample = []
        g = signal.groupby(exercising_periods)
        del g.groups[0]
        for _, positions in g.groups.items():
            first_sample.append(positions[0])
            last_sample.append(positions[-1])

        first_sample = pd.Series(True, index=first_sample)
        last_sample = pd.Series(True, index=last_sample)

        signal = signal.join(first_sample.rename("first_sample"))
        signal.first_sample = signal.first_sample.fillna(False)
        signal = signal.join(last_sample.rename("last_sample"))
        signal.last_sample = signal.last_sample.fillna(False)

        # find closest peak to first sample
        res = signal.apply(
            _find_closest_peak,
            args=(
                "first_sample",
                signal.peaks,
            ),
            axis=1,
        )
        res = pd.Series(True, index=res[res.notna()])
        signal = signal.join(res.rename("first_sample_closest_peak"))
        signal.first_sample_closest_peak = signal.first_sample_closest_peak.fillna(
            False
        )

        # find closest peak to last sample
        res = signal.apply(
            _find_closest_peak,
            args=(
                "last_sample",
                signal.peaks,
            ),
            axis=1,
        )
        res = pd.Series(True, index=res[res.notna()])
        signal = signal.join(res.rename("last_sample_closest_peak"))
        signal.last_sample_closest_peak = signal.last_sample_closest_peak.fillna(False)

        signal = signal.drop(columns=[exercise_column_name])

    else:
        # if the workout is a junk workout, there's no true repetitions, so we set all the values to False
        signal["first_sample"] = False
        signal["last_sample"] = False
        signal["first_sample_closest_peak"] = False
        signal["last_sample_closest_peak"] = False

    workout["data"] = signal

    return workout