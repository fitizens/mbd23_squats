from fitizens_libraries.get_true_repetitions import get_true_repetitions
from fitizens_libraries.load_data import load_labeled_data
import pandas as pd

def load_timeseries_data(
    filelist: list,
    signals: list,
    time_column_name: str = "time",
    exercise_column_name: str = "exercise",
    is_peak_minima: bool = False,
    target_signal: str = "linAccZ",
    smoothing_algorithm: str = "ZP-MA",
    smoothing_duration: int = 200,
    target_frequency: int = None,
):
    """Load data for simulation, optimization and visualization purposes. Concatenates all the workouts in the filelist and returns a single timeseries with the labels as ranges.

    Parameters
    ----------
    filelist : list of str
        List of zip files to load data from. The zip files must contain a json file with the data in the format of the output of the labeling tool
    signals : list of str
        List of signals to include in the dataframe. The signals must be present in the input data
    time_column_name : str, optional
        Name of the column containing the timestamp
    exercise_column_name : str, optional
        Name of the column containing the exercise label
    is_peak_minima : bool, optional
        If True, the peaks are minima instead of maxima. Defaults to False.
    target_signal : str, optional
        Signal to use as target for the peak detection algorithm. Defaults to "linAccZ".
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
    pd.DataFrame
        Timeseries containing the data from all the workouts in the filelist
    list of dict
        List of dictionaries containing the label ranges
    """

    # load labeled data
    workouts = load_labeled_data(
        filelist=filelist,
        signals=signals,
        target_signal=target_signal,
        is_peak_minima=is_peak_minima,
        time_column_name=time_column_name,
        exercise_column_name=exercise_column_name,
        smoothing_algorithm=smoothing_algorithm,
        smoothing_duration=smoothing_duration,
        target_frequency=target_frequency,
    )

    timeseries = []
    labels_ranges = []
    false_labels_ranges = []
    timedelta = pd.Timedelta(milliseconds=0)
    for workout in workouts:
        # get timeseries from workout data
        wo = workout["data"]
        wo.index = wo.index + timedelta
        timeseries.append(wo)

        # update timedelta to avoid overlapping of next workout
        timedelta = pd.Timedelta(seconds=wo.index.max().timestamp()) + pd.Timedelta(
            milliseconds=1000
        )

        # get true repetitions
        true_repetitions = get_true_repetitions(workout)
        labels_ranges.extend(true_repetitions)

    def _get_label_range(repetition):
        newchunk = {}
        newchunk["start"] = repetition["series"].index[0]
        newchunk["end"] = repetition["series"].index[-1]
        newchunk["event"] = repetition["target"]
        return newchunk

    labels_ranges = [_get_label_range(repetition) for repetition in labels_ranges]
    false_labels_ranges = [
        _get_label_range(repetition) for repetition in false_labels_ranges
    ]

    timeseries = pd.concat(timeseries)

    return timeseries, labels_ranges