# Macro Función
# Esquema Mental
from fitizens_libraries.generate_false_repetitions import generate_false_repetitions
from fitizens_libraries.get_true_repetitions import get_true_repetitions
from fitizens_libraries.load_data import load_labeled_data


def load_training_data(
    filelist: list,
    signals: list,
    target_exercise: str,
    other_exercises: list,
    target_signal: str = "linAccZ",
    is_peak_minima: bool = False,
    time_column_name: str = "time",
    exercise_column_name: str = "exercise",
    smoothing_algorithm: str = "ZP-MA",
    smoothing_duration: int = 200,
    target_frequency: int = None,
    no_exercise_label="NO_EXERCISE",
    oos_repetitions=True,
    oos_repetitions_max_chunks=6,
) -> list:
    """Load data for trainig a model to detect repetitions of a given exercise. Returns a list of dictionaries with sequences representing repetitions and no repetitions.

    Parameters
    ----------
    filelist : list of str
        List of zip files to load data from. The zip files must contain a json file with the data in the format of the output of the labeling tool
    signals : list of str
        List of signals to include in the dataframe. The signals must be present in the input data
    target_exercise : str
        Exercise to detect repetitions
    other_exercises : list of str
        List of exercises to use as negative examples
    target_signal : str, optional
        Signal to use as target for the peak detection algorithm. Defaults to "linAccZ".
    is_peak_minima : bool, optional
        If True, the peaks are minima instead of maxima. Defaults to False.
    time_column_name : str, optional
        Name of the column containing the timestamp
    exercise_column_name : str, optional
        Name of the column containing the exercise label
    smoothing_algorithm : str, optional
        Smoothing algorithm to use. Supported values are [None,"MA","ZP-MA"]. Defaults to None.
        - None: No smoothing
        - MA: Moving average filter
        - ZP-MA: Zero phase moving average filter
    smoothing_duration : int, optional
        Duration of the smoothing window in milliseconds. Defaults to None.
    target_frequency : int, optional
        Target frequency to resample the data. Defaults to None.
    no_exercise_label : str, optional
        Label to use for negative examples. Defaults to "NO_EXERCISE".
    oos_repetitions : bool, optional
        If True, generate sequences on no repetitions from the serie. Defaults to True.
    oos_repetitions_max_chunks : int, optional
        Maximum number of chunks to generate for every detected peak. Defaults to 3.

    Returns
    -------
    list of dict
        List of dictionaries containing the sequences of repetitions and no repetitions
    """

    data = []

    # load labeled data
    workouts = load_labeled_data(
        filelist,
        signals=signals,
        target_signal=target_signal,
        is_peak_minima=is_peak_minima,
        time_column_name=time_column_name,
        exercise_column_name=exercise_column_name,
        smoothing_algorithm=smoothing_algorithm,
        smoothing_duration=smoothing_duration,
        target_frequency=target_frequency,
    )

    # process exercise workouts
    true_repetitions = []
    false_repetitions = []
    target_workouts = filter(
        lambda x: x[exercise_column_name] in [target_exercise], workouts
    )

    for wo in target_workouts:
        # get true repetitions
        treps = get_true_repetitions(wo)
        true_repetitions.extend(treps)

        # get false random repetitions from same exercise
        if oos_repetitions:
            freps = generate_false_repetitions(
                workout=wo,
                max_chunks=oos_repetitions_max_chunks,
                true_repetitions=treps,
                label=no_exercise_label,
            )
            false_repetitions.extend(freps)

    # process junk workouts
    junk_workouts = filter(
        lambda x: x[exercise_column_name] == no_exercise_label, workouts
    )
    for wo in junk_workouts:
        # get false random repetitions from junk workouts
        freps = generate_false_repetitions(
            workout=wo,
            true_repetitions=[],
            max_chunks=oos_repetitions_max_chunks,
            label=no_exercise_label,
        )
        false_repetitions.extend(freps)

    if true_repetitions:
        # get minimum length of true repetitions
        min_duration = min(true_repetitions, key=lambda x: x["duration"])["duration"]
        max_duration = max(true_repetitions, key=lambda x: x["duration"])["duration"]

        # filter false repetitions by duration
        false_repetitions = list(
            filter(
                lambda rep: max_duration >= rep["duration"] >= min_duration,
                false_repetitions,
            )
        )
        data.extend(true_repetitions)
        data.extend(false_repetitions)

        # get false random repetitions from other exercises
        if other_exercises:
            other_workouts = filter(
                lambda x: x[exercise_column_name] in other_exercises, workouts
            )
            for wo in other_workouts:
                others_true_repetitions = get_true_repetitions(wo)

                # change target label to no exercise
                for repetition in others_true_repetitions:
                    repetition["target"] = no_exercise_label

                data.extend(others_true_repetitions)

        return data, workouts
    else:
        raise ValueError("No repetitions found in the exercise data")