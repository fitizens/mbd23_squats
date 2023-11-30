from fitizens_libraries.adjust_labels_to_peaks import _find_closest_last_sample


def get_true_repetitions(workout: dict):
    signal = workout["data"]
    last_samples = signal[signal.last_sample_closest_peak].last_sample_closest_peak
    last_samples = signal.apply(_find_closest_last_sample, args=(last_samples,), axis=1)
    exercises = last_samples[last_samples.notna()]
    exercises = exercises.reset_index()
    exercises.columns = ["start", "end"]

    data = []
    for exercise in exercises.iterrows():
        start = exercise[1]["start"]
        end = exercise[1]["end"]
        array = signal.loc[start:end, :]
        data.append(
            {
                "series": array,
                "duration": round(array.shape[0] / workout["sampling_rate"] * 1000, 2),
                "length": array.shape[0],
                "target": workout["exercise"],
                "sampling_rate": workout["sampling_rate"],
                "id": workout["id"],
            }
        )

    return data