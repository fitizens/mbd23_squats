import pandas as pd
import numpy as np

def _calculate_matching(chunk, true_exercises):
    """
    Calculates the matching between two sequnces in the same signal. It's used to calculate the
    percentage of matching between a potential false repetition and a true repetition.
    """
    differences = []
    chunk_range = pd.period_range(chunk["start"], chunk["end"], freq="1ms")
    for true_exercise in true_exercises:
        true_range = pd.period_range(
            true_exercise["start"], true_exercise["end"], freq="1ms"
        )
        intersection = chunk_range.intersection(true_range)
        diff_pct = np.abs((intersection.size / max(chunk_range.size, true_range.size)))
        differences.append(diff_pct)

    return differences

def generate_false_repetitions(
    workout,
    true_repetitions,
    max_chunks=4,
    label="NO_EXERCISE",
    discard_similarity_th=0.9,
):
    true_exercises = [
        {
            "start": series["series"].index[0],
            "end": series["series"].index[-1],
            "event": series["target"],
        }
        for series in true_repetitions
    ]

    peaks = workout["data"][
        ["peaks", "last_sample_closest_peak", "first_sample_closest_peak"]
    ]
    peaks = peaks[peaks["peaks"].notna()]

    # generate no-repetition chunks
    chunks = []
    for peak_ts, _ in peaks.iterrows():
        selector = peaks.loc[:peak_ts, :].tail(max_chunks + 1)[:-1]
        for index, _ in selector.iterrows():
            chunks.append(
                {
                    "start": index,
                    "end": peak_ts,
                    "length": (peak_ts - index).total_seconds() * 1000,
                }
            )

    # Remove chunks that are too close to the true repetitions
    chunks_clean = list(
        filter(
            lambda chunk: (
                np.array(_calculate_matching(chunk, true_exercises))
                < discard_similarity_th
            ).all(),
            chunks,
        )
    )

    data = []
    for chunk in chunks_clean:
        array = workout["data"].loc[chunk["start"] : chunk["end"], :]
        data.append(
            {
                "series": array,
                "duration": round(array.shape[0] / workout["sampling_rate"] * 1000, 2),
                "length": array.shape[0],
                "target": label,
                "sampling_rate": workout["sampling_rate"],
                "id": workout["id"],
            }
        )

    return data