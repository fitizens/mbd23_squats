import pandas as pd
from scipy.signal import find_peaks

def find_peaks_cust(workout,
                    is_peak_minima,
                    target_signal="linAccZ"):
    """
       Find and annotate peaks in a workout signal.

       Parameters:
       - workout:
            Data containing workout information.
       - is_peak_minima: (bool)
            True if searching for minima peaks, False if searching for maxima peaks.
       - target_signal: (str)
            Name of the column in 'workout' containing the target signal (default is "linAccZ").

       """
    signal = workout["data"]
    signal_values = -signal[target_signal] if is_peak_minima else signal[target_signal]

    # Encontrar los picos en la señal
    peaks, height = find_peaks(signal_values, height=1, distance=1)

    # Crear una serie con las alturas de los picos y sus índices
    peaks_series = pd.Series(
        -height["peak_heights"] if is_peak_minima else height["peak_heights"],
        index=signal[target_signal].index[peaks]
    )

    # Añadir la información de los picos a la señal
    signal = signal.join(peaks_series.rename("peaks"))

    return signal
