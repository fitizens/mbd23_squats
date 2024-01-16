import pandas as pd
from scipy.signal import find_peaks

def find_peaks_cust(
        workout,
        target_signal="linAccZ"
):
    signal = workout["data"]

    # Encontrar los picos mínimos en la señal
    peaks, height = find_peaks(-signal[target_signal], height=1, distance=1)
    
    # Crear una serie con las alturas de los picos mínimos y sus índices
    lowest_peaks = pd.Series(
        -height["peak_heights"], index=signal[target_signal].index[peaks]
    )
    
    # Añadir la información de los picos mínimos a la señal
    signal = signal.join(lowest_peaks.rename("peaks"))
    
    return signal



