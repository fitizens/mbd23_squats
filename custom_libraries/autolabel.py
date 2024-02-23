from custom_libraries.create_training_data_stats import create_training_data_stats
import pandas as pd
import warnings
from pycaret.classification import *

# diccionario con los modelos según el tipo de ejercicio
exercise_models = {
    'SQUAT': 'squat_model',
    'PUSHUP': 'Push_upModel',
    'BURPEE': 'BURPEE',
    'PULLUP': 'pullup_model',
    'SITUP': 'situp_upModel',
    'SQUAT JUMP': 'jump_model',
    'JUMPING JACK': 'jack_model',
    'DUMBBELL': 'DUMBBELL'
}

def label(df: pd.DataFrame,
          num_picos: int,
          exercise: str) -> pd.DataFrame:
    """
    Label exercise intervals based on peak occurrences in the provided DataFrame.

    Parameters:
    - df: (pd.DataFrame)
        DataFrame containing workout information with 'peaks' column.
    - num_picos: (int)
        Number of peaks to consider in each exercise interval.
    - exercise: (str)
        Type of exercise, should be one of 'SQUAT', 'PUSHUP', 'BURPEE', 'PULLUP', 'SITUP',
      'SQUAT JUMP', 'JUMPING JACK', or 'DUMBBELL'.

    Returns:
    - pd.DataFrame: DataFrame with exercise predictions, start, and end points for each interval.

    """
    if 'peaks' not in df.columns or not pd.api.types.is_numeric_dtype(df['peaks']):
        raise ValueError("La columna 'peaks' no está presente o no es de tipo numérico.")

    # Encontrar los índices donde hay picos
    indices_picos = df.index[df['peaks'].notna()].tolist()

    # Verificar que haya suficientes picos para fraccionar
    if len(indices_picos) < num_picos:
        raise ValueError("No hay suficientes picos para fraccionar según la cantidad especificada.")

    # Crear una lista de diccionarios para almacenar las predicciones, el inicio y fin de cada ventana
    resultados = []

    # Cargar el modelo
    model_name = exercise_models.get(exercise)
    
    if model_name is None:
        raise ValueError("Por favor, revisa el tipo de ejercicio. El ejercicio debe ser: SQUAT, PUSHUP, BURPEE, PULLUP, SITUP, SQUAT JUMP, JUMPING JACK o DUMBBELL")
    
    pipeline = load_model(model_name=model_name)

    # Iterar a través de los picos
    for i in range(len(indices_picos) - num_picos + 1):
        for j in range(i + 1, i + num_picos + 1):
            warnings.filterwarnings('ignore')
            # Fraccionar el DataFrame
            inicio_ventana = indices_picos[i]
            fin_ventana = indices_picos[j] if j < len(indices_picos) else None
            df_fraccionado = df.loc[inicio_ventana:fin_ventana]

            # Ingresar datos al modelo y predecir
            data = create_training_data_stats(df_fraccionado, exercise)
            data_custom = pd.DataFrame([data])
            prediction = predict_model(pipeline, data_custom, raw_score=True)
            y_pred = prediction.prediction_label.iloc[0]

            # Almacenar la predicción en la lista como un diccionario
            resultados.append({'Prediccion': y_pred, 'Inicio_Ventana': inicio_ventana, 'Fin_Ventana': fin_ventana})

    # Convertir la lista de diccionarios en un DataFrame
    df_resultados = pd.DataFrame(resultados)

    return df_resultados






















