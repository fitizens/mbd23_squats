from custom_libraries.obtener_estadisticas import obtener_estadisticas
from custom_libraries.create_training_data_stats import create_training_data_stats
from custom_libraries.create_training_data_stats import create_custom_dataframe
import pandas as pd
import joblib 
import warnings
from pycaret.classification import *

def label(df, num_picos:int, exercise:str):
    # Asegúrate de que la columna "peaks" esté presente y sea de tipo numérico
    if 'peaks' not in df.columns or not pd.api.types.is_numeric_dtype(df['peaks']):
        raise ValueError("La columna 'peaks' no está presente o no es de tipo numérico.")

    # Encuentra los índices donde hay picos
    indices_picos = df.index[df['peaks'].notna()].tolist()

    # Verifica que haya suficientes picos para fraccionar
    if len(indices_picos) < num_picos:
        raise ValueError("No hay suficientes picos para fraccionar según la cantidad especificada.")

    # Crear una lista de diccionarios para almacenar las predicciones, el inicio y fin de cada ventana
    resultados = []

    # Iterar a través de los picos
    for i in range(len(indices_picos) - num_picos + 1):
        for j in range(i + 1, i + num_picos + 1):
            warnings.filterwarnings('ignore')
            # Fraccionar el DataFrame
            inicio_ventana = indices_picos[i]
            fin_ventana = indices_picos[j] if j < len(indices_picos) else None
            df_fraccionado = df.loc[inicio_ventana:fin_ventana]

            # Ingresar datos al modelo y predecir
            if exercise == 'SQUAT':
                modelo = joblib.load('modeloXGB.pkl')
                X_estad = obtener_estadisticas(df_fraccionado)
                y_pred = modelo.predict(X_estad)
                
            elif exercise == 'PUSHUP':
                modelo = joblib.load('logistic_model.pkl')
                scaler = joblib.load('scaler_pushup.pkl')
                data_info = create_training_data_stats(df_fraccionado,exercise)
                data_custom = pd.DataFrame([data_info])
                features = ['linAccZ_mean', 'linAccZ_std', 'linAccZ_median', 'linAccZ_min',
                            'linAccZ_max', 'linAccZ_range', 'linAccZ_quartile_25',
                            'linAccZ_quartile_75', 'linAccZ_iqr']
                X1 = data_custom[features].copy()
                X_escal = scaler.transform(X1)
                y_pred = modelo.predict(X_escal)
                
            elif exercise =='BURPEE':
                pipeline = load_model(model_name="BURPEE")
                data = create_training_data_stats(df_fraccionado,exercise)
                data_custom = pd.DataFrame([data])
                prediction = predict_model(pipeline, data_custom, raw_score=True)
                y_pred = prediction.prediction_label.iloc[0]
                
            elif exercise == 'PULLUP':
                modelo = joblib.load('pullup_model.pkl')
                data_info = create_training_data_stats(df_fraccionado,exercise)
                data_custom = pd.DataFrame([data_info])
                features = ['linAccZ_mean', 'linAccZ_std', 'linAccZ_median',
                            'linAccZ_kurtosis', 'linAccZ_quartile_25', 'linAccZ_iqr',
                            'gyroX_quartile_25', 'gyroX_quartile_75', 'gyroX_iqr']
                X1 = data_custom[features].copy()
                y_pred = modelo.predict(X1)
                
            elif exercise == 'SITUP':
                modelo = joblib.load('situp_model.pkl')
                scaler = joblib.load('scaler_situp.pkl')
                data_info = create_training_data_stats(df_fraccionado,exercise)
                data_custom = pd.DataFrame([data_info])
                features = ['linAccY_mean', 'linAccY_median', 'linAccY_min',
                            'linAccY_quartile_25', 'linAccZ_quartile_75', 'gyroX_median',
                            'magnX_skewness', 'magnY_skewness', 'magnZ_median']
                X1 = data_custom[features].copy()
                X_escal = scaler.transform(X1)
                y_pred = modelo.predict(X_escal)
                
            elif exercise == 'SQUAT JUMP':
                modelo = joblib.load('jump_model.pkl')
                scaler = joblib.load('scaler_jump.pkl')
                data_info = create_training_data_stats(df_fraccionado,exercise)
                data_custom = pd.DataFrame([data_info])
                features = ['linAccZ_std', 'linAccZ_min', 'linAccZ_quartile_25',
                            'linAccZ_quartile_75', 'linAccZ_iqr', 'gyroX_std', 'gyroX_min',
                            'gyroX_quartile_25', 'gyroX_iqr']
                X1 = data_custom[features].copy()
                X_escal = scaler.transform(X1)
                y_pred = modelo.predict(X_escal)
                
            elif exercise =='JUMPING JACK':
                pipeline = load_model(model_name="jack_model")
                data = create_training_data_stats(df_fraccionado,exercise)
                data_custom = pd.DataFrame([data])
                prediction = predict_model(pipeline, data_custom, raw_score=True)
                y_pred = prediction.prediction_label.iloc[0]
                
            elif exercise =='DUMBBELL':
                pipeline = load_model(model_name="DUMBBELL")
                data = create_training_data_stats(df_fraccionado,exercise)
                data_custom = pd.DataFrame([data])
                prediction = predict_model(pipeline, data_custom, raw_score=True)
                y_pred = prediction.prediction_label.iloc[0]
                
            else:
                raise ValueError("Please review the written exercise. Exercise must be: SQUAT, PUSHUP, BURPEE, PULLUP,                                         SITUP, SQUAT JUMP, JUMPING JACK or DUMBBELL")


            # Almacenar la predicción en la lista como un diccionario
            resultados.append({'Prediccion': y_pred, 'Inicio_Ventana': inicio_ventana, 'Fin_Ventana': fin_ventana})

    # Convertir la lista de diccionarios en un DataFrame
    df_resultados = pd.DataFrame(resultados)

    return df_resultados






