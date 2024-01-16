import pandas as pd
def obtener_estadisticas(df):
    features = ['mean_accX', 'mean_accZ', 'mean_linAccZ', 'mean_accZ_mod',
            'mean_gyroX_mod', 'mean_linAccX_mod', 'std_accZ', 'std_gyroX', 'std_gyroZ',
            'std_magnX', 'std_linAccX', 'std_accZ_mod', 'std_linAccX_mod',
            'median_accX', 'median_accZ', 'median_linAccZ', 'median_accZ_mod',
            'median_gyroX_mod', 'median_linAccX_mod']
    promedio = df.mean().add_prefix('mean_')
    std = df.std().add_prefix('std_')
    median = df.median().add_prefix('median_')
    nuevo_df_estad = pd.DataFrame().append(pd.concat([promedio, std, median]), ignore_index=True)

    X1 = nuevo_df_estad[features].copy()

    return X1