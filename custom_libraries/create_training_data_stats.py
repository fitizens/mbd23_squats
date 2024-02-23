import numpy as np
def create_custom_dataframe(series):
    """
      Create a custom DataFrame containing specific columns from the input series.

      Parameters:
      - series: (pd.Series)
        Input series containing data.

      Returns:
      - pd.DataFrame: Custom DataFrame containing selected columns.
      """
    df =  series[["linAccX", "linAccY", "linAccZ", "gyroX", "gyroY", "gyroZ", "magnX", "magnY", "magnZ"]]
    return df

def create_training_data_stats(df, exercise):
    """
        Create statistical features from the input DataFrame based on the specified exercise type.

        Parameters:
        - df: (pd.DataFrame)
            DataFrame containing workout information.
        - exercise: (str)
            Type of exercise for which statistical features are generated.

        Returns:
        - dict:
            Dictionary containing statistical features for each selected column.

        """
    stats_dict = {}
    series = create_custom_dataframe(df)

    for column in series.columns:
        
        if exercise == 'SQUAT':
            mean = series[column].mean()
            std = series[column].std()
            median = series[column].median()

            stats_dict[f"{column}_mean"] = mean
            stats_dict[f"{column}_std"] = std
            stats_dict[f"{column}_median"] = median

        if exercise == 'PUSHUP' or exercise == 'PULLUP' or exercise == 'SITUP' or exercise == 'SQUAT JUMP':
            mean = series[column].mean()
            std = series[column].std()
            median = series[column].median()
            skewness = series[column].skew()
            kurtosis = series[column].kurtosis()
            min_val = series[column].min()
            max_val = series[column].max()
            range_val = max_val - min_val
            quartile_25 = np.percentile(series[column], 25)
            quartile_75 = np.percentile(series[column], 75)
            iqr = quartile_75 - quartile_25

            stats_dict[f"{column}_mean"] = mean
            stats_dict[f"{column}_std"] = std
            stats_dict[f"{column}_median"] = median
            stats_dict[f"{column}_skewness"] = skewness
            stats_dict[f"{column}_kurtosis"] = kurtosis
            stats_dict[f"{column}_min"] = min_val
            stats_dict[f"{column}_max"] = max_val
            stats_dict[f"{column}_range"] = range_val
            stats_dict[f"{column}_quartile_25"] = quartile_25
            stats_dict[f"{column}_quartile_75"] = quartile_75
            stats_dict[f"{column}_iqr"] = iqr

        elif exercise == 'BURPEE':
            mean = series[column].mean()
            std = series[column].std()
            median = series[column].median()
            quartile_25 = np.percentile(series[column], 25)
            quartile_75 = np.percentile(series[column], 75)
            iqr = quartile_75 - quartile_25
            
            stats_dict[f"{column}_mean"] = mean
            stats_dict[f"{column}_std"] = std
            stats_dict[f"{column}_median"] = median
            max_val = series[column].max()
            stats_dict[f"{column}_quartile_25"] = quartile_25
            stats_dict[f"{column}_quartile_75"] = quartile_75
            stats_dict[f"{column}_iqr"] = iqr
            stats_dict[f"{column}_max"] = max_val
        
        elif exercise == 'JUMPING JACK':
            mean = series[column].mean()
            std = series[column].std()
            median = series[column].median()            
            skewness = series[column].skew()
            kurtosis = series[column].kurtosis()
            variance = series[column].var()  # Calculate variance
            rms = np.sqrt(np.mean(series[column]**2))
            
            stats_dict[f"{column}_mean"] = mean
            stats_dict[f"{column}_std"] = std
            stats_dict[f"{column}_median"] = median           
            stats_dict[f"{column}_median"] = median
            stats_dict[f"{column}_skewness"] = skewness
            stats_dict[f"{column}_kurtosis"] = kurtosis
            stats_dict[f"{column}_variance"] = variance  
            stats_dict[f"{column}_rms"] = rms
            
        elif exercise == 'DUMBBELL':
            mean = series[column].mean()
            std = series[column].std()
            median = series[column].median()
            skewness = series[column].skew()
            kurtosis = series[column].kurtosis()
            
            stats_dict[f"{column}_mean"] = mean
            stats_dict[f"{column}_std"] = std
            stats_dict[f"{column}_median"] = median           
            stats_dict[f"{column}_skewness"] = skewness
            stats_dict[f"{column}_kurtosis"] = kurtosis


    return stats_dict















