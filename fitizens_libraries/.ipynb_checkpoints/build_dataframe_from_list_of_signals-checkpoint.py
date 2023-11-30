import pandas as pd
def build_dataframe(data):
    series = map(lambda x: x["series"].to_dict(), data)
    df = pd.DataFrame(list(series)).applymap(pd.Series)
    return df.reset_index(drop=True)