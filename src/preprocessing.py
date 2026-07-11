def clean_data(df):
    """
    Fill missing values in Income column.
    """
    df["Income"] = df["Income"].fillna(df["Income"].median())
    return df