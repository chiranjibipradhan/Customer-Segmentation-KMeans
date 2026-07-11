import pandas as pd

def load_data():
    """
    Load the customer dataset.
    """
    df = pd.read_csv("data/marketing_campaign.csv", sep="\t")
    return df