import matplotlib.pyplot as plt

def plot_income_distribution(df):
    """
    Plot the distribution of customer income.
    """
    plt.figure(figsize=(8,5))
    plt.hist(df["Income"], bins=30)
    plt.title("Income Distribution")
    plt.xlabel("Income")
    plt.ylabel("Number of Customers")

    plt.savefig("outputs/graphs/income_distribution.png")
    plt.close()