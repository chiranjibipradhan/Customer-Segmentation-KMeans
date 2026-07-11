import matplotlib.pyplot as plt


def plot_cluster_counts(df):
    """
    Plot number of customers in each cluster.
    """

    plt.figure(figsize=(8,5))

    df["Cluster"].value_counts().sort_index().plot(
        kind="bar"
    )

    plt.title("Customers in Each Cluster")
    plt.xlabel("Cluster")
    plt.ylabel("Number of Customers")

    plt.savefig("outputs/graphs/cluster_counts.png")
    plt.close()