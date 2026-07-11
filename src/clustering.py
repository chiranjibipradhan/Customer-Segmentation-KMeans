from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def elbow_method(df):

    X = df[[
        "Income",
        "Recency",
        "MntWines",
        "MntMeatProducts",
        "MntFishProducts",
        "NumWebPurchases",
        "NumStorePurchases"
    ]]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    inertia = []

    for i in range(1, 11):
        kmeans = KMeans(
            n_clusters=i,
            random_state=42,
            n_init=10
        )

        kmeans.fit(X)
        inertia.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 11), inertia, marker="o")
    plt.title("Elbow Method")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")

    plt.savefig("outputs/graphs/elbow_method.png")
    plt.close()


def perform_clustering(df):

    X = df[[
        "Income",
        "Recency",
        "MntWines",
        "MntMeatProducts",
        "MntFishProducts",
        "NumWebPurchases",
        "NumStorePurchases"
    ]]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    kmeans = KMeans(
        n_clusters=5,
        random_state=42,
        n_init=10
    )

    df["Cluster"] = kmeans.fit_predict(X)

    plt.figure(figsize=(8, 6))

    plt.scatter(
        df["Income"],
        df["MntWines"],
        c=df["Cluster"]
    )

    plt.xlabel("Income")
    plt.ylabel("Wine Spending")
    plt.title("Customer Segments using K-Means")

    plt.savefig("outputs/graphs/customer_segments.png")
    plt.close()

    df.to_csv("outputs/customer_segments.csv", index=False)

    print("\nCluster Counts:")
    print(df["Cluster"].value_counts())

    print("\nBusiness Insights:")
    print(
        df.groupby("Cluster")[["Income", "MntWines"]]
        .mean()
        .round(2)
    )

    return df