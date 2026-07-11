from src.data_loader import load_data
from src.preprocessing import clean_data
from src.eda import plot_income_distribution
from src.clustering import elbow_method
from src.clustering import perform_clustering
from src.visualization import plot_cluster_counts

# Load dataset
df = load_data()

# Clean dataset
df = clean_data(df)

print("Dataset Loaded Successfully!")
print(df.head())

print("\nShape:", df.shape)

plot_income_distribution(df)

print("\nIncome distribution graph saved successfully!")

elbow_method(df)

print("Elbow Method graph saved successfully!")

df = perform_clustering(df)

print("Customer Segmentation completed successfully!")

plot_cluster_counts(df)

print("Cluster count graph saved successfully!")