import streamlit as st
import plotly.express as px

from src.data_loader import load_data
from src.preprocessing import clean_data
from src.clustering import elbow_method, perform_clustering

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("📊 Dashboard")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Overview",
        "📂 Dataset",
        "📈 EDA",
        "🤖 Clustering",
        "💡 Business Insights",
        "⬇ Download",
        "👨 About"
    ]
)

st.sidebar.success("✅ Dashboard Ready")

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_customer_data():
    df = load_data()
    df = clean_data(df)
    elbow_method(df)
    df = perform_clustering(df)
    return df

df = load_customer_data()
# ==========================
# OVERVIEW PAGE
# ==========================

if page == "🏠 Overview":

    st.title("📊 Customer Segmentation Dashboard")
    st.markdown("### Machine Learning using K-Means Clustering")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    col1.metric("👥 Total Customers", len(df))
    col2.metric("🎯 Number of Clusters", df["Cluster"].nunique())
    col3.metric("💰 Average Income", f"${df['Income'].mean():,.0f}")

    st.markdown("---")

    st.subheader("📖 Project Overview")

    st.info("""
This project uses the **K-Means Clustering Algorithm**
to divide customers into different groups based on their purchasing behaviour.

The dashboard helps businesses:

- Understand customer segments
- Identify high-value customers
- Improve marketing strategies
- Increase customer retention
""")
    # ==========================
# DATASET PAGE
# ==========================

elif page == "📂 Dataset":

    st.title("📂 Dataset")

    st.markdown("### Customer Dataset Preview")

    st.dataframe(df.head(20), use_container_width=True)

    st.markdown("---")

    st.subheader("📊 Dataset Information")

    col1, col2 = st.columns(2)

    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])

    st.write("### Column Names")

    st.write(list(df.columns))

    st.markdown("---")

    with st.expander("Show Complete Dataset"):
        st.dataframe(df, use_container_width=True)

        # ---------------- Clustering ----------------
if page == "🧩 Clustering":

    st.title("🧩 Customer Clustering")

    st.markdown("### Customer Segments")

    fig = px.scatter(
        df,
        x="Income",
        y="MntWines",
        color=df["Cluster"].astype(str),
        title="Customer Segments"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Customers in Each Cluster")

    cluster_counts = df["Cluster"].value_counts().reset_index()
    cluster_counts.columns = ["Cluster", "Customers"]

    fig2 = px.bar(
        cluster_counts,
        x="Cluster",
        y="Customers",
        color="Cluster",
        title="Customers per Cluster"
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Cluster Statistics")

    st.dataframe(
        df.groupby("Cluster")[["Income", "MntWines"]].mean().round(2),
        use_container_width=True
    )


        # ---------------- EDA ----------------
if page == "📈 EDA":

    st.title("📈 Exploratory Data Analysis")

    st.subheader("Income Distribution")

    fig = px.histogram(
        df,
        x="Income",
        nbins=30,
        title="Income Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Income vs Wine Spending")

    fig2 = px.scatter(
        df,
        x="Income",
        y="MntWines",
        color=df["Cluster"].astype(str),
        title="Income vs Wine Spending"
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Income by Cluster")

    fig3 = px.box(
        df,
        x="Cluster",
        y="Income",
        color=df["Cluster"].astype(str),
        title="Income by Cluster"
    )
    st.plotly_chart(fig3, use_container_width=True)


    # ---------------- Clustering ----------------
if page == "🤖 Clustering":

    st.title("🤖 Customer Clustering")

    st.markdown("### Customer Segments")

    fig = px.scatter(
        df,
        x="Income",
        y="MntWines",
        color=df["Cluster"].astype(str),
        title="Customer Segments"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Customers in Each Cluster")

    cluster_counts = df["Cluster"].value_counts().reset_index()
    cluster_counts.columns = ["Cluster", "Customers"]

    fig2 = px.bar(
        cluster_counts,
        x="Cluster",
        y="Customers",
        color="Cluster",
        title="Customers per Cluster"
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Cluster Statistics")

    st.dataframe(
        df.groupby("Cluster")[["Income", "MntWines"]].mean().round(2),
        use_container_width=True
    )


    # ---------------- Business Insights ----------------
if page == "💡 Business Insights":

    st.title("💡 Business Insights")

    st.markdown("### Average Income by Cluster")

    avg_income = df.groupby("Cluster")["Income"].mean().round(2)

    for cluster, income in avg_income.items():
        st.info(f"Cluster {cluster}: Average Income = ${income:,.2f}")

    st.markdown("---")

    st.subheader("📊 Cluster Summary")

    summary = (
        df.groupby("Cluster")
        .agg(
            Customers=("Cluster", "count"),
            Average_Income=("Income", "mean"),
            Average_Wine_Spending=("MntWines", "mean")
        )
        .round(2)
    )

    st.dataframe(summary, use_container_width=True)

    st.markdown("---")

    st.success("""
### Key Business Recommendations

- 🎯 Target high-income customers with premium products.
- 🛍️ Offer discounts to low-spending customers.
- 📧 Create personalized marketing campaigns.
- 💰 Reward loyal customers with exclusive offers.
- 📈 Use customer segmentation to improve sales and retention.
""")
    
    # ---------------- Download ----------------
if page == "⬇ Download":

    st.title("⬇ Download Results")

    st.write("Download the clustered customer dataset.")

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇ Download Clustered Dataset",
        data=csv,
        file_name="customer_segments.csv",
        mime="text/csv"
    )

    st.success("The downloaded file contains all customers with their assigned cluster.")


    # ---------------- About ----------------
if page == "👨 About":

    st.title("👨 About This Project")

    st.markdown("""
# 📊 Customer Segmentation using K-Means Clustering

This project uses the **K-Means Clustering Algorithm**
to divide customers into different groups based on
their purchasing behavior.

---

## 🎯 Project Objective

- Understand customer behavior
- Identify customer segments
- Improve marketing strategies
- Increase business revenue

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Plotly
- Streamlit

---

## 🤖 Machine Learning Algorithm

- K-Means Clustering

---

## 📂 Dataset

Marketing Campaign Dataset

- Total Customers: 2240
- Features: Customer demographics and purchasing behavior

---

## 👨‍💻 Developer

**Chiranjibi Pradhan**

IoT Academy Capstone Project

---

## 🚀 Future Improvements

- Customer Recommendation System
- Predictive Analytics
- Real-time Dashboard
- Advanced Machine Learning Models

""")

    st.success("Thank you for exploring this Customer Segmentation Dashboard!")





