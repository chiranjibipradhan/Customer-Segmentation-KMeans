import streamlit as st
import pandas as pd
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

if page == "🏠 Overview":

     st.title("📊 Customer Segmentation Dashboard")
     st.markdown("### Machine Learning using K-Means Clustering")
     st.markdown("---")

# -----------------------------
# Run Button
# -----------------------------
if st.button("🚀 Run Analysis"):

    # Load and clean data
    df = load_data()
    df = clean_data(df)

    # Generate clustering
    elbow_method(df)
    df = perform_clustering(df)


        # -----------------------------
    # KPI Cards
    # -----------------------------
    st.markdown("## 📈 Dashboard Summary")

    col1, col2, col3 = st.columns(3)

    col1.metric("👥 Total Customers", len(df))
    col2.metric("🎯 Number of Clusters", df["Cluster"].nunique())
    col3.metric("💰 Average Income", f"${df['Income'].mean():,.0f}")

    st.markdown("---")

    # -----------------------------
    # Interactive Charts
    # -----------------------------
    st.subheader("📊 Interactive Visualizations")

    chart1, chart2 = st.columns(2)

    with chart1:
        fig1 = px.histogram(
            df,
            x="Income",
            nbins=30,
            title="Income Distribution"
        )
        st.plotly_chart(fig1, use_container_width=True)

    with chart2:
        fig2 = px.scatter(
            df,
            x="Income",
            y="MntWines",
            color=df["Cluster"].astype(str),
            title="Customer Segments"
        )
        st.plotly_chart(fig2, use_container_width=True)

    chart3, chart4 = st.columns(2)

    with chart3:
        cluster_counts = df["Cluster"].value_counts().reset_index()
        cluster_counts.columns = ["Cluster", "Count"]

        fig3 = px.bar(
            cluster_counts,
            x="Cluster",
            y="Count",
            title="Customers in Each Cluster"
        )
        st.plotly_chart(fig3, use_container_width=True)

    with chart4:
        fig4 = px.box(
            df,
            x="Cluster",
            y="Income",
            color=df["Cluster"].astype(str),
            title="Income by Cluster"
        )
        st.plotly_chart(fig4, use_container_width=True)


            # -----------------------------
    # Business Insights
    # -----------------------------
    st.markdown("---")
    st.subheader("💡 Business Insights")

    avg_income = df.groupby("Cluster")["Income"].mean().round(2)

    for cluster, income in avg_income.items():
        st.info(
            f"Cluster {cluster}: Average Income = ${income:,.2f}"
        )

    # -----------------------------
    # Dataset Preview
    # -----------------------------
    st.markdown("---")
    st.subheader("📋 Dataset Preview")

    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("📋 Clustered Customers")

    st.dataframe(df, use_container_width=True)

    # -----------------------------
    # Download Button
    # -----------------------------
    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇ Download Clustered Dataset",
        data=csv,
        file_name="customer_segments.csv",
        mime="text/csv"
    )

    st.markdown("---")
    st.caption("Developed by Chiranjibi Pradhan | IoT Academy Capstone Project")