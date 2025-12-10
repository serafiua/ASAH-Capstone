import streamlit as st
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from re import M
from sklearn.preprocessing import RobustScaler, PowerTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from scipy.stats import mode
from streamlit_extras.stylable_container import stylable_container
import styles

st.set_page_config(
    page_title="Customer Segmentation Dashboard - A25-CS281", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(styles.global_css, unsafe_allow_html=True)

#########################################
# 1. DATA LOADING AND PREPROCESSING
#########################################
@st.cache_data
def load_and_preprocess_data():
    df = pd.read_parquet("transaction_data.parquet") 
    df_eda = df.copy()
    df = df[df['Country'] == 'United Kingdom']
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    neg_cols = ['UserId', 'ItemCode', 'NumberOfItemsPurchased', 'CostPerItem']
    df = df[(df[neg_cols] > 0).all(axis=1)]
    df['TransactionTime'] = pd.to_datetime(df['TransactionTime'], format='%a %b %d %H:%M:%S IST %Y')
    cat_cols = ['UserId', 'TransactionId', 'ItemCode', 'ItemDescription', 'Country']
    for col in cat_cols:
        df[col] = df[col].astype('category')
    df = df.astype({'NumberOfItemsPurchased': 'int64', 'CostPerItem': 'float64'})
    mask_2028 = df['TransactionTime'].dt.year == 2028
    df.loc[mask_2028, 'TransactionTime'] = df.loc[mask_2028, 'TransactionTime'].apply(lambda x: x.replace(year=2020))
    columns = ['CostPerItem', 'NumberOfItemsPurchased']
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = np.where(df[col] < lower, lower, df[col])
        df[col] = np.where(df[col] > upper, upper, df[col])
    df_clustering = df.copy()
    df_clustering['TotalPrice'] = df_clustering['NumberOfItemsPurchased'] * df_clustering['CostPerItem']
    return df, df_eda, df_clustering

df, df_eda, df_clustering = load_and_preprocess_data()

#########################################
# 2. FORMAT AND CARDS
#########################################
@st.cache_data
def format_currency(value):
    return f"${value:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")

@st.cache_data
def format_metric_text(row, metric):
    if pd.isna(row):
        return "N/A"
    if metric == "Recency":
        return f"{row:.0f} hari"
    elif metric == "Frequency":
        return f"{row:.0f} kali"
    elif metric == "Monetary":
        return f"${row:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")
    else:
        return str(row)

@st.cache_data
def metric_card(title, value, delta=None, bg='#313234', text_color="white"):
    st.markdown(
        f"""
        <div style="
            padding: 18px;
            border-radius: 18px;
            background: {bg};
            color: {text_color};
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            text-align: left;
            margin: 8px 0px;
        ">
            <div style="font-size: 14px; opacity: 0.75; margin-bottom: 6px;">{title}</div>
            <div style="font-size: 26px; font-weight: 700; margin-bottom: 4px;">{value}</div>
            {"<div style='font-size: 14px; color: #16c784;'>"+delta+"</div>" if delta else ""}
        </div>
        """,
        unsafe_allow_html=True
    )

#########################################
# 3. RFM 
#########################################

# 3.1. hitung rfm
@st.cache_data
def calculate_rfm(df_clustering):
    reference_date = df_clustering['TransactionTime'].max() + pd.Timedelta(days=1)
    rfm = df_clustering.groupby('UserId').agg({
        'TransactionTime': lambda x: (reference_date - x.max()).days, 
        'TransactionId': 'nunique',
        'TotalPrice': 'sum'
    }).reset_index()
    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
    rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1])
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
    rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1, 2, 3, 4, 5])
    rfm['RFM_Group'] = (
        rfm['R_Score'].astype(str) +
        rfm['F_Score'].astype(str) +
        rfm['M_Score'].astype(str)
    )
    rfm['RFM_Score'] = (
        rfm[['R_Score', 'F_Score', 'M_Score']].sum(axis=1)
    ).astype(int)

    return rfm

# rfm = calculate_rfm(df_clustering) # pindahin ke page

# 3.2. rfm table per cust (page RFM, tab 1)
@st.cache_data
def rfm_customer_table(rfm, rfm_kmeans, segment_labels):
    rfm_selected = rfm[[
        "CustomerID", 
        "Recency", 
        "Frequency", 
        "Monetary"
    ]]

    rfm_with_cluster = pd.merge(rfm_selected, rfm_kmeans[['CustomerID', 'Cluster']], on='CustomerID', how='left')
    rfm_with_cluster['Segment'] = rfm_with_cluster['Cluster'].map(segment_labels)
    rfm_with_cluster.drop(columns=['Cluster'], inplace=True)
    rfm_with_cluster = rfm_with_cluster[['Segment', 'CustomerID', 'Recency', 'Frequency', 'Monetary']]
    rfm_with_cluster = rfm_with_cluster.sort_values(by="Segment", ascending=False)
    rfm_with_cluster["Monetary"] = rfm_with_cluster["Monetary"].apply(lambda x: f"${x:,.2f}")

    return rfm_with_cluster

#########################################
# 3. FINAL RFM PREPROCESSING
#########################################

# power transformer dan scaling
@st.cache_data
def prepare_rfm_for_clustering(rfm):
    rfm_pt = rfm.copy()
    pt = PowerTransformer(method='yeo-johnson')
    rfm_pt[['Recency', 'Frequency', 'Monetary']] = pt.fit_transform(
        rfm_pt[['Recency', 'Frequency', 'Monetary']]
    )
    scaler = RobustScaler()
    rfm_scaled = scaler.fit_transform(
        rfm_pt[['Recency', 'Frequency', 'Monetary']]
    )
    rfm_scaled = pd.DataFrame(
        rfm_scaled,
        columns=['Recency_Scaled', 'Frequency_Scaled', 'Monetary_Scaled']
    )
    rfm_final_2 = pd.concat(
        [rfm_pt[['CustomerID']].reset_index(drop=True), rfm_scaled],
        axis=1
    )

    return rfm_final_2, pt, scaler

# rfm_final_2, pt, scaler = prepare_rfm_for_clustering(rfm) # pindahin ke page

#########################################
# 4. KMEANS
#########################################

@st.cache_data
def run_kmeans(rfm_final_2, optimal_k=4):
    X = rfm_final_2[['Recency_Scaled', 'Frequency_Scaled', 'Monetary_Scaled']]

    wcss = []
    for i in range(1, 11):
        kmeans_tmp = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
        kmeans_tmp.fit(X)
        wcss.append(kmeans_tmp.inertia_)

    kmeans_model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    labels = kmeans_model.fit_predict(X)

    rfm_kmeans = rfm_final_2.copy()
    rfm_kmeans['Cluster'] = labels

    cluster_counts = (
        rfm_kmeans.groupby('Cluster')
        .size()
        .reset_index(name='Count')
        .sort_values('Cluster')
    )

    segment_labels = {
        0: "Low-Value",
        1: "At-risk",
        2: "Loyalist",
        3: "Top Customers",
    }

    sil = silhouette_score(X, labels)
    cal = calinski_harabasz_score(X, labels)
    db = davies_bouldin_score(X, labels)

    metrics = pd.DataFrame({
        "Metric": ["Silhouette Score", "Calinski-Harabasz", "Davies-Bouldin"],
        "Value": [sil, cal, db]
    })

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X)
    rfm_kmeans['PCA1'] = pca_result[:, 0]
    rfm_kmeans['PCA2'] = pca_result[:, 1]

    centroids = kmeans_model.cluster_centers_
    centroids_df = pd.DataFrame(centroids, columns=X.columns)
    centroids_pca = pca.transform(centroids_df)

    return rfm_kmeans, wcss, metrics, kmeans_model, pca, centroids_pca, cluster_counts, segment_labels
# (rfm_kmeans, wcss, kmeans_metrics, kmeans_model, pca, centroids_pca, cluster_counts, segment_labels) = run_kmeans(rfm_final_2) # pindahin ke page


#########################################
# 5. GMM
#########################################

@st.cache_resource
def run_gmm_clustering(rfm_final_2):

    X = rfm_final_2[['Recency_Scaled', 'Frequency_Scaled','Monetary_Scaled']]

    gmm_final = GaussianMixture(
        n_components=4,
        covariance_type='full',
        random_state=42,
        n_init=10
    )

    labels = gmm_final.fit_predict(X)

    rfm_gmm_final = rfm_final_2.copy()
    rfm_gmm_final['Cluster'] = labels

    sil_score = silhouette_score(X, labels)
    calinski = calinski_harabasz_score(X, labels)
    db_index = davies_bouldin_score(X, labels)

    aic = gmm_final.aic(X)
    bic = gmm_final.bic(X)

    results = pd.DataFrame({
        "Metric": ["Silhouette Score", "Calinski-Harabasz", "Davies-Bouldin", "AIC", "BIC"],
        "Value": [sil_score, calinski, db_index, aic, bic]
    })

    return rfm_gmm_final, results, gmm_final

# rfm_gmm_final, gmm_metrics, model_gmm = run_gmm_clustering(rfm_final_2) # pindahin ke page

#########################################
# 6. INVERSE RFM
#########################################

@st.cache_data
def inverse_rfm(rfm_kmeans, _scaler, _pt):
    # 1. Inverse robust scaler
    inverse_scaled = _scaler.inverse_transform(
        rfm_kmeans[['Recency_Scaled', 'Frequency_Scaled', 'Monetary_Scaled']]
    )
    inverse_scaled = pd.DataFrame(
        inverse_scaled,
        columns=['Recency', 'Frequency', 'Monetary']
    )

    # 2. Inverse PowerTransformer
    inverse_original = _pt.inverse_transform(inverse_scaled)
    inverse_original = pd.DataFrame(
        inverse_original,
        columns=['Recency', 'Frequency', 'Monetary']
    )

    # 3. MERGE dengan ID + Cluster
    rfm_inverse = pd.concat([
        rfm_kmeans[['CustomerID', 'Cluster']].reset_index(drop=True),
        inverse_original.reset_index(drop=True)
    ], axis=1)

    return rfm_inverse

# rfm_kmeans_inverse = inverse_rfm(rfm_kmeans, scaler, pt) # pindahin ke page

#########################################
# 7. MARKET BASKET ANALYSIS (MBA) GLOBAL 
#########################################
# 7.1. Data preparation
@st.cache_data
def data_mba(df):
    df_mba = df.copy()
    df_mba['TransactionId'] = df_mba['TransactionId'].astype(str)
    df_mba['ItemDescription'] = df_mba['ItemDescription'].astype(str).str.strip()
    df_mba = df_mba[df_mba['ItemDescription'] != '']
    return df_mba

# df_mba = data_mba(df) # pindahin ke page
 
# 7.2. Build Basket
@st.cache_data
def build_basket(df_mba):
    basket = (
        df_mba
        .groupby(['TransactionId', 'ItemDescription'])['NumberOfItemsPurchased']
        .sum()
        .unstack()
        .fillna(0)
    )
    basket = (basket > 0).astype(int)
    return basket

# basket = build_basket(df_mba) # pindahin ke page

# 7.3. Run Apriori MBA
@st.cache_data
def run_mba(basket):
    frequent = apriori(
        basket,
        min_support=0.02,   
        use_colnames=True
    )

    rules = association_rules(
        frequent,
        metric='lift',
        min_threshold=1    
    )

    rules = rules[
        (rules['lift'] > 1.2) &
        (rules['confidence'] > 0.3)
    ]

    rules = rules.sort_values('lift', ascending=False).reset_index(drop=True)
    return rules

# rules = run_mba(basket) # pindahin ke page

# # 7.4. Build Combo Mapping (dedupe A↔B)

# # Extract single-item antecedent & consequent
# rules['ante'] = rules['antecedents'].apply(lambda x: list(x)[0])
# rules['cons'] = rules['consequents'].apply(lambda x: list(x)[0])

# # Normalize pair (alphabetical order) → ensures A,B == B,A
# def normalize_pair(row):
#     a, b = row['ante'], row['cons']
#     return tuple(sorted([a, b]))

# rules['pair'] = rules.apply(normalize_pair, axis=1)

# # Keep only the strongest rule (highest lift) for each pair
# rules_unique = (
#     rules
#     .sort_values("lift", ascending=False)
#     .drop_duplicates(subset="pair", keep="first")
#     .reset_index(drop=True)
# )

# # Build combo text after dedupe
# rules_unique['combo'] = rules_unique.apply(
#     lambda x: f"{x['ante']} → {x['cons']}", axis=1
# )

# # Sidebar combo list
# combo_list = rules_unique['combo'].tolist()

# # Dictionary for quick lookup
# combo_dict = {
#     row['combo']: {
#         "ante": row['ante'],
#         "cons": row['cons'],
#         "support": row['support'],
#         "confidence": row['confidence'],
#         "lift": row['lift']
#     }
#     for _, row in rules_unique.iterrows()
# }

#########################################
# 8. MARKET BASKER ANALYSIS (MBA) PER CLUSTER
#########################################
# 8.1. Data preparation
@st.cache_data
def prepare_mba_cluster(df_mba, rfm_kmeans):
    rfm_kmeans_mba = rfm_kmeans.rename(columns={'CustomerID':'UserId'})
    df_mba_cluster = df_mba.merge(
        rfm_kmeans_mba[['UserId', 'Cluster']],
        on='UserId',
        how='left'
    )
    return df_mba_cluster

# df_mba_cluster = prepare_mba_cluster(df_mba, rfm_kmeans) # pindahin ke page

# 8.2. Build basket per cluster
@st.cache_data
def build_basket_cluster(df_mba_cluster, cluster_id):
    subset = df_mba_cluster[df_mba_cluster['Cluster'] == cluster_id]
    basket = (
        subset
        .groupby(['TransactionId','ItemDescription'])['NumberOfItemsPurchased']
        .sum()
        .unstack()
        .fillna(0)
    )
    basket = (basket > 0).astype(int)
    return basket

# 8.3. Run Apriori per cluster
@st.cache_data
def run_mba_cluster(df_mba_cluster, cluster_id, min_support=0.03, min_lift=1.2, min_conf=0.3):
    basket = build_basket_cluster(df_mba_cluster, cluster_id)
    frequent = apriori(basket, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent, metric='lift', min_threshold=1)
    rules = rules[(rules['lift'] > min_lift) & (rules['confidence'] > min_conf)].copy()

    # Ambil item pertama dari antecedents & consequents
    rules['ante'] = rules['antecedents'].apply(lambda x: list(x)[0])
    rules['cons'] = rules['consequents'].apply(lambda x: list(x)[0])

    # Normalisasi pair (A,B == B,A)
    rules['pair'] = rules.apply(lambda x: tuple(sorted([x['ante'], x['cons']])), axis=1)

    # Drop duplikat, keep lift tertinggi
    rules_unique = rules.sort_values('lift', ascending=False).drop_duplicates(subset='pair', keep='first').reset_index(drop=True)

    # Build combo text
    rules_unique['combo'] = rules_unique.apply(lambda x: f"{x['ante']} → {x['cons']}", axis=1)

    # Dictionary untuk lookup cepat
    combo_dict = {
        row['combo']: {
            "ante": row['ante'],
            "cons": row['cons'],
            "support": row['support'],
            "confidence": row['confidence'],
            "lift": row['lift']
        }
        for _, row in rules_unique.iterrows()
    }

    return rules_unique, combo_dict

# 8.4. Hitung rata-rata basket size per cluster
@st.cache_data
def basket_size_per_cluster(df_mba_cluster):
    return (
        df_mba_cluster
        .groupby(['Cluster','TransactionId'])['NumberOfItemsPurchased']
        .sum()
        .groupby('Cluster')
        .mean()
        .reset_index()
        .rename(columns={'NumberOfItemsPurchased':'AvgBasketSize'})
    )

# 8.5. Top 3 kategori per cluster
@st.cache_data
def top3_categories_per_cluster(df_mba_cluster):
    return (
        df_mba_cluster
        .groupby(['Cluster','ItemDescription'])['NumberOfItemsPurchased']
        .sum()
        .reset_index()
        .sort_values(['Cluster','NumberOfItemsPurchased'], ascending=[True, False])
        .groupby('Cluster')
        .head(3)
    )

# rules_cluster_0, combo_dict_0 = run_mba_cluster(df_mba_cluster, 0)
# rules_cluster_1, combo_dict_1 = run_mba_cluster(df_mba_cluster, 1)
# rules_cluster_2, combo_dict_2 = run_mba_cluster(df_mba_cluster, 2)
# rules_cluster_3, combo_dict_3 = run_mba_cluster(df_mba_cluster, 3)

# basket_avg = basket_size_per_cluster(df_mba_cluster)
# top3_items = top3_categories_per_cluster(df_mba_cluster)

#########################################
# SIMPAN SEMUA FUNGSI KE st.session_state
#########################################

if "rfm_final_2" not in st.session_state:
    st.session_state.rfm_final_2 = None

if "kmeans_result" not in st.session_state:
    st.session_state.kmeans_result = None

if "gmm_result" not in st.session_state:
    st.session_state.gmm_result = None

if "inverse_rfm" not in st.session_state:
    st.session_state.inverse_rfm = None

if "mba_global" not in st.session_state:
    st.session_state.mba_global = None

if "mba_cluster" not in st.session_state:
    st.session_state.mba_cluster = {}
    
if "basket_avg" not in st.session_state:
    st.session_state.basket_avg = None

if "top3_items" not in st.session_state:
    st.session_state.top3_items = None

#########################################
# 9. SIDEBAR 
#########################################

if "page" not in st.session_state:
    st.session_state.page = "Home"

def set_page(p):
    st.session_state.page = p

st.sidebar.title("📌 Navigation")
st.sidebar.button("Home", on_click=set_page, args=("Home",)),
st.sidebar.button("Analisis RFM", on_click=set_page, args=("Analisis RFM",))
st.sidebar.button("Clustering", on_click=set_page, args=("Clustering",))
st.sidebar.button("Market Basket Analysis", on_click=set_page, args=("Market Basket Analysis",))
st.sidebar.button("Interpretasi Bisnis", on_click=set_page, args=("Interpretasi Bisnis",))

page = st.session_state.page

st.sidebar.markdown("---")
st.sidebar.write("Customer Segmentation For Personalized Retail Marketing")
st.sidebar.write("Oleh tim A25-CS281")

#########################################
# 10. PAGE: HOME
#########################################
if page == "Home":
    st.title("Customer Segmentation For Personalized Retail Marketing")
    st.write("Disusun oleh tim **A25-CS281** - Asah led by Dicoding in association with Accenture")

    st.markdown("""
    ### Tujuan Proyek
    Proyek ini bertujuan untuk melakukan **segmentasi pelanggan** guna:
    - Mengidentifikasi karakteristik dan pola perilaku pelanggan.
    - Mengelompokkan pelanggan berdasarkan nilai dan aktivitas transaksi.
    - Memberikan rekomendasi strategi bisnis yang lebih terukur dan berbasis data.

    Pendekatan ini diharapkan dapat membantu pengambilan keputusan yang lebih efektif dan efisien.

    ---

    ### Anggota Proyek
    - Gisella Serafina Lukman Mebanua  (M284D5X0696) - Machine Learning - Universitas Negeri Surabaya  
    - Daviq Qaishar (M015D5Y0441) - Machine Learning - Universitas Negeri Yogyakarta
    - Fahalliza Nastitie Dewi (M223D5X0561) - Machine Learning - Universitas Islam Negeri Sunan Kalijaga
    """)
    st.markdown("---")
    st.info("Silakan pilih menu pada sidebar untuk navigasi halaman.")


#########################################
# 11. PAGE: RFM
#########################################
elif page == "Analisis RFM":
    st.title("Analisis RFM")

    # Hitung RFM saat masuk halaman ini
    rfm = calculate_rfm(df_clustering)
    rfm_final_2, pt, scaler = prepare_rfm_for_clustering(rfm)

    # Simpan ke session_state buat halaman Clustering
    st.session_state.rfm_final_2 = rfm_final_2
    st.session_state.pt = pt
    st.session_state.scaler = scaler

    row1_col1, row1_col2, row1_col3 = st.columns(3)
    row2_col1, row2_col2, row2_col3 = st.columns(3) 
    
    total_customers = rfm.shape[0]
    products = df_clustering['ItemDescription'].nunique()
    total_revenue = rfm['Monetary'].sum()
    total_recency = rfm['Recency'].sum()
    avg_revenue = rfm['Monetary'].mean()
    avg_recency = rfm['Recency'].mean()
    avg_frequency = rfm['Frequency'].mean()
    # kpi cards di atas
    with stylable_container(
        key="container_metrics_eda",
        css_styles=styles.card
    ):
        with row1_col1:
            metric_card("Rata-rata Waktu Transaksi Terakhir", f"{avg_recency:,.2f} Hari")

        with row1_col2:
            metric_card("Rata-rata Transaksi per Customer", f"{avg_frequency:,.2f} Kali")

        with row1_col3:
            metric_card("Rata-rata Pendapatan per Customer", format_currency(avg_revenue))

        with row2_col1:
            metric_card("Total Pelanggan", f"{total_customers:,.0f}")

        with row2_col2:
            metric_card("Total Produk", f"{products} Produk")

        with row2_col3:
            metric_card("Total Pendapatan", format_currency(total_revenue))

    # tabs
    tab1, tab2, tab3 = st.tabs(["RFM Overview", "RFM Movements", "Segmentation"])
    with tab1:
        st.subheader("Data Overview")

        ####### TOP CUST DAN TOP PRODUCTS #########
        col_left, col_right = st.columns([1.3, 2])

        ########## kolom kiri: top customers ###########
        with col_left:
            top_monetary = rfm[['CustomerID', 'Recency', 'Frequency', 'Monetary']].sort_values(
                'Monetary', ascending=False).head(3)
            top_monetary['Monetary'] = top_monetary['Monetary'].apply(format_currency)

            top_frequency = rfm[['CustomerID', 'Recency', 'Frequency', 'Monetary']].sort_values(
                'Frequency', ascending=False).head(3)
            top_frequency['Monetary'] = top_frequency['Monetary'].apply(format_currency)

            top_recency = rfm[['CustomerID', 'Recency', 'Frequency', 'Monetary']].sort_values(
                'Recency').head(3)
            top_recency['Monetary'] = top_recency['Monetary'].apply(format_currency)

            with stylable_container(
                key="container_top_recency",
                css_styles=styles.card
            ):
                st.write('### Top Customer by RFM')
                st.write("##### Recency")
                st.dataframe(top_recency, hide_index=True)

                st.write("##### Frequency")
                st.dataframe(top_frequency, hide_index=True)

                st.write("##### Monetary")
                st.dataframe(top_monetary, hide_index=True)

        # ######### kolom kanan: top products ##########
        with col_right:
            with stylable_container(
                key="container_kanan",
                css_styles=styles.card
            ):
                st.subheader("Produk Populer")
                
                top_products = df_clustering['ItemDescription'].value_counts().head(10)
                df_plot = top_products.reset_index()
                df_plot.columns = ["Produk", "Jumlah"]

                fig = px.bar(
                    df_plot,
                    x="Jumlah",
                    y="Produk",
                    orientation="h",
                    color="Jumlah",
                    color_continuous_scale=["#d0e8ff", "#a8d8ff", "#80c8ff", "#58b7ff"],
                )

                fig.update_yaxes(
                    categoryorder="total ascending"   #terbesar di atas
                )
                
                fig.update_layout(
                    height=618,
                    plot_bgcolor="rgba(0,0,0,0)",  
                    paper_bgcolor="rgba(0,0,0,0)", 
                    font_color="white",
                    coloraxis_showscale=False,
                    margin=dict(l=10, r=10, t=10, b=10)
                )

                fig.update_xaxes(showgrid=False, color="white")
                fig.update_yaxes(showgrid=False, color="white")

                st.plotly_chart(fig, use_container_width=True)

        st.divider()
        
        ###### HEATMAP AND RFM TABLE ######
        col1, col2 = st.columns(2)

        with col1:
            ###### heatmap ########
            with stylable_container(
                key="segmentation_heatmap",
                css_styles=styles.card
            ):
                st.subheader("RFM Score Heatmap")
                agg = rfm.groupby(["R_Score", "F_Score", "M_Score"]).agg(
                    customer_count=("CustomerID", "count"),
                    avg_recency=("Recency", "mean"),
                    avg_frequency=("Frequency", "mean"),
                    avg_monetary=("Monetary", "mean")
                ).reset_index()

                pivot = agg.pivot_table(
                    index=["R_Score", "F_Score"],
                    columns="M_Score",
                    values="customer_count",
                    fill_value=0
                )

                pivot = pivot.sort_index(ascending=[False, False])
                pivot = pivot.reindex(sorted(pivot.columns, reverse=True), axis=1)

                tooltip = []
                for (r, f), row in pivot.iterrows():
                    tt_row = []
                    for m in pivot.columns:
                        if row[m] == 0:
                            tt_row.append(f"<b>RFM : {r}{f}{m}</b><br><i>No Data</i>")
                            continue
                        
                        temp = agg[(agg["R_Score"]==r) & 
                                (agg["F_Score"]==f) & 
                                (agg["M_Score"]==m)]
                        t = temp.iloc[0]
                        
                        text = (
                            f"<b>RFM : {r}{f}{m}</b><br>"
                            f"Recency: {r}<br>"
                            f"Frequency: {f}<br>"
                            f"Monetary: {m}<br><br>"
                            f"Customer Count: {int(t.customer_count)}<br>"
                            f"Avg Recency: {t.avg_recency:.1f} days<br>"
                            f"Avg Frequency: {t.avg_frequency:.1f} times<br>"
                            f"Avg Monetary: {t.avg_monetary:,.0f}"
                        )
                        tt_row.append(text)

                    tooltip.append(tt_row)

                # --- heatmap z ---
                z = pivot.values.copy()
                z[z == 0] = -1      # Sel kosong = value -1

                # --- custom colorscale ---
                colorscale = [
                    [0.00, "white"],     # kosong
                    [0.01, "#BBD6FF"], # awal
                    [1.00, "#08306B"]  # akhir
                ]

                fig = go.Figure(
                    data=go.Heatmap(
                        z=z,
                        x=pivot.columns,
                        y=[f"R{r} - F{f}" for r, f in pivot.index],
                        text=tooltip,
                        hoverinfo="text",
                        colorscale=colorscale,
                        showscale=True
                    )
                )

                fig.update_layout(
                    height=700,
                    width=500,
                    xaxis_title="Monetary (M)",
                    yaxis_title="Recency (R) & Frequency (F)",
                    xaxis=dict(side="top"),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    margin=dict(t=50, l=0, r=0, b=20)
                )

                st.plotly_chart(fig)

        with col2:
            ##### table rfm per cust ######
            with stylable_container(
                key="rfm_table_cust",
                css_styles=styles.card
            ):
                st.subheader("RFM Table per Customer")
                rfm_table = rfm_customer_table(rfm, rfm_kmeans, segment_labels)
                st.dataframe(rfm_table, height=700, hide_index=True)

        st.divider()

    with tab2:
        st.subheader("RFM Movements")
        
        # PLOT JUMLAH PER SEGMEN PER BULAN
        rfm = rfm.rename(columns={'CustomerID': 'UserId'})
        rfm_kmeans = rfm_kmeans.rename(columns={'CustomerID': 'UserId'})
        df_merged = (
            df_clustering
            .merge(rfm, on='UserId', how='left')
            .merge(rfm_kmeans[['UserId', 'Cluster']], on='UserId', how='left')
        )

        df_merged['Segment'] = df_merged['Cluster'].map(segment_labels)
        df_merged['TransactionTime'] = pd.to_datetime(df_merged['TransactionTime'])
        df_merged['Month'] = df_merged['TransactionTime'].dt.to_period('M').astype(str)

        df_monthly = (
            df_merged.groupby(['Month','Segment'])['UserId']
            .nunique()
            .reset_index(name='Users')
        )
        total_users = df_monthly.groupby('Month')['Users'].sum()
        valid_months = total_users[total_users > 0].index
        df_monthly = df_monthly[df_monthly['Month'].isin(valid_months)]

        df_monthly_sorted = pd.to_datetime(df_monthly['Month']).sort_values()
        gaps = df_monthly_sorted.diff().dt.days
        if (gaps > 90).any():
            cutoff = df_monthly_sorted[gaps > 90].iloc[0]
            df_monthly = df_monthly[pd.to_datetime(df_monthly['Month']) < cutoff]
        segment_order = ["Top Customer", "Loyalist", "At-risk", "Low-Value"]

        color_map = {
            "Top Customer": "#E67E22",
            "Loyalist": "#8E44AD",
            "At-risk": "#3498DB",
            "Low-Value": "#27AE60",
        }

        fig = go.Figure()
        for seg in segment_order:
            df_seg = df_monthly[df_monthly['Segment'] == seg]
            
            fig.add_trace(go.Bar(
                x=df_seg['Month'],
                y=df_seg['Users'],
                name=seg,
                text=df_seg['Users'],
                marker_color=color_map[seg],
                textposition='inside',
                insidetextanchor='middle',
                textfont=dict(color='white', size=13)
            ))

        fig.update_layout(
            barmode='stack',
            height=550,
            title="Customers per Segment by Month",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter", size=14),
            margin=dict(l=40, r=40, t=60, b=40),
        )

        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        #  PLOT PERSENTASE SEGMENT PER BULAN
        df_pct = (
            df_monthly.groupby('Month')['Users']
            .sum()
            .reset_index()
            .rename(columns={'Users': 'TotalUsers'})
        )
        df_pct = df_monthly.merge(df_pct, on='Month')
        df_pct['Pct'] = df_pct['Users'] / df_pct['TotalUsers'] * 100

        fig_pct = go.Figure()

        for seg in segment_order:
            df_seg = df_pct[df_pct['Segment'] == seg]
            
            fig_pct.add_trace(go.Bar(
                x=df_seg['Month'],
                y=df_seg['Pct'],
                name=seg,
                text=df_seg['Pct'].round(2).astype(str) + '%',
                textposition='inside',
                insidetextanchor='middle',
                marker_color=color_map[seg],
                textfont=dict(color='white', size=13)
            ))

        fig_pct.update_layout(
            barmode='stack',
            height=550,
            title="% of Customers per Segment by Month",
            yaxis=dict(ticksuffix="%", range=[0, 100]),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter", size=14),
            margin=dict(l=40, r=40, t=60, b=40)
        )

        st.plotly_chart(fig_pct, use_container_width=True)

    with tab3:

        # Pastikan clustering sudah selesai
        if "kmeans_result" not in st.session_state or st.session_state.kmeans_result is None:
            st.warning("Jalankan K-Means terlebih dahulu di halaman Clustering.")
            st.stop()
    
        # Ambil data clustering dari session_state
        rfm_kmeans = st.session_state.kmeans_result["rfm_kmeans"]
        scaler = st.session_state.scaler
        pt = st.session_state.pt
    
        # Hitung inverse RFM (HANYA SAAT TAB3 DIBUKA!)
        rfm_kmeans_inverse = inverse_rfm(rfm_kmeans, scaler, pt)

        # UI
        col1, col2 = st.columns([1.5, 2])

        # TREEMAP SEGMENT
        with col1:
            with stylable_container(
                key="container_segmentation_overview",
                css_styles=styles.card
            ):
                st.subheader("Segmentation Overview")
                cluster_counts['Cluster'] = cluster_counts['Cluster'].map(segment_labels)
                fig = px.treemap(
                    cluster_counts,
                    path=['Cluster'],
                    values='Count',
                    color='Cluster',
                    color_discrete_sequence=['#ff7f0e', '#d62728', '#1f77b4', '#2ca02c']
                )

                fig.update_traces(
                    tiling=dict(
                        packing='squarify',
                        squarifyratio=1.0
                    ),
                    marker=dict(
                        line=dict(color="black", width=2)
                    )
                )

                fig.update_layout(
                    width=430,
                    height=383.5,
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(t=15, l=25, r=25, b=25)
                )

                st.plotly_chart(fig, use_container_width=False)

        # BARPLOT SEGMENT
        with col2:
            with stylable_container(
                key="container_cluster_summary",
                css_styles=styles.card
            ):
                st.subheader("Segment Summary")

                rfm_kmeans_with_orig = rfm_kmeans.copy()
                rfm_kmeans_with_orig['Recency_Orig'] = rfm['Recency']
                rfm_kmeans_with_orig['Frequency_Orig'] = rfm['Frequency']
                rfm_kmeans_with_orig['Monetary_Orig'] = rfm['Monetary']

                metric_option = st.selectbox(
                    "Pilih metric yang ingin ditampilkan",
                    ["Recency", "Frequency", "Monetary"]
                )

                metric_map_orig = {
                    "Recency": "Recency_Orig",
                    "Frequency": "Frequency_Orig",
                    "Monetary": "Monetary_Orig"
                }

                selected_col = metric_map_orig[metric_option]

                data_avg = (
                    rfm_kmeans_with_orig
                    .groupby("Cluster")[[selected_col]]
                    .mean()
                    .reset_index()
                )

                # Rename supaya nama kolom sama dengan pilihan dropdown (biar plot konsisten)
                data_avg = data_avg.rename(columns={selected_col: metric_option})
                data_avg["Segment"] = data_avg["Cluster"].map(segment_labels)
                data_avg['text'] = data_avg[metric_option].apply(lambda x: format_metric_text(x, metric_option))
                data_avg = data_avg.sort_values(metric_option)

                invert = (metric_option == "Recency")  # Recency paling kecil = terbaik
                if invert:
                    color_discrete_sequence=['#2ca02c', '#1f77b4', '#d62728', '#ff7f0e' ]  # hijau untuk rendah
                else:
                    color_discrete_sequence=['#ff7f0e', '#d62728', '#1f77b4', '#2ca02c']

                fig = px.bar(
                    data_avg,
                    y="Segment",
                    x=metric_option,
                    orientation="h",
                    color="Segment",
                    color_discrete_sequence=color_discrete_sequence,
                    text='text'
                )

                fig.update_traces(
                    texttemplate="%{text}",
                    textposition="outside",
                    cliponaxis=False
                )

                fig.update_layout(
                    height=300,
                    coloraxis_showscale=False,
                    title=f"Rata-rata {metric_option}",
                    xaxis_title=metric_option,
                    yaxis_title="Customer Segment",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(t=50, l=25, r=50, b=25),
                    showlegend=False
                )

                st.plotly_chart(fig, use_container_width=True)
        
        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            # HITUNG RINGKASAN PER CLUSTER
            rfm_inverse_df = rfm_kmeans_inverse[['Recency', 'Frequency', 'Monetary', 'Cluster']].copy()
            inverse_cols = ['Recency', 'Frequency', 'Monetary']

            summary_rows_inverse = []

            for cluster in sorted(rfm_inverse_df['Cluster'].unique()):
                data = rfm_inverse_df[rfm_inverse_df['Cluster'] == cluster]
                for metric in inverse_cols:
                    summary_rows_inverse.append([metric, 'mean', cluster, data[metric].mean()])
                    summary_rows_inverse.append([metric, 'median', cluster, data[metric].median()])

                    m = mode(data[metric], keepdims=False)
                    mode_value = np.atleast_1d(m.mode)[0]
                    summary_rows_inverse.append([metric, 'mode', cluster, mode_value])

                    summary_rows_inverse.append([metric, 'max', cluster, data[metric].max()])
                    summary_rows_inverse.append([metric, 'min', cluster, data[metric].min()])

            summary_inverse_df = pd.DataFrame(
                summary_rows_inverse,
                columns=['Metric', 'Statistic', 'Cluster', 'Value']
            )

            stat_order = ['mean', 'median', 'mode', 'max', 'min']
            summary_inverse_df['Statistic'] = pd.Categorical(
                summary_inverse_df['Statistic'],
                categories=stat_order,
                ordered=True
            )

            summary_inverse_pivot = summary_inverse_df.pivot(
                index=['Metric', 'Statistic'],
                columns='Cluster',
                values='Value'
            )

            summary_inverse_pivot.columns = [f'Cluster {c}' for c in summary_inverse_pivot.columns]

            st.subheader("Summary Statistik RFM Setelah Inverse")
            st.dataframe(summary_inverse_pivot.style.format("{:,.2f}"), height=563, use_container_width=True)


        with col2:
            # VISUALISASI 3D RFM (ASLI)
            st.subheader("Visualisasi 3D RFM Per Cluster Setelah Inverse")

            rfm_kmeans_inverse_3d = rfm_kmeans_inverse.copy()
            labels = rfm_kmeans_inverse_3d['Cluster']

            color_map = {
                "0": "#ff7f0e",
                "1": "#d62728",
                "2": "#1f77b4",
                "3": "#2ca02c",
            }

            fig = go.Figure()

            for k in sorted(labels.unique()):
                cluster_data = rfm_kmeans_inverse_3d[labels == k]
                color = color_map[str(k)]

                fig.add_trace(go.Scatter3d(
                    x=cluster_data['Recency'],
                    y=cluster_data['Frequency'],
                    z=cluster_data['Monetary'],
                    mode='markers',
                    marker=dict(size=4, color=color),
                    name=f"Cluster {k}",

                    customdata=np.stack([
                        cluster_data['Recency'],
                        cluster_data['Frequency'],
                        cluster_data['Monetary']
                    ], axis=-1),

                    hovertemplate=
                        "<b>Cluster %{text}</b><br>" +
                        "Recency = %{customdata[0]:,.2f}<br>" +
                        "Frequency = %{customdata[1]:,.2f}<br>" +
                        "Monetary = %{customdata[2]:,.2f}<br>" +
                        "<extra></extra>",
                    text=[k] * len(cluster_data)
                ))

            fig.update_layout(
                title="",
                width=500,
                height=563,
                legend=dict(
                    font=dict(size=12),
                    itemsizing="constant"
                ),
                scene=dict(
                    xaxis_title='Recency (Original)',
                    yaxis_title='Frequency (Original)',
                    zaxis_title='Monetary (Original)',
                )
            )

            st.plotly_chart(fig, use_container_width=True) 

# =========================
#   PAGE: CLUSTERING
# =========================
elif page == "Clustering":
    st.title("Clustering")

    if st.session_state.rfm_final_2 is None:
        st.warning("Silakan buka halaman Analisis RFM terlebih dahulu untuk menyiapkan data.")
        st.stop()

    rfm_final_2 = st.session_state.rfm_final_2

    tab1, tab2 = st.tabs(["K-Means", "Gaussian Mixture Model"]) 
 
    # ============================
    # KMEANS TAB
    # ============================

    with tab1:               
        st.subheader("K-Means")

        # run kmeans
        rfm_kmeans, wcss, kmeans_metrics, kmeans_model, pca_kmeans, centroids_pca, cluster_counts, segment_labels = run_kmeans(rfm_final_2)

        # simpan hasil ke session_state
        st.session_state.kmeans_result = {
            "rfm_kmeans": rfm_kmeans,
            "scaler": scaler,
            "pt": pt,
            "wcss": wcss,
            "metrics": kmeans_metrics,
            "model": kmeans_model,
            "pca": pca_kmeans,
            "centroids": centroids_pca,
            "clusters": cluster_counts,
            "labels": segment_labels,
        }
    
        # ============================
        # METRIC TABLE
        st.write("### 📊 Evaluation Metrics")
        st.dataframe(kmeans_metrics, use_container_width=True)

        st.markdown("---")
        # ============================
        # ELBOW METHOD 
        st.write("### 📈 Elbow Method") 

        X = rfm_final_2[['Recency_Scaled', 'Frequency_Scaled','Monetary_Scaled']]

        # Hitung WCSS untuk jumlah kluster yang berbeda
        wcss = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
            kmeans.fit(X)
            wcss.append(kmeans.inertia_)

        # Plot Elbow Method dengan Streamlit
        fig_elbow, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(1, 11), wcss, marker='o', linestyle='--', linewidth=2)
        ax.set_title('')
        ax.set_xlabel('Number of Clusters (K)')
        ax.set_ylabel('WCSS')
        ax.grid(True)

        # Garis vertikal untuk optimal k
        optimal_k = 4
        ax.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal K = {optimal_k}')
        ax.legend()

        st.pyplot(fig_elbow)


        st.markdown("---")

        # ============================
        # PCA 2D Visualization
        st.write("### 🎨 2D Visualization (PCA)")

        # Dataset
        X = rfm_kmeans[['Recency_Scaled', 'Frequency_Scaled', 'Monetary_Scaled']]
        labels = rfm_kmeans['Cluster']

        # PCA transform
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(X)
        rfm_kmeans['PCA1'] = pca_result[:, 0]
        rfm_kmeans['PCA2'] = pca_result[:, 1]

        # Centroid transform
        centroids = kmeans_model.cluster_centers_
        centroids_df = pd.DataFrame(centroids, columns=X.columns)
        centroids_pca = pca.transform(centroids_df)

        # Colors
        palette = {0: 'orange', 1: 'red', 2: 'blue', 3: 'green'}

        # Figure
        fig2d, ax = plt.subplots(figsize=(10, 7))

        sns.scatterplot(
            x=rfm_kmeans['PCA1'],
            y=rfm_kmeans['PCA2'],
            hue=rfm_kmeans['Cluster'],
            palette=palette,
            s=80,
            alpha=0.85,
            ax=ax
        )

        # Centroid
        ax.scatter(
            centroids_pca[:, 0],
            centroids_pca[:, 1],
            s=300,
            c='black',
            marker='X',
            edgecolors='white',
            linewidths=2,
            label='Centroid'
        )

        ax.set_title("")
        ax.set_xlabel("PCA Component 1")
        ax.set_ylabel("PCA Component 2")
        ax.grid(True)
        ax.legend(title='Cluster')

        st.pyplot(fig2d)

        st.markdown("---")

        # ============================
        # 3D Visualization
        st.write("### 🌐 3D Visualization (RFM Space)")

        labels = rfm_kmeans['Cluster'].astype(str)
        color_map = {
            "0": "#ff7f0e",
            "1": "#d62728",
            "2": "#1f77b4",
            "3": "#2ca02c",
        }

        fig3d = go.Figure()

        for k in sorted(labels.unique()):
            cluster_data = rfm_kmeans[labels == k]
            color = color_map[str(k)]

            fig3d.add_trace(go.Scatter3d(
                x=cluster_data['Recency_Scaled'],
                y=cluster_data['Frequency_Scaled'],
                z=cluster_data['Monetary_Scaled'],
                mode='markers',
                marker=dict(size=4, color=color),
                name=f"Cluster {k}",

                customdata=np.stack([
                    cluster_data['Recency_Scaled'],
                    cluster_data['Frequency_Scaled'],
                    cluster_data['Monetary_Scaled']
                ], axis=-1),

                hovertemplate=
                    "<b>Cluster %{text}</b><br>" +
                    "Recency (Scaled) = %{customdata[0]:.4f}<br>" +
                    "Frequency (Scaled) = %{customdata[1]:.4f}<br>" +
                    "Monetary (Scaled) = %{customdata[2]:.4f}<br>" +
                    "<extra></extra>",

                text=[k] * len(cluster_data)
            ))

        fig3d.update_layout(
            title="",
            width=900,
            height=700,
            legend=dict(          
                font=dict(size=16),
                itemsizing="constant"
            ),
            scene=dict(
                xaxis_title='Recency (Scaled)',
                yaxis_title='Frequency (Scaled)',
                zaxis_title='Monetary (Scaled)'
            )
        )

        st.plotly_chart(fig3d, use_container_width=True)


    # ============================
    # GMM TAB
    # ============================

    with tab2:
        st.subheader("Gaussian Mixture Model (GMM)")

        # run GMM
        rfm_gmm_final, gmm_metrics, model_gmm = run_gmm_clustering(rfm_final_2)

        # simpan hasil ke session_state
        st.session_state.gmm_result = {
            "rfm_gmm": rfm_gmm_final,
            "metrics": gmm_metrics,
            "model": model_gmm,
        }
        
        st.write("### 📊 Evaluation Metrics")
        st.dataframe(gmm_metrics, use_container_width=True)

        st.markdown("---")

        # ======================================================
        # 2D PCA VISUALIZATION 
        st.write("### 🎨 2D Visualization (PCA)")

        X = rfm_gmm_final[['Recency_Scaled','Frequency_Scaled','Monetary_Scaled']]
        labels = rfm_gmm_final['Cluster']

        pca = PCA(n_components=2, random_state=42)
        pca_result = pca.fit_transform(X)

        rfm_gmm_final['PCA1'] = pca_result[:, 0]
        rfm_gmm_final['PCA2'] = pca_result[:, 1]

        palette = {0: 'orange', 1: 'red', 2: 'blue', 3: 'green'}

        fig, ax = plt.subplots(figsize=(10, 7))
        sns.scatterplot(
            x=rfm_gmm_final['PCA1'],
            y=rfm_gmm_final['PCA2'],
            hue=rfm_gmm_final['Cluster'],
            palette=palette,
            s=80,
            alpha=0.85,
            ax=ax
        )

        ax.set_title("")  
        ax.set_xlabel("PCA Component 1")
        ax.set_ylabel("PCA Component 2")
        ax.grid(True)

        # Ellipse
        means_df = pd.DataFrame(model_gmm.means_, columns=X.columns)
        means_pca = pca.transform(means_df)
        covariances = model_gmm.covariances_
        cov_pca_list = []

        for cov in covariances:
            cov_pca = pca.components_ @ cov @ pca.components_.T
            cov_pca_list.append(cov_pca)

        colors = ['orange', 'red', 'blue', 'green']

        def plot_gmm_ellipse(mean, cov, ax, color):
            vals, vecs = np.linalg.eigh(cov)
            scale = 5
            vals = scale * vals
            order = vals.argsort()[::-1]
            vals, vecs = vals[order], vecs[:, order]
            theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
            width, height = 2 * np.sqrt(vals)
            ell = Ellipse(
                xy=mean,
                width=width,
                height=height,
                angle=theta,
                edgecolor=color,
                fc='none',
                lw=2,
                linestyle='--'
            )
            ax.add_patch(ell)

        for i in range(len(means_pca)):
            plot_gmm_ellipse(means_pca[i], cov_pca_list[i], ax, colors[i])

        st.pyplot(fig)

        st.markdown("---")

        # ======================================================
        # 3D VISUALIZATION GMM 
        st.write("### 🌐 3D Visualization (RFM Space)")

        X = rfm_gmm_final[['Recency_Scaled','Frequency_Scaled','Monetary_Scaled']]
        labels = rfm_gmm_final['Cluster']

        means = model_gmm.means_
        covs = model_gmm.covariances_

        color_map = {
            "0": "#ff7f0e",
            "1": "#d62728",
            "2": "#1f77b4",
            "3": "#2ca02c",
        }

        def get_ellipsoid_wireframe(mean, cov, scale=3, n_points=25):
            vals, vecs = np.linalg.eigh(cov)
            radii = scale * np.sqrt(vals)
            u = np.linspace(0, 2*np.pi, n_points)
            v = np.linspace(0, np.pi, n_points)
            x = radii[0] * np.outer(np.cos(u), np.sin(v))
            y = radii[1] * np.outer(np.sin(u), np.sin(v))
            z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
            ellipsoid = np.array([x, y, z])
            ellipsoid = vecs @ ellipsoid.reshape(3, -1)
            ellipsoid = ellipsoid.reshape(3, n_points, n_points)
            return (ellipsoid[0] + mean[0], ellipsoid[1] + mean[1], ellipsoid[2] + mean[2])

        fig = go.Figure()

        for k in sorted(labels.unique()):
            cluster_data = rfm_gmm_final[labels == k]
            color = color_map[str(k)]

            fig.add_trace(go.Scatter3d(
                x=cluster_data['Recency_Scaled'],
                y=cluster_data['Frequency_Scaled'],
                z=cluster_data['Monetary_Scaled'],
                mode='markers',
                marker=dict(size=4, color=color),
                name=f"Cluster {k}",
                hovertemplate=(
                    "<b>Cluster %{text}</b><br>" +
                    "Recency = %{x:.3f}<br>" +
                    "Frequency = %{y:.3f}<br>" +
                    "Monetary = %{z:.3f}<br><extra></extra>"
                ),
                text=[k] * len(cluster_data)
            ))

        for i in range(len(means)):
            Xell, Yell, Zell = get_ellipsoid_wireframe(means[i], covs[i], scale=3)
            fig.add_trace(go.Scatter3d(
                x=Xell.flatten(),
                y=Yell.flatten(),
                z=Zell.flatten(),
                mode="lines",
                line=dict(color=color_map[str(i)], width=2),
                name=f"Ellipsoid {i}",
                hoverinfo="skip"
            ))

        fig.update_layout(
            title="", 
            width=900,
            height=700,
            legend=dict(   
                font=dict(size=16),
                itemsizing="constant"
            ),
            scene=dict(
                xaxis_title='Recency (Scaled)',
                yaxis_title='Frequency (Scaled)',
                zaxis_title='Monetary (Scaled)'
            )
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")


# =========================
#   PAGE: MARKET BASKET ANALYSIS
# =========================
elif page == "Market Basket Analysis":
    st.title("Market Basket Analysis (MBA)")

    tab_global, tab_cluster = st.tabs(["MBA Global", "MBA per Cluster"])


    # ============================================
    # 1) TAB MBA GLOBAL
    # ============================================
    with tab_global:

        # --- compute ONLY ONCE ---
        if st.session_state.mba_global is None:
            df_mba = data_mba(df)
            basket = build_basket(df_mba)
            rules = run_mba(basket)

            # combo building
            rules['ante'] = rules['antecedents'].apply(lambda x: list(x)[0])
            rules['cons'] = rules['consequents'].apply(lambda x: list(x)[0])
            rules['pair'] = rules.apply(lambda r: tuple(sorted([r['ante'], r['cons']])), axis=1)

            rules_unique = (
                rules.sort_values("lift", ascending=False)
                .drop_duplicates(subset="pair", keep="first")
                .reset_index(drop=True)
            )

            rules_unique['combo'] = rules_unique.apply(lambda x: f"{x['ante']} → {x['cons']}", axis=1)

            combo_dict = {
                row['combo']: {
                    "ante": row['ante'],
                    "cons": row['cons'],
                    "support": row['support'],
                    "confidence": row['confidence'],
                    "lift": row['lift']
                }
                for _, row in rules_unique.iterrows()
            }

            st.session_state.mba_global = {
                "df_mba": df_mba,
                "rules_unique": rules_unique,
                "combo_dict": combo_dict,
            }

        # --- load from session ---
        df_mba = st.session_state.mba_global["df_mba"]
        rules_unique = st.session_state.mba_global["rules_unique"]
        combo_dict = st.session_state.mba_global["combo_dict"]

        st.subheader("MBA Global Product Combos")

        selected_combo = st.selectbox(
            "Pilih combo produk",
            rules_unique['combo'].tolist()
        )

        st.markdown("---")

        info = combo_dict[selected_combo]
        ante, cons = info["ante"], info["cons"]

        st.write(f"**Jika pelanggan membeli:** `{ante}` → **Mereka mungkin juga membeli:** `{cons}`")

        col1, col2, col3 = st.columns(3)
        
        col1.metric("Support", f"{info['support']*100:.2f}%")
        col2.metric("Confidence", f"{info['confidence']*100:.2f}%")
        col3.metric("Lift", f"{info['lift']:.2f}")

        st.markdown("---")

        # pelanggan potensial
        customers = (
            df_mba[df_mba["ItemDescription"] == ante]["UserId"]
            .dropna()
            .unique()
            .tolist()
        )

        st.subheader(f"Pelanggan Potensial untuk `{ante}`")
        st.write(f"**{len(customers)} pelanggan ditemukan**")

        customer_grid = to_grid(customers, cols=10)
        st.dataframe(customer_grid)

        st.markdown("---")

        with st.expander("Seluruh Aturan MBA (Diurutkan berdasarkan Lift)"):
            st.dataframe(rules_unique[['ante','cons','support','confidence','lift']])


    # ============================================
    # 2) TAB MBA PER CLUSTER
    # ============================================
    with tab_cluster:

        st.subheader("MBA Per Cluster")

        # compute df_mba_cluster once
        if st.session_state.kmeans_result is None:
            st.warning("Jalankan K-Means terlebih dahulu.")
            st.stop()

        # build cluster mba df if not exist
        if st.session_state.basket_avg is None:
            df_mba = data_mba(df)
            rfm_kmeans = st.session_state.kmeans_result["rfm_kmeans"]

            df_mba_cluster = prepare_mba_cluster(df_mba, rfm_kmeans)
            st.session_state.basket_avg = basket_size_per_cluster(df_mba_cluster)
            st.session_state.top3_items = top3_categories_per_cluster(df_mba_cluster)
            st.session_state.df_mba_cluster = df_mba_cluster
        else:
            df_mba_cluster = st.session_state.df_mba_cluster

        cluster_list = sorted(df_mba_cluster['Cluster'].unique())
        cluster_tabs = st.tabs([f"Cluster {c}" for c in cluster_list])


        # cluster-level tabs
        for idx, c in enumerate(cluster_list):
            with cluster_tabs[idx]:

                # compute mba cluster ONLY ONCE per cluster
                if c not in st.session_state.mba_cluster:
                    rules_cluster, combo_dict_cluster = run_mba_cluster(df_mba_cluster, c)
                    st.session_state.mba_cluster[c] = {
                        "rules_cluster": rules_cluster,
                        "combo_dict_cluster": combo_dict_cluster
                    }

                rules_cluster = st.session_state.mba_cluster[c]["rules_cluster"]
                combo_dict_cluster = st.session_state.mba_cluster[c]["combo_dict_cluster"]

                st.success(f"Cluster **{c}**: {len(rules_cluster)} combo ditemukan")

                if len(rules_cluster) > 0:
                    selected_combo_cluster = st.selectbox(
                        "Pilih combo produk",
                        rules_cluster['combo'].tolist(),
                        key=f"combo_{c}"
                    )

                    info_c = combo_dict_cluster[selected_combo_cluster]
                    ante_c = info_c["ante"]
                    cons_c = info_c["cons"]

                    st.markdown("---")
                    st.write(f"**Jika pelanggan membeli:** `{ante_c}` → **Mereka mungkin juga membeli:** `{cons_c}`")

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Support", f"{info_c['support']*100:.2f}%")
                    col2.metric("Confidence", f"{info_c['confidence']*100:.2f}%")
                    col3.metric("Lift", f"{info_c['lift']:.2f}")

                    st.markdown("---")

                    customers_c = (
                        df_mba_cluster[
                            (df_mba_cluster["ItemDescription"] == ante_c) &
                            (df_mba_cluster["Cluster"] == c)
                        ]["UserId"].dropna().unique().tolist()
                    )

                    st.subheader(f"Pelanggan Potensial Cluster {c}")
                    st.write(f"**{len(customers_c)} pelanggan ditemukan**")

                    customer_grid_c = to_grid(customers_c, cols=10)
                    st.dataframe(customer_grid_c)

                else:
                    st.write("Tidak ada combo untuk cluster ini.")

                st.markdown("---")

                # basket size
                avg_size = st.session_state.basket_avg[
                    st.session_state.basket_avg['Cluster'] == c
                ]['AvgBasketSize'].values[0]

                st.subheader("Rata-rata Ukuran Keranjang")
                st.write(f"Cluster **{c}**: **{avg_size:.2f} item** per transaksi")

                st.markdown("---")

                # top 3 items
                top3_cluster = st.session_state.top3_items[
                    st.session_state.top3_items['Cluster'] == c
                ]
                st.subheader("Top 3 Item Terbanyak")
                st.dataframe(top3_cluster[['ItemDescription','NumberOfItemsPurchased']])





# =========================
#   PAGE: INTERPRETASI BISNIS
# =========================
elif page == "Interpretasi Bisnis":
    st.title("Interpretasi Bisnis")
    st.info("Interpretasi mengacu pada hasil clustering dari **K-Means** karena model K-Means memberi kombinasi terbaik antara skor evaluasi dan visualisasi cluster yang jelas dibanding dengan model GMM. Selain itu, strategi bisnis juga disesuaikan dengan hasil dari **Market Basket Analysis** per cluster dengan Apriori.")
    st.subheader("Interpretasi Pemasaran Per Cluster")
    st.info("Berdasarkan hasil clustering RFM, pelanggan terbagi menjadi empat segmen yang berbeda secara jelas. Analisis ini memadukan profil RFM dan hasil Market Basket Analysis tiap cluster untuk menyusun strategi pemasaran.")

    # =======================
    # CLUSTER 0
    # =======================
    with st.expander("🟧 Cluster 0 — Low-Value Customers"):
        st.subheader("Profil RFM")
        st.markdown("""
        - **Frequency (mean):** 1.77 transaksi  
        - **Monetary (mean):** £1,559  
        - **Recency (mean):** 421 hari
        """)

        st.subheader("Preferensi Produk (Market Basket Analysis)")
        st.markdown("""
        - **Rata-rata item per transaksi:** 394.3  
        - **Top 3 produk:**
            1. ASSORTED COLOUR BIRD ORNAMENT  
            2. WHITE HANGING HEART T-LIGHT HOLDER  
            3. WOODEN STAR CHRISTMAS SCANDINAVIAN  
        - **Kombinasi populer:**  
            - WOODEN HEART CHRISTMAS SCANDINAVIAN + WOODEN STAR CHRISTMAS SCANDINAVIAN
        """)

        st.subheader("Interpretasi & Strategi Bisnis")
        st.markdown("""
        - Cluster 0 terdiri dari pelanggan yang jarang berbelanja (rata-rata tiap pelanggan hanya melakukan kurang dari 2 transaksi selama periode data), belanja dalam jumlah kecil (Monetary rata-rata £1.559), dan tidak terlalu aktif (pelanggan pada cluster ini rata-rata tidak bertransaksi selama 421 hari). Mereka memberikan kontribusi pendapatan yang rendah dan memerlukan strategi promosi dasar untuk mendorong pembelian.
        - Ukuran basket menengah menunjukkan bahwa meski jarang berbelanja, transaksi mereka biasanya terdiri dari beberapa item sekaligus.

        **Strategi:**
        1. Reactivation Campaign: promosi atau email reminder untuk meningkatkan frekuensi belanja.   
        2. Bundle atau Cross-sell: memanfaatkan produk kombinasi populer untuk mendorong pembelian beberapa item per transaksi.  
        3. Targeted Offer/Promo Personal: memberi diskon pada produk sesuai referensi untuk melakukan transaksi lebih sering.
        """)

    # =======================
    # CLUSTER 1
    # =======================
    with st.expander("🟥 Cluster 1 — At-Risk Customers"):
        st.subheader("Profil RFM")
        st.markdown("""
        - **Frequency (mean):** 1.45  
        - **Monetary (mean):** £1,329  
        - **Recency (mean):** 618 hari
        """)

        st.subheader("Preferensi Produk")
        st.markdown("""
        - **Rata-rata item per transaksi:** 348.3  
        - **Top 3 produk:**
            1. WHITE HANGING HEART T-LIGHT HOLDER  
            2. ASSORTED COLOUR BIRD ORNAMENT  
            3. HOMEMADE JAM SCENTED CANDLES  
        - **Kombinasi populer:**  
            - GREEN REGENCY TEACUP + ROSES REGENCY TEACUP  
        """)

        st.subheader("Interpretasi & Strategi Bisnis")
        st.markdown("""
        - Cluster 1 terdiri dari pelanggan yang sangat jarang berbelanja (rata-rata 1–2 transaksi per pelanggan), dengan nilai transaksi rendah (Monetary rata-rata £1.329) dan aktivitas belanja yang lama tidak terjadi (pelanggan pada cluster ini rata-rata tidak bertransaksi selama 618 hari). Pelanggan ini berisiko hilang sehingga memerlukan strategi reaktivasi yang personal.
        - Ukuran basket kecil menandakan mereka membeli sedikit item per transaksi.

        **Strategi:**
        1. Reactivation Campaign: gunakan kombinasi produk populer sebagai penawaran personal untuk mendorong pelanggan kembali aktif.
        2. Targeted Offer/Promo Personal: diskon atau bundle produk yang sesuai dengan preferensi mereka.
        3. Highlight Top Produk & Kombinasi: mempermudah pelanggan memilih produk, mendorong transaksi kembali.
        """)

    # =======================
    # CLUSTER 2
    # =======================
    with st.expander("🔵 Cluster 2 — Loyal Customers"):
        st.subheader("Profil RFM")
        st.markdown("""
        - **Frequency (mean):** 7.94  
        - **Monetary (mean):** £10,734  
        - **Recency (mean):** 396 hari
        """)

        st.subheader("Preferensi Produk")
        st.markdown("""
        - **Rata-rata item per transaksi:** 542.8  
        - **Top 3 produk:**  
            1. JUMBO BAG RED RETROSPOT  
            2. ASSORTED COLOUR BIRD ORNAMENT  
            3. WHITE HANGING HEART T-LIGHT HOLDER  
        - **Kombinasi populer:**  
            - ROSES REGENCY TEACUP + GREEN REGENCY TEACUP  
            - GARDENERS KNEELING PAD (CUP OF TEA + KEEP CALM)  
            - ALARM CLOCK BAKELITE (RED + GREEN)  
            - LUNCH BAG PINK POLKADOT + BLACK SKULL  
            - JUMBO BAG RED RETROSPOT + PINK POLKADOT  
        """)

        st.subheader("Interpretasi & Strategi Bisnis")
        st.markdown("""
        - Cluster 2 terdiri dari pelanggan yang cukup sering berbelanja (rata-rata 8 transaksi per pelanggan) dengan nilai transaksi yang signifikan (Monetary rata-rata £10.734) dan aktivitas belanja yang masih terbilang aktif (Recency rata-rata 396 hari). Mereka merupakan pelanggan loyal dengan potensi untuk ditingkatkan.
        - Ukuran basket besar menunjukkan mereka membeli beberapa item sekaligus, cocok untuk strategi bundling.  

        **Strategi:**
        1. Bundling & Upsell: kombinasi produk populer seperti daftar kombinasi diatas untuk mendorong penjualan multi-item.
        2. Loyalty Program: penghargaan atau diskon khusus bagi pelanggan setia untuk mempertahankan retensi.  
        3. Promo targeted: diskon atau voucher khusus untuk bundle populer agar meningkatkan frekuensi dan nilai transaksi.
        """)

    # =======================
    # CLUSTER 3
    # =======================
    with st.expander("🟩 Cluster 3 — Top Customers"):
        st.subheader("Profil RFM")
        st.markdown("""
        - **Frequency (mean):** 14.98  
        - **Monetary (mean):** £21,210  
        - **Recency (mean):** 1 hari  
        """)

        st.subheader("Preferensi Produk")
        st.markdown("""
        - **Rata-rata item per transaksi:** 496.1  
        - **Top 3 produk:**  
            1. WHITE HANGING HEART T-LIGHT HOLDER  
            2. JUMBO BAG RED RETROSPOT  
            3. ASSORTED COLOUR BIRD ORNAMENT  
        - **Kombinasi populer:**  
            - SUKI DESIGN LUNCH BAG + RED RETROSPOT  
            - JUMBO BAG (PINK POLKADOT + RED RETROSPOT)  
            - JUMBO BAG RED RETROSPOT + STRAWBERRY  
        """)

        st.subheader("Interpretasi & Strategi Bisnis")
        st.markdown("""
        - Cluster 3 terdiri dari pelanggan paling bernilai, yang sering berbelanja (rata-rata hampir 15 transaksi per pelanggan), total belanja tinggi (Monetary rata-rata £21.210), dan baru saja melakukan transaksi (Recency rata-rata 1 hari). Mereka adalah top customers yang sangat aktif dan menjadi prioritas utama untuk retention.
        - Ukuran basket cukup besar menunjukkan mereka membeli beberapa item sekaligus, cocok untuk strategi bundling.

        **Strategi:**
        1. Premium Upsell & Cross-sell: menawarkan produk high-value atau bundle eksklusif sesuai top item.
        2. Exclusive Offers/Early Access: mendorong loyalitas dan pembelian berulang melalui promosi terbatas.  
        3. Program VIP: penghargaan untuk mempertahankan engagement dan meningkatkan lifetime value. 
        """)


