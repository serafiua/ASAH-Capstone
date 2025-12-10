import streamlit as st
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules

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
# 7. MARKET BASKET ANALYSIS (MBA) GLOBAL
#########################################
@st.cache_resource
def data_mba(df):
    df_mba = df.copy()
    df_mba['TransactionId'] = df_mba['TransactionId'].astype(str)
    df_mba['ItemDescription'] = df_mba['ItemDescription'].astype(str).str.strip()
    df_mba = df_mba[df_mba['ItemDescription'] != '']
    return df_mba

@st.cache_resource
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

@st.cache_resource
def run_mba(basket):
    frequent = apriori(basket, min_support=0.02, use_colnames=True)
    rules = association_rules(frequent, metric='lift', min_threshold=1)
    rules = rules[(rules['lift'] > 1.2) & (rules['confidence'] > 0.3)]
    rules = rules.sort_values('lift', ascending=False).reset_index(drop=True)
    return rules

#########################################
# 9. SIDEBAR
#########################################
if "page" not in st.session_state:
    st.session_state.page = "Market Basket Analysis"

def set_page(x):
    st.session_state.page = x

st.sidebar.title("📌 Navigation")
st.sidebar.button("Market Basket Analysis", on_click=set_page, args=("Market Basket Analysis",))

page = st.session_state.page

#########################################
# PAGE: MBA
#########################################
if page == "Market Basket Analysis":
    st.title("Market Basket Analysis (MBA)")

    df_mba = data_mba(df)
    basket = build_basket(df_mba)
    rules = run_mba(basket)

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

    st.subheader("MBA Global Product Combos")
    selected_combo = st.selectbox("Pilih combo produk", rules_unique['combo'].tolist())

    st.markdown("---")

    info = combo_dict[selected_combo]
    ante, cons = info["ante"], info["cons"]

    st.write(f"**Jika pelanggan membeli:** `{ante}` → **Mereka mungkin juga membeli:** `{cons}`")

    col1, col2, col3 = st.columns(3)
    col1.metric("Support", f"{info['support']*100:.2f}%")
    col2.metric("Confidence", f"{info['confidence']*100:.2f}%")
    col3.metric("Lift", f"{info['lift']:.2f}")

    customers = (
        df_mba[df_mba["ItemDescription"] == ante]["UserId"]
        .dropna()
        .unique()
        .tolist()
    )

    st.subheader(f"Pelanggan Potensial untuk `{ante}`")
    st.write(f"**{len(customers)} pelanggan ditemukan**")

    def to_grid(data, cols=10):
        rows = int(np.ceil(len(data) / cols))
        grid = np.array(data + [""] * (rows*cols - len(data)))
        grid = grid.reshape(rows, cols)
        return pd.DataFrame(grid, columns=[f"Kolom {i+1}" for i in range(cols)])

    customer_grid = to_grid(customers, cols=10)
    st.dataframe(customer_grid)

    st.markdown("---")

    with st.expander("Seluruh Aturan MBA (Diurutkan berdasarkan Lift)"):
        st.dataframe(rules_unique[['ante','cons','support','confidence','lift']])
