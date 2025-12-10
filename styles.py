#########################################
# 1. GLOBAL STYLES (Font, Sidebar, Icons)
#########################################
global_css = """
<style>
    @import url('https://fonts.googleapis.com/icon?family=Material+Icons');
    @import url('https://fonts.googleapis.com/css2?family=Lexend:wght@100..900&family=Outfit:wght@100..900&display=swap');

    /* ubah font custom buat keseluruhan */
    html, body, span, div, p, label, h1, h2, h3, h4, h5, h6, [class*="css"],
    .stMarkdown, .stText, .stMetric, .stDataFrame, .stButton {
        font-family: "Outfit", sans-serif !important;
        color: white;
    }

    .stSidebar .stButton>button {
        background-color: #141414;
        color: white;
        border: none;
        text-align: left;
        font-size: 14px !important; /* Saya koreksi 5px jadi 14px biar terbaca */
    }

    .stSidebar .stButton>button:hover {
        background-color: #aaaaaa;
        color: black;
    }

    .stApp{
        background-color: #292929; 
    }

    [data-testid="stMainBlockContainer"] {
        margin-left: 0px;
        width: calc(100% + 0px);
        margin-right: 0px
    }

    header[data-testid="stHeader"] {
        background-color: #292929 !important;
        visibility: display;
    }

    section[data-testid="stSidebar"] {
        width: 230px !important; 
        background-color: #141414 !important; 
    }

    /* replace icon */
    [data-testid="stIconMaterial"] {
        font-size: 0 !important;
    }

    /* icon custom  */
    [data-testid="stIconMaterial"]::after {
        content: "";
        display: inline-block;
        width: 22px;
        height: 22px;
        /* di svg harus pake kutip 1 */
        background-image: url("data:image/svg+xml;utf8,<svg fill='white' xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'><path d='M5 4l7 8-7 8M13 4l7 8-7 8'/></svg>");
        background-size: contain;
    }

    /* hapus gaps antar kolom */
    .block-container {
        padding-top: 1rem !important;
    }
</style>
"""

#########################################
# 2. COMPONENT STYLES
#########################################
card = """
    {
        background-color: #313234;
        color: white;
        padding: 10px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        min-width: 310px;
        align-items: center;
    }
    /* Target elemen anak (Header h3) */
    h3, h4 {
        color: white !important;
        margin-bottom: 20px !important;
    }
"""
