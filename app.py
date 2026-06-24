import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 【關鍵升級 1】開啟寬螢幕模式，確保三欄有足夠的顯示空間 (這行必須在最上面)
st.set_page_config(page_title="最小平方法互動學習工具", layout="wide")

st.title("最小平方法互動學習工具")
# --- 自訂 CSS 來全面放大字體與元件 ---
st.markdown("""
<style>
    /* 1. 上方主標題 (st.title) 放大 */
    h1 {
        font-size: 55px !important;
        font-weight: 800 !important;
        padding-bottom: 40px !important;
    }
    
    /* 2. 區塊小標題 (st.subheader) 放大 */
    h3 {
        font-size: 45px !important;
        color: #333333 !important;
    }
    
    /* 3. 輸入框標籤、拉桿標籤與一般文字放大 */
    p, label {
        font-size: 40px !important;
    }
    .stSlider div[data-testid="stThumbValue"] {
        font-size: 40px !important; /* 拉桿上浮動的數字 */
    }
    
    /* 4. 圖表下方的數據指標 (st.metric) 放大 */
    [data-testid="stMetricLabel"] p {
        font-size: 40px !important; /* 數據點數、MSE 的標題 */
        color: #555555 !important;
    }
    [data-testid="stMetricValue"] div {
        font-size: 48px !important; /* 具體的數字大小 */
        font-weight: 900 !important;
    }
    
    /* 5. 右側 DataFrame 表格整體放大 */
    /* 因為 Streamlit 的表格是畫布(Canvas)渲染，用 zoom 屬性放大最有效 */
    [data-testid="stDataFrame"] {
        zoom: 1.35; 
    }
    
    /* 放大按鈕內的文字 */
    .stButton button p {
        font-size: 20px !important;
        font-weight: bold !important;
    }
</style>
""", unsafe_allow_html=True)
# ---------------------------
st.write("---")

# 初始化 Session State (儲存數據點)
if 'data_x' not in st.session_state:
    st.session_state.data_x = list(np.linspace(0, 10, 5))
    st.session_state.data_y = list(2.5 * np.linspace(0, 10, 5) + np.random.normal(0, 2, 5))

# 【關鍵升級 2】建立三個主欄位，並設定寬度比例 (左1.2 : 中2.2 : 右1.5)
col_left, col_mid, col_right = st.columns([1.2,1,1.2], gap="large")

# ==========================================
# 區塊 1：左方 (數據與參數控制區)
# ==========================================
with col_left:
    st.subheader("⚙️ 數據與參數控制區")
    
    st.markdown("**1. 新增自訂數據點**")
    # 在窄欄位中，將輸入框改為垂直排列會更好看
    new_x = st.number_input("輸入 X 座標", value=12.0, step=1.0)
    new_y = st.number_input("輸入 Y 座標", value=20.0, step=1.0)
    if st.button("➕ 新增點位", use_container_width=True):
        st.session_state.data_x.append(new_x)
        st.session_state.data_y.append(new_y)

    st.write("") # 增加一點垂直間距
    st.markdown("**2. 多項式參數調整**")
    degree = st.slider("選擇多項式階數", 1, 4, 1, help="1=直線, 2=拋物曲線, 3=S型曲線")

    st.write("")
    st.markdown("**3. 移除與重置**")
    if st.button("🗑️ 移除最後一點", use_container_width=True):
        if len(st.session_state.data_x) > 3:
            st.session_state.data_x.pop()
            st.session_state.data_y.pop()
        else:
            st.error("至少需要 3 個點才能計算！")
            
    if st.button("🔄 重置所有數據", use_container_width=True):
        st.session_state.data_x = list(np.linspace(0, 10, 5))
        st.session_state.data_y = list(2.5 * np.linspace(0, 10, 5) + np.random.normal(0, 2, 5))

# --- 先在背景執行矩陣運算，以便中、右欄位能直接讀取結果 ---
x = np.array(st.session_state.data_x)
y = np.array(st.session_state.data_y)

A_cols = [x**(d) for d in range(degree, -1, -1)]
A = np.vstack(A_cols).T

ATA = A.T @ A
ATb = A.T @ y
coeffs = np.linalg.inv(ATA) @ ATb

# ==========================================
# 區塊 2：中間 (動態視覺化圖表)
# ==========================================
with col_mid:
    st.subheader("📈 動態視覺化圖表")
    
    fig, ax = plt.subplots(figsize=(3, 3)) # 調整長寬比以適應欄位
    ax.scatter(x, y, color='red')

    x_min, x_max = np.min(x), np.max(x)
    x_range = np.linspace(x_min - 2, x_max + 5, 200)
    y_fit = np.polyval(coeffs, x_range)
    ax.plot(x_range, y_fit, color='blue')

    y_min, y_max = np.min(y), np.max(y)
    ax.set_ylim(y_min - 10, y_max + 15)
    st.pyplot(fig)
    
    # 將數據指標移到圖表正下方，對應視覺化結果
    y_pred = np.polyval(coeffs, x)
    mse = np.mean((y - y_pred)**2)
    
    col_metric1, col_metric2 = st.columns(2)
    with col_metric1:
        st.metric(label="目前數據點總數", value=f"{len(x)}")
    with col_metric2:
        st.metric(label="擬合誤差 (MSE)", value=f"{mse:.4f}")

# ==========================================
# 區塊 3：右方 (底層矩陣運算解析區)
# ==========================================
with col_right:
    st.subheader("📊 底層矩陣運算解析區")
    
    table_data = []

    # 建立矩陣 A (XᵀX)
    for i in range(len(ATA)):
        step_name = "矩陣 A (XᵀX)" if i == 0 else ""
        row_str = "[ " + " , ".join([f"{val:.2f}" for val in ATA[i]]) + " ]"
        table_data.append([step_name, row_str, "正規矩陣"])

    # 建立向量 B (Xᵀy)
    vec_str = "[ " + " , ".join([f"{val:.2f}" for val in ATb]) + " ]ᵀ"
    table_data.append(["向量 B (Xᵀy)", vec_str, "常數項"])

    # 建立回歸方程式
    eq_terms = []
    for i, c in enumerate(coeffs):
        power = degree - i
        if power == 0:
            eq_terms.append(f"{c:+.2f}")
        elif power == 1:
            eq_terms.append(f"{c:+.2f}x")
        else:
            eq_terms.append(f"{c:+.2f}x^{power}")
    eq_str = "f(x) = " + " ".join(eq_terms).lstrip("+ ")
    table_data.append(["回歸方程式", eq_str, "擬合結果"])

    # --- 終極客製化表格 ---
    df = pd.DataFrame(table_data, columns=["運算步驟", "矩陣數值 / 內容", "類別"])

    custom_table_css = """
    <style>
        .academic-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 22px !important; 
            color: #333333;
        }
        .academic-table th {
            background-color: #f8f9fa;
            font-weight: 800;
            padding: 30px 32px; 
            border-bottom: 15px solid #cccccc;
            text-align: left;
        }
        .academic-table td {
            padding: 30px 32px !important; 
            border-bottom:10px solid #eeeeee;
            vertical-align: middle;
        }
        .academic-table th:nth-child(1), .academic-table td:nth-child(1) { width: 10%; }
        .academic-table th:nth-child(2), .academic-table td:nth-child(2) { width: 20%; }
        .academic-table th:nth-child(3), .academic-table td:nth-child(3) { width: 9%; }
    </style>
    """
    
    st.markdown(custom_table_css, unsafe_allow_html=True)
    st.markdown(df.to_html(index=False, classes="academic-table", escape=False), unsafe_allow_html=True)
