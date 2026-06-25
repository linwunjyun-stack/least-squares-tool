import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ==========================================
# 0. 全局前端樣式美化 (CSS 魔法)
# ==========================================
st.markdown(
    """
    <style>
    /* 放大所有 st.metric 的數字字體 */
    [data-testid="stMetricValue"] {
        font-size: 40px;
        font-weight: bold;
    }
    /* 自訂大文字樣式，供後續介面標題使用 */
    .custom-label {
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 網頁主標題
st.title("最小平方法互動學習工具")

# ==========================================
# 1. 初始化 Session State (確保互動時數據點不會消失)
# ==========================================
if 'data_x' not in st.session_state:
    st.session_state.data_x = list(np.linspace(0, 10, 5))
    st.session_state.data_y = list(2.5 * np.linspace(0, 10, 5) + np.random.normal(0, 2, 5))

# ==========================================
# 2. 互動控制介面 (自訂點位與重置按鈕)
# ==========================================
st.write("---")
st.subheader("⚙️ 數據點與參數控制")

col1, col2 = st.columns(2)

with col1:
    st.markdown("<p class='custom-label'>1. 新增自訂數據點</p>", unsafe_allow_html=True)
    sub_col1, sub_col2, sub_col3 = st.columns([1, 1, 1])
    with sub_col1:
        new_x = st.number_input("輸入 X 座標", value=12.0, step=1.0)
    with sub_col2:
        new_y = st.number_input("輸入 Y 座標", value=20.0, step=1.0)
    with sub_col3:
        st.write("") # 對齊高度
        st.write("")
        if st.button("➕ 新增點位"):
            st.session_state.data_x.append(new_x)
            st.session_state.data_y.append(new_y)

with col2:
    st.markdown("<p class='custom-label'>2. 移除與重置</p>", unsafe_allow_html=True)
    st.write("") # 對齊高度
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("🗑️ 移除最後一點"):
            if len(st.session_state.data_x) > 3:
                st.session_state.data_x.pop()
                st.session_state.data_y.pop()
            else:
                st.error("至少需要 3 個點才能計算！")
    with btn_col2:
        if st.button("🔄 重置所有數據"):
            st.session_state.data_x = list(np.linspace(0, 10, 5))
            st.session_state.data_y = list(2.5 * np.linspace(0, 10, 5) + np.random.normal(0, 2, 5))

# 讓使用者自由拉動切換多項式階數
degree = st.slider("選擇多項式階數 (1=直線, 2=拋物曲線, 3=S型曲線)", 1, 4, 1)

# ==========================================
# 3. 矩陣運算核心 (正規方程: AᵀA x̂ = Aᵀb)
# ==========================================
x = np.array(st.session_state.data_x)
y = np.array(st.session_state.data_y)

# 動態構建設計矩陣 A (依據所選的 degree)
A_cols = [x**(d) for d in range(degree, -1, -1)]
A = np.vstack(A_cols).T

ATA = A.T @ A
ATb = A.T @ y

# 預設係數矩陣，並透過 try-except 進行工程防呆
coeffs_calculated = False
try:
    coeffs = np.linalg.inv(ATA) @ ATb
    coeffs_calculated = True
except np.linalg.LinAlgError:
    coeffs = np.zeros(degree + 1) # 奇異矩陣時給予零陣列，避免繪圖崩潰

# ==========================================
# 4. 視覺化區
