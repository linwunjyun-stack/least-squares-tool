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
# 4. 視覺化區塊 (使用 Plotly 達成滑鼠懸浮互動顯示座標)
# ==========================================
fig = go.Figure()

# 繪製紅色觀測點
fig.add_trace(go.Scatter(
    x=x, y=y,
    mode='markers',
    marker=dict(color='red', size=11),
    name='Data Points',
    hovertemplate='X 座標: %{x}<br>Y 座標: %{y}<extra></extra>' # 游標移上去顯示精準座標
))

# 如果能正常計算反矩陣，才繪製藍色擬合曲線
if coeffs_calculated:
    x_min, x_max = np.min(x), np.max(x)
    x_range = np.linspace(x_min - 2, x_max + 5, 200)
    y_fit = np.polyval(coeffs, x_range)

    fig.add_trace(go.Scatter(
        x=x_range, y=y_fit,
        mode='lines',
        line=dict(color='blue', width=2.5),
        name='擬合曲線',
        hoverinfo='skip' # 曲線上不顯示懸浮框，聚焦在觀測點
    ))

# 動態設定畫面的 Y 軸視野範圍
y_min, y_max = np.min(y), np.max(y)
fig.update_layout(
    yaxis_range=[y_min - 10, y_max + 15],
    hovermode='closest',
    margin=dict(l=0, r=0, t=30, b=0),
    showlegend=False,
    font=dict(size=15, color="black") # 統一放大圖表內的所有標籤字體
)

st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 5. 高階響應式排版：展示計算邏輯 (解決表格擠壓)
# ==========================================
st.write("---")
st.subheader("📊 運算細節與結果 (正規方程邏輯)")

st.markdown("#### 1. $A^T A$ (正規矩陣)")
st.dataframe(pd.DataFrame(ATA), use_container_width=True)

st.markdown("#### 2. $(A^T A)^{-1}$ (反矩陣)")
if coeffs_calculated:
    st.dataframe(pd.DataFrame(np.linalg.inv(ATA)), use_container_width=True)
else:
    st.error("❌ 目前的數據點組合無法計算反矩陣（奇異矩陣），請嘗試新增、刪除或調整數據點座標！")

st.markdown("#### 3. $\hat{x}$ (計算出的係數，由高次項至常數項)")
if coeffs_calculated:
    st.dataframe(pd.DataFrame(coeffs, columns=["係數數值"]), use_container_width=True)
else:
    st.write("無法求出唯一係數。")

st.write("---")

# ==========================================
# 6. 底部指標區 (數據點數與均方誤差 MSE)
# ==========================================
if coeffs_calculated:
    y_pred = np.polyval(coeffs, x)
    mse = np.mean((y - y_pred)**2)
else:
    mse = 0.0

col_metric1, col_metric2 = st.columns(2)
with col_metric1:
    st.metric(label="數據點數", value=f"{len(x)}")
with col_metric2:
    st.metric(label="擬合誤差 (MSE)", value=f"{mse:.4f}")
