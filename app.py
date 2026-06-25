import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# 網頁標題
st.title("最小平方法互動學習工具")

# 使用 CSS 放大所有 st.metric 的字體
st.markdown(
    """
    <style>
    [data-testid="stMetricValue"] {
        font-size: 45px; /* 這裡調整數字的大小 */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 1. 初始化 Session State (儲存數據點)
if 'data_x' not in st.session_state:
    st.session_state.data_x = list(np.linspace(0, 10, 5))
    st.session_state.data_y = list(2.5 * np.linspace(0, 10, 5) + np.random.normal(0, 2, 5))

# 2. 互動控制介面 (自訂點位與階數)
st.write("---")
st.subheader("⚙️ 數據點與參數控制")

col1, col2 = st.columns(2)

with col1:
    st.markdown("<p style='font-size: 20px; font-weight: bold;'>1. 新增自訂數據點</p>", unsafe_allow_html=True)
    sub_col1, sub_col2, sub_col3 = st.columns([1, 1, 1])
    with sub_col1:
        new_x = st.number_input("輸入 X 座標", value=12.0, step=1.0)
    with sub_col2:
        new_y = st.number_input("輸入 Y 座標", value=20.0, step=1.0)
    with sub_col3:
        st.write("")
        st.write("")
        if st.button("➕ 新增點位"):
            st.session_state.data_x.append(new_x)
            st.session_state.data_y.append(new_y)

with col2:
    st.write("**2. 移除與重置**")
    st.write("")
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

degree = st.slider("選擇多項式階數 (1=直線, 2=拋物曲線, 3=S型曲線)", 1, 4, 1)

# 3. 矩陣運算核心 (正規方程)
x = np.array(st.session_state.data_x)
y = np.array(st.session_state.data_y)

A_cols = [x**(d) for d in range(degree, -1, -1)]
A = np.vstack(A_cols).T

ATA = A.T @ A
ATb = A.T @ y
try:
    ATA_inv = np.linalg.inv(ATA)
    coeffs = ATA_inv @ ATb
except np.linalg.LinAlgError:
    st.error("矩陣不可逆！請調整數據點。")
    st.stop()

# 4. 視覺化 (使用 Plotly 達成互動懸浮提示)
fig = go.Figure()

# 加入數據點 (散佈圖)
fig.add_trace(go.Scatter(
    x=x, y=y,
    mode='markers',
    marker=dict(color='red', size=10),
    name='Data Points',
    hovertemplate='X 座標: %{x}<br>Y 座標: %{y}<extra></extra>'
))

# 計算並加入擬合曲線
x_min, x_max = np.min(x), np.max(x)
x_range = np.linspace(x_min - 2, x_max + 5, 200)
y_fit = np.polyval(coeffs, x_range)

fig.add_trace(go.Scatter(
    x=x_range, y=y_fit,
    mode='lines',
    line=dict(color='blue', width=2),
    name='擬合曲線',
    hoverinfo='skip' # 曲線上不顯示懸浮框，避免干擾
))

# 設定圖表版面 (動態範圍、外觀與字體大小)
y_min, y_max = np.min(y), np.max(y)
fig.update_layout(
    yaxis_range=[y_min - 10, y_max + 15],
    hovermode='closest',
    margin=dict(l=0, r=0, t=30, b=0),
    showlegend=False,
    # 加入這行來統一調整圖表內的字體大小 (包含座標軸數字與懸浮提示)
    font=dict(
        size=20,      # 數字越大字體越大，建議設定在 14 到 18 之間
        color="black" # 也可以順便指定字體顏色
    )
)

st.plotly_chart(fig, use_container_width=True)

# 5. 作業要求：高階排版展示計算邏輯 (解決表格擠壓的上下排版)
st.write("---")
st.subheader("📊 運算細節與結果 (正規方程邏輯)")

# 放棄 columns 並排，改為上下排列，並設定 use_container_width=True 釋放寬度
st.markdown("#### 1. $A^T A$ (正規矩陣)")
st.dataframe(pd.DataFrame(ATA), use_container_width=True)

st.markdown("#### 2. $(A^T A)^{-1}$ (反矩陣)")
# 這裡加入 try-except 避免某些極端情況下矩陣無法反轉導致整個網頁崩潰
try:
    inv_ATA = np.linalg.inv(ATA)
    st.dataframe(pd.DataFrame(inv_ATA), use_container_width=True)
except np.linalg.LinAlgError:
    st.error("目前的數據點組合無法計算反矩陣（奇異矩陣），請嘗試新增或調整數據點！")

st.markdown("#### 3. $\hat{x}$ (計算出的係數)")
st.dataframe(pd.DataFrame(coeffs, columns=["係數數值"]), use_container_width=True)

st.write("---")

# 計算 MSE 時確保預測值 y_pred 只針對實際資料點 x 計算
y_pred = np.polyval(coeffs, x)
mse = np.mean((y - y_pred)**2)

# 渲染底部數據指標 (保持並排，因為數字通常不會太長)
col_metric1, col_metric2 = st.columns(2)
with col_metric1:
    st.metric(label="數據點數", value=f"{len(x)}")
with col_metric2:
    st.metric(label="擬合誤差 (MSE)", value=f"{mse:.4f}")
