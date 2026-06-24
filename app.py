import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# 網頁標題
st.title("最小平方法互動學習工具")

# 1. 初始化 Session State 來儲存數據點 (確保互動時數據不會重置)
if 'data_x' not in st.session_state:
    st.session_state.data_x = list(np.linspace(0, 10, 5))
    st.session_state.data_y = list(2.5 * np.linspace(0, 10, 5) + np.random.normal(0, 2, 5))

# 2. 互動控制介面 (升級版：支援自訂點位輸入)
st.write("---")
st.subheader("⚙️ 數據點控制")

col1, col2 = st.columns(2)

with col1:
    st.write("**1. 新增自訂數據點**")
    # 把輸入框和按鈕排在同一行
    sub_col1, sub_col2, sub_col3 = st.columns([1, 1, 1])
    with sub_col1:
        new_x = st.number_input("輸入 X 座標", value=12.0, step=1.0)
    with sub_col2:
        new_y = st.number_input("輸入 Y 座標", value=20.0, step=1.0)
    with sub_col3:
        st.write("") # 用來對齊按鈕高度的空白
        st.write("")
        if st.button("➕ 新增點位"):
            st.session_state.data_x.append(new_x)
            st.session_state.data_y.append(new_y)

with col2:
    st.write("**2. 移除與重置**")
    st.write("") # 對齊高度
    
    # 建立兩個並排的按鈕
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

# --- 請將這段加在「重置數據」按鈕的下方 ---

# 2.5 互動功能：多項式階數調整 (符合作業要求)
st.write("---")
degree = st.slider("選擇多項式階數 (1=直線, 2=拋物曲線, 3=S型曲線)", 1, 4, 1)

# 3. 矩陣運算核心 (正規方程: A^T * A * x = A^T * b)
x = np.array(st.session_state.data_x)
y = np.array(st.session_state.data_y)

# 動態構建設計矩陣 A
# 如果 degree=1，矩陣是 [x, 1]
# 如果 degree=2，矩陣是 [x^2, x, 1]，以此類推
A_cols = [x**(d) for d in range(degree, -1, -1)]
A = np.vstack(A_cols).T

ATA = A.T @ A
ATb = A.T @ y

# 求解係數
coeffs = np.linalg.inv(ATA) @ ATb 

# 4. 視覺化
fig, ax = plt.subplots()

# 移除 label，並畫出數據點
ax.scatter(x, y, color='red') 

# 【修正 1：動態 X 軸範圍】
# 找出目前所有 X 數據的最大值與最小值，並在左右各加一點寬度(例如寬度 2 和 5)
x_min = np.min(x)
x_max = np.max(x)
x_range = np.linspace(x_min - 2, x_max + 5, 200) # 切成 200 個點讓曲線更平滑

# 計算並畫出曲線 (同樣移除 label)
y_fit = np.polyval(coeffs, x_range)
ax.plot(x_range, y_fit, color='blue')

# 【修正 2：動態 Y 軸範圍】
# 為了避免高階曲線數值暴增導致畫面比例失衡，我們也讓 Y 軸動態對齊數據點
y_min = np.min(y)
y_max = np.max(y)
ax.set_ylim(y_min - 10, y_max + 15) 

# 【修正 3：移除圖例】
# 已經刪除 ax.legend()，右上角圖示不會再出現

st.pyplot(fig)
# 5. 作業要求：展示計算邏輯
with st.expander("查看矩陣運算細節 (正規方程邏輯)"):
    st.write(f"**目前選擇 {degree} 階多項式，設計矩陣 A 的維度為 {A.shape}**")
    st.write("**設計矩陣 A (Design Matrix):**")
    st.dataframe(A) # 用 dataframe 顯示矩陣會更整齊
    st.write("**正規方程 A^T * A:**")
    st.dataframe(ATA)
    st.write("**計算出的係數 (從高次項到常數項):**")
    st.write(coeffs)
