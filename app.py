import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# 網頁標題
st.title("最小平方法互動學習工具")

# 1. 初始化 Session State 來儲存數據點 (確保互動時數據不會重置)
if 'data_x' not in st.session_state:
    st.session_state.data_x = list(np.linspace(0, 10, 5))
    st.session_state.data_y = list(2.5 * np.linspace(0, 10, 5) + np.random.normal(0, 2, 5))

# 2. 互動控制介面
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("新增一個極端離群值"):
        st.session_state.data_x.append(12)
        st.session_state.data_y.append(np.random.uniform(15, 25))

with col2:
    if st.button("移除最後一個離群值"):
        # 確保至少保留 3 個點，避免矩陣計算錯誤
        if len(st.session_state.data_x) > 3:
            st.session_state.data_x.pop()
            st.session_state.data_y.pop()
        else:
            st.warning("至少需要 3 個點才能進行最小平方法運算！")

with col3:
    if st.button("重置數據"):
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
ax.scatter(x, y, color='red', label='Data Points (觀測值)')
x_range = np.linspace(0, 13, 100)

# 使用 np.polyval 根據求出的係數陣列畫出曲線
y_fit = np.polyval(coeffs, x_range)

ax.plot(x_range, y_fit, label=f'{degree} 階擬合曲線', color='blue')
ax.set_ylim(-10, 35) # 稍微放寬 y 軸讓高階曲線顯示更完整
ax.legend()
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
