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

# 3. 矩陣運算核心 (正規方程: A^T * A * x = A^T * b)
x = np.array(st.session_state.data_x)
y = np.array(st.session_state.data_y)

# 構建設計矩陣 A (y = ax + b)
A = np.vstack([x, np.ones(len(x))]).T
ATA = A.T @ A
ATb = A.T @ y

# 求解係數
coeffs = np.linalg.inv(ATA) @ ATb 

# 4. 視覺化
fig, ax = plt.subplots()
ax.scatter(x, y, color='red', label='Data Points')
x_range = np.linspace(0, 12, 100)
ax.plot(x_range, coeffs[0]*x_range + coeffs[1], label=f'Fit: y={coeffs[0]:.2f}x + {coeffs[1]:.2f}')
ax.set_ylim(-5, 30)
ax.legend()
st.pyplot(fig)

# 5. 作業要求：展示計算邏輯
with st.expander("查看矩陣運算細節 (正規方程邏輯)"):
    st.write("**設計矩陣 A (Design Matrix):**")
    st.write(A)
    st.write("**正規方程 A^T * A:**")
    st.write(ATA)
    st.write("**計算出的斜率 (a) 與截距 (b):**")
    st.write(f"a = {coeffs[0]:.4f}, b = {coeffs[1]:.4f}")
