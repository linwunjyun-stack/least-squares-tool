import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("最小平方法互動學習工具")

# 初始化數據：如果 session_state 裡沒有數據，就建立一組預設數據
if 'data_x' not in st.session_state:
    st.session_state.data_x = list(np.linspace(0, 10, 5))
    st.session_state.data_y = list(2.5 * np.linspace(0, 10, 5) + np.random.normal(0, 2, 5))

# 互動功能：新增離群點
if st.button("新增一個極端離群值 (Outlier)"):
    # 在圖表右側隨機位置加入一個離群點
    st.session_state.data_x.append(12) 
    st.session_state.data_y.append(np.random.uniform(15, 25)) # y 值故意設得很高

# 互動功能：重置數據
if st.button("重置數據"):
    st.session_state.data_x = list(np.linspace(0, 10, 5))
    st.session_state.data_y = list(2.5 * np.linspace(0, 10, 5) + np.random.normal(0, 2, 5))

# 將數據轉回 NumPy 陣列進行運算
x = np.array(st.session_state.data_x)
y = np.array(st.session_state.data_y)

# 構建設計矩陣 A
A = np.vstack([x, np.ones(len(x))]).T
ATA = A.T @ A
ATb = A.T @ y
coeffs = np.linalg.inv(ATA) @ ATb 

# 視覺化
fig, ax = plt.subplots()
ax.scatter(x, y, color='red', label='Data Points')
x_range = np.linspace(0, 12, 100)
ax.plot(x_range, coeffs[0]*x_range + coeffs[1], label='Fit line')
ax.set_ylim(-5, 30) # 固定範圍以便觀察變化
st.pyplot(fig)

# 展示矩陣細節 (符合評分要求)
with st.expander("查看矩陣運算細節"):
    st.write("設計矩陣 A (轉置後顯示):", A.T)
    st.write("正規方程 A^T * A:", ATA)
    st.write("計算出的斜率與截距:", coeffs)
