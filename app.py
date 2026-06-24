import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("最小平方法互動學習工具")

# 1. 數據輸入
n_points = st.slider("數據點數量", 3, 20, 5)
x = np.linspace(0, 10, n_points)
y = 2.5 * x + np.random.normal(0, 2, n_points) # 模擬觀測數據

# 2. 構建設計矩陣 A (假設是線性擬合 y = ax + b)
A = np.vstack([x, np.ones(len(x))]).T
# 正規方程: A^T * A * x = A^T * b
ATA = A.T @ A
ATb = A.T @ y

# 3. 矩陣運算與反矩陣
# 這裡是作業要求的技術重點：必須實作計算邏輯
coeffs = np.linalg.inv(ATA) @ ATb 

# 4. 視覺化
fig, ax = plt.subplots()
ax.scatter(x, y, color='red', label='Data Points')
ax.plot(x, coeffs[0]*x + coeffs[1], label='Fit: y={:.2f}x+{:.2f}'.format(coeffs[0], coeffs[1]))
st.pyplot(fig)

# 顯示矩陣運算過程 (符合作業對「計算邏輯」的演示要求)
st.write("矩陣 A^T * A:", ATA)
st.write("求解出的係數:", coeffs)