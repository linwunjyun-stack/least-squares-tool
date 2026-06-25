import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# 網頁標題
st.title("最小平方法互動學習工具")

# 1. 初始化 Session State (儲存數據點)
if 'data_x' not in st.session_state:
    st.session_state.data_x = list(np.linspace(0, 10, 5))
    st.session_state.data_y = list(2.5 * np.linspace(0, 10, 5) + np.random.normal(0, 2, 5))

# 2. 互動控制介面 (自訂點位與階數)
st.write("---")
st.subheader("⚙️ 數據點與參數控制")

col1, col2 = st.columns(2)

with col1:
    st.write("**1. 新增自訂數據點**")
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
                st.session_state
