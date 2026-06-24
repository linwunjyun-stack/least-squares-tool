# 互動功能：新增或移除數據點
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("新增一個極端離群值"):
        st.session_state.data_x.append(12)
        st.session_state.data_y.append(np.random.uniform(15, 25))

with col2:
    # 檢查長度，確保不會刪到剩下太少點（例如至少保留 3 個點）
    if st.button("移除最後一個離群值"):
        if len(st.session_state.data_x) > 3:
            st.session_state.data_x.pop()
            st.session_state.data_y.pop()
        else:
            st.warning("至少需要 3 個點才能進行最小平方法運算！")

with col3:
    if st.button("重置數據"):
        st.session_state.data_x = list(np.linspace(0, 10, 5))
        st.session_state.data_y = list(2.5 * np.linspace(0, 10, 5) + np.random.normal(0, 2, 5))
