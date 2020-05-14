import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

SEED = 42

st.title('My First ML App')

split_type = st.sidebar.radio(  # or selectbox
    'テストデータのとり方',
    ('下から', 'ランダム')
)

test_size = st.sidebar.slider(
    'テストデータの割合（％）',
    0, 100, 20, format="%d"
)

st.write("(i) Upload file")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write(data.head())

    st.write("(ii) Choose target column")
    col_target = st.selectbox(
        'Which column ',
        data.columns)
    col_features = [e for e in data.columns if e != col_target]

    if st.button('(iii) Start training'):
        data_tr, data_ts = train_test_split(
            data,
            shuffle=split_type == "ランダム",
            random_state=SEED,
            test_size=test_size)
        clf = RandomForestClassifier()
        clf.fit(data_tr[col_features], data_tr[col_target])
        st.write(clf.score(data_ts[col_features], data_ts[col_target]))