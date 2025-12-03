import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
import numpy as np

st.title("🚗 Prediksi Harga Mobil – KNN (CarDekho Dataset)")

# LOAD DATASET
@st.cache_data
def load_data():
    df = pd.read_csv("CAR DETAILS FROM CAR DEKHO.csv")
    return df

df = load_data()

# PREPROCESSING
df_processed = df.copy()

# Label Encode kolom kategorikal
label_cols = ["name", "fuel", "seller_type", "transmission", "owner"]
encoders = {}

for col in label_cols:
    enc = LabelEncoder()
    df_processed[col] = enc.fit_transform(df_processed[col])
    encoders[col] = enc


# Fitur & target
X = df_processed.drop("selling_price", axis=1)
y = df_processed["selling_price"]

# Standarisasi
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Moodel KNN
model = KNeighborsRegressor(n_neighbors=5)
model.fit(X_train, y_train)


# Form Input
st.subheader("Form Input Prediksi:")

# Input user
col1, col2 = st.columns(2)

with col1:
    year = st.number_input("Tahun Mobil", min_value=1990, max_value=2025, value=2015)
    km_driven = st.number_input("KM Driven", min_value=100, max_value=500000, value=50000)
    seats = st.number_input("Jumlah Kursi", min_value=2, max_value=10, value=5)
    transmission = st.selectbox("Transmisi", df["transmission"].unique())

with col2:
    name = st.selectbox("Nama / Model Mobil", df["name"].unique())
    fuel = st.selectbox("Jenis Bahan Bakar", df["fuel"].unique())
    seller_type = st.selectbox("Tipe Penjual", df["seller_type"].unique())
    owner = st.selectbox("Status Kepemilikan", df["owner"].unique())

# Encode input
input_data = pd.DataFrame({
    "name": [encoders["name"].transform([name])[0]],
    "year": [year],
    "selling_price": [0],
    "km_driven": [km_driven],
    "fuel": [encoders["fuel"].transform([fuel])[0]],
    "seller_type": [encoders["seller_type"].transform([seller_type])[0]],
    "transmission": [encoders["transmission"].transform([transmission])[0]],
    "owner": [encoders["owner"].transform([owner])[0]],
    "seats": [seats]
})

# Hapus kolom selling_price
input_data = input_data.drop("selling_price", axis=1)

input_data = input_data[X.columns]

# Scaling input
input_scaled = scaler.transform(input_data)

KURS_INR_TO_IDR = 190 # contoh rata-rata 1 INR ≈ 190 IDR

# Prediksi
if st.button("Prediksi Harga"):
    pred = model.predict(input_scaled)[0]
    pred_idr = pred * KURS_INR_TO_IDR
    
    st.subheader("Hasil Prediksi Harga")
    st.success(f"Perkiraan harga mobil: Rp {pred_idr:,.0f}")
