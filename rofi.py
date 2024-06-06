import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Pengumpulan Data
data = pd.read_csv('data_penjualan.csv')

# 2. Data Cleaning
print(data.info())
data.drop_duplicates(inplace=True)
data.dropna(inplace=True)

# 3. Data Transformation
data['Tanggal'] = pd.to_datetime(data['Tanggal'])
data['Total Penjualan'] = data['Jumlah'] * data['Harga Satuan']

# 4. Exploratory Data Analysis (EDA)
print(data.describe())

# Diagram Batang: Total Penjualan per Jenis Barang
plt.figure(figsize=(10, 6))
sns.barplot(x='Jenis Barang', y='Total Penjualan', data=data, estimator=sum)
plt.title('Total Penjualan per Jenis Barang')
plt.show()

# Diagram Lingkaran: Distribusi Penjualan Berdasarkan Jenis Kelamin
plt.figure(figsize=(6, 6))
data['Jenis Kelamin'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightcoral'])
plt.title('Distribusi Penjualan Berdasarkan Jenis Kelamin')
plt.ylabel('')
plt.show()

# Diagram Garis: Tren Penjualan dari Waktu ke Waktu
plt.figure(figsize=(10, 6))
data.groupby('Tanggal')['Total Penjualan'].sum().plot()
plt.title('Tren Penjualan dari Waktu ke Waktu')
plt.xlabel('Tanggal')
plt.ylabel('Total Penjualan')
plt.show()

# Plot Korelasi
plt.figure(figsize=(8, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Korelasi Antar Variabel')
plt.show()

# Heatmap dari plot korelasi
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Korelasi Antar Variabel')
plt.show()

# 5. Modelling Data
X = data[['Jumlah', 'Harga Satuan']]
y = data['Total Penjualan']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 6. Validasi dan Tuning Model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# 7. Interpretasi dan Penyajian Hasil
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(results.head())

# 8. Deploy dan Monitoring
# (Simulasi) Deploy model ke lingkungan produksi dan monitoring performa

# 9. Maintenance dan Iterasi
# (Simulasi) Melakukan pemeliharaan dan iterasi berdasarkan feedback
