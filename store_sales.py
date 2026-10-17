import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

# 1. VERİLERİ YÜKLEME
print("Veriler yükleniyor...")
train = pd.read_csv(r'C:\Users\alata\Desktop\store-sales-time-series-forecasting\train.csv')
test = pd.read_csv(r'C:\Users\alata\Desktop\store-sales-time-series-forecasting\test.csv')
stores = pd.read_csv(r'C:\Users\alata\Desktop\store-sales-time-series-forecasting\stores.csv')
oil = pd.read_csv(r'C:\Users\alata\Desktop\store-sales-time-series-forecasting\oil.csv')
holidays = pd.read_csv(r'C:\Users\alata\Desktop\store-sales-time-series-forecasting\holidays_events.csv')
transactions = pd.read_csv(r'C:\Users\alata\Desktop\store-sales-time-series-forecasting\transactions.csv')

# 2. TEMEL TEMİZLİK VE DÖNÜŞÜMLER
train["date"] = pd.to_datetime(train["date"])
test["date"] = pd.to_datetime(test["date"])
oil["date"] = pd.to_datetime(oil["date"])
holidays['date'] = pd.to_datetime(holidays['date'])
transactions['date'] = pd.to_datetime(transactions['date'])

# Petrol fiyatlarındaki eksik değerleri doldurma
oil["dcoilwtico"] = oil["dcoilwtico"].ffill().bfill()

# 3. TATİLLERİ HAZIRLAMA
holidays['is_holiday'] = 1
holidays_national = holidays[holidays['locale'] == 'National'][['date', 'is_holiday']].drop_duplicates()

# 4. LAG (GECİKMELİ) VE ROLLING ÖZELLİKLERİNİN OLUŞTURULMASI
print("Lag özellikleri hesaplanıyor...")
train['is_test'] = 0
test['sales'] = np.nan
test['is_test'] = 1

df_all = pd.concat([train, test], ignore_index=True)
df_all = df_all.sort_values(['store_nbr', 'family', 'date']).reset_index(drop=True)

# Gecikmeli değişkenler (Lag features)
df_all['lag_7'] = df_all.groupby(['store_nbr', 'family'])['sales'].shift(7)
df_all['lag_14'] = df_all.groupby(['store_nbr', 'family'])['sales'].shift(14)
df_all['lag_28'] = df_all.groupby(['store_nbr', 'family'])['sales'].shift(28)

# Hareketli ortalama (Rolling Mean)
df_all['rolling_mean_7'] = df_all.groupby(['store_nbr', 'family'])['sales'].shift(1).rolling(7).mean()

# Tekrar train ve test olarak ayırma
train = df_all[df_all['is_test'] == 0].drop(columns=['is_test']).copy()
test = df_all[df_all['is_test'] == 1].drop(columns=['is_test', 'sales']).copy()

# 5. DİĞER VERİLERİ BİRLEŞTİRME (MERGE)
train = train.merge(stores, on="store_nbr", how="left")
train = train.merge(oil, on="date", how="left")
train = train.merge(holidays_national, on="date", how="left")
# Transactions sadece train setinde olduğu için sol birleştirme yapıyoruz
train = train.merge(transactions, on=["store_nbr", "date"], how="left")

test = test.merge(stores, on="store_nbr", how="left")
test = test.merge(oil, on="date", how="left")
test = test.merge(holidays_national, on="date", how="left")
# Test setinde transactions olmadığı için sütunu NaN olarak bırakıyoruz
test['transactions'] = np.nan

# Eksik verileri doldurma
train['is_holiday'] = train['is_holiday'].fillna(0).astype(int)
test['is_holiday'] = test['is_holiday'].fillna(0).astype(int)

train['lag_7'] = train['lag_7'].fillna(0)
train['lag_14'] = train['lag_14'].fillna(0)
train['lag_28'] = train['lag_28'].fillna(0)
train['rolling_mean_7'] = train['rolling_mean_7'].fillna(0)

test['lag_7'] = test['lag_7'].fillna(0)
test['lag_14'] = test['lag_14'].fillna(0)
test['lag_28'] = test['lag_28'].fillna(0)
test['rolling_mean_7'] = test['rolling_mean_7'].fillna(0)

# 6. ÖZNİTELİK MÜHENDİSLİĞİ - TARİH ÖZELLİKLERİ
def create_date_features(df):
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["dayofweek"] = df["date"].dt.dayofweek
    df["weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    df["quarter"] = df["date"].dt.quarter
    return df

train = create_date_features(train)
test = create_date_features(test)

# 7. KATEGORİK DEĞİŞKENLERİN DÖNÜŞTÜRÜLMESİ (LABEL ENCODING)
cat_cols = ['family', 'city', 'type', 'state']

for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([train[col], test[col]], axis=0).astype(str)
    le.fit(combined)
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

# 8. ZAMAN SERİSİ SIRALAMASI VE EĞİTİM / DOĞRULAMA AYRIMI
train_sorted = train.sort_values('date')
split_idx = int(len(train_sorted) * 0.9)

# 'transactions' sütununu da özellikler (features) listemize ekliyoruz
features = ['store_nbr', 'family', 'onpromotion', 'dcoilwtico',
            'year', 'month', 'day', 'dayofweek', 'weekend', 'quarter',
            'city', 'type', 'cluster', 'is_holiday',
            'lag_7', 'lag_14', 'lag_28', 'rolling_mean_7', 'transactions']

target = 'sales'

X_train = train_sorted[features].iloc[:split_idx]
y_train = train_sorted[target].iloc[:split_idx]

X_val = train_sorted[features].iloc[split_idx:]
y_val = train_sorted[target].iloc[split_idx:]

# 9. MODEL EĞİTİMİ (LightGBM)
print("Model eğitiliyor...")
model = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    random_state=42
)

model.fit(
    X_train, y_train,
    eval_X=X_val,
    eval_y=y_val,
    eval_metric='rmse',
    callbacks=[lgb.early_stopping(stopping_rounds=50)]
)

# 10. TEST SETİ İÇİN TAHMİN VE SUBMISSION OLUŞTURMA
print("Tahminler yapılıyor ve submission dosyası oluşturuluyor...")
X_test = test[features]
test['sales'] = model.predict(X_test)

# Negatif satış tahminlerini 0'a eşitleme
test['sales'] = test['sales'].clip(lower=0)

submission = test[['id', 'sales']]
submission.to_csv(r'C:\Users\alata\Desktop\store-sales-time-series-forecasting\submission.csv', index=False)

print("İşlem tamam! 'submission.csv' başarıyla oluşturuldu.")