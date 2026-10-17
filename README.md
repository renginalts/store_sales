# Store Sales - Time Series Forecasting (Kaggle)

Bu proje, Ekvador'daki bir süpermarket zinciri olan **Corporación Favorita**'nın farklı mağazalarındaki ürün satışlarını tahmin etmek amacıyla Kaggle üzerinde gerçekleştirilmiş bir zaman serisi (time series forecasting) projesidir.

## Proje Adımları ve Özellikler
1. **Veri Ön İşleme ve Temizlik:** 
   - `train`, `test`, `stores`, `oil`, `holidays_events` ve `transactions` veri setleri birleştirildi.
   - Petrol fiyatlarındaki (`dcoilwtico`) eksik veriler ileriye ve geriye dönük dolduruldu (`ffill` / `bfill`).
2. **Öznitelik Mühendisliği (Feature Engineering):**
   - **Takvim Özellikleri:** Yıl, ay, gün, haftanın günü, hafta sonu ve çeyrek bilgileri üretildi.
   - **Tatil Özellikleri:** Ulusal tatiller (`is_holiday`) modele eklendi.
   - **Gecikmeli Değişkenler (Lag Features):** 7, 14, 28 günlük geçmiş satışlar (`lag`) ve hareketli ortalamalar (`rolling_mean_7`) hesaplandı.
   - **İşlem Verileri:** Mağaza işlem hacimleri (`transactions`) modele dahil edildi.
3. **Modelleme:**
   - Güçlü bir ağaç tabanlı model olan **LightGBM (LGBMRegressor)** kullanıldı.
   - Zaman serisi yapısına uygun olarak kronolojik doğrulama (validation) stratejisi uygulandı.

## Kullanılan Kütüphaneler
- `pandas`, `numpy`
- `scikit-learn`
- `lightgbm`
