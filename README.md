# Karaciğer Sirozu Evre Tahmin Sistemi

## Gemini API Key Alma (Ücretsiz, Kart Gerekmez)

1. https://aistudio.google.com/app/apikey adresine git
2. Google hesabınla giriş yap
3. "Create API Key" butonuna bas
4. Çıkan key'i kopyala

## Streamlit Cloud'a Deploy

1. Bu 3 dosyayı GitHub'da yeni public repoya yükle:
   - `app.py`
   - `requirements.txt`  
   - `.gitignore`

2. https://share.streamlit.io → "New app" → repoyu seç → Deploy

3. App settings → Secrets → şunu yapıştır:
```toml
GEMINI_API_KEY = "buraya-kendi-key-ini-yaz"
```

## Yerel Çalıştırma

```bash
pip install -r requirements.txt

mkdir .streamlit
echo 'GEMINI_API_KEY = "AIza..."' > .streamlit/secrets.toml

streamlit run app.py
```

## Özellikler
- Histolojik evre tahmini (Evre 1–4)  
- Evre olasılık dağılımı  
- Özellik önemi analizi  
- Anormal değer uyarıları  
- Gemini AI klinik analizi (tamamen ücretsiz)

## Model
Stacking Ensemble: XGBoost + Random Forest + LightGBM → Logistic Regression  
Doğruluk: %89.9 | Veri Seti: Mayo Clinic PBC
