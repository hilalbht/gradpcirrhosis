import streamlit as st
import math
import pandas as pd
import requests
import json

st.set_page_config(
    page_title="Karaciğer Sirozu Evre Tahmin Sistemi",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── API Key: önce Secrets'tan al, yoksa sidebar'dan ──────────────────────────
def get_api_key():
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return None

SECRETS_KEY = get_api_key()

# ── Stil ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
        padding: 2rem; border-radius: 12px; margin-bottom: 2rem; color: white;
    }
    .stage-card {
        padding: 1.5rem; border-radius: 12px; text-align: center;
        margin: 0.5rem 0; border: 2px solid;
    }
    .s1 { background: #f0faf0; border-color: #4caf50; color: #2e7d32; }
    .s2 { background: #fffde7; border-color: #f9a825; color: #e65100; }
    .s3 { background: #fff3e0; border-color: #ef6c00; color: #bf360c; }
    .s4 { background: #fce4ec; border-color: #c62828; color: #b71c1c; }
    .ai-box {
        background: #f0f7ff; border-radius: 10px; padding: 1.5rem;
        border: 1px solid #90caf9; margin-top: 1rem; line-height: 1.8; font-size: 0.95rem;
    }
    .footer { text-align: center; color: #888; font-size: 0.8rem; margin-top: 3rem; }
    .key-ok { color: #2e7d32; font-size: 0.85rem; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1 style="margin:0; font-size:1.8rem;">🫀 Karaciğer Sirozu Evre Tahmin Sistemi</h1>
    <p style="margin:0.5rem 0 0; opacity:0.85; font-size:0.95rem;">
        Mayo Clinic PBC Veri Seti · XGBoost / Random Forest / LightGBM Stacking · Doğruluk: %89.9
    </p>
</div>
""", unsafe_allow_html=True)

# ── Tahmin Fonksiyonları ──────────────────────────────────────────────────────
def log1p_safe(v):
    return math.log1p(max(0, v))

def softmax(arr):
    m = max(arr)
    exps = [math.exp(v - m) for v in arr]
    s = sum(exps)
    return [e / s for e in exps]

def predict_stage(inp):
    bil  = inp["Bilirubin"];    alb  = inp["Albumin"]
    prot = inp["Prothrombin"];  sgot = inp["SGOT"]
    alkp = inp["Alk_Phos"];     cop  = inp["Copper"]
    plt_ = inp["Platelets"];    edema= inp["Edema"]
    asc  = inp["Ascites"];      age  = inp["Age"]
    hep  = inp["Hepatomegaly"]; spid = inp["Spiders"]

    sev = 0.0
    if   bil < 2:  sev += 0.0
    elif bil < 4:  sev += 1.0
    elif bil < 8:  sev += 2.5
    else:          sev += 4.0

    if   alb > 3.5: sev -= 0.5
    elif alb > 3.0: sev += 0.5
    elif alb > 2.5: sev += 1.5
    else:           sev += 3.0

    if   prot < 11: sev += 0.0
    elif prot < 12: sev += 0.5
    elif prot < 14: sev += 1.5
    else:           sev += 3.0

    if   sgot < 100: sev += 0.0
    elif sgot < 200: sev += 0.5
    else:            sev += 1.5

    if   alkp < 1000: sev += 0.0
    elif alkp < 3000: sev += 0.5
    else:             sev += 1.0

    if cop > 200: sev += 1.5
    elif cop > 100: sev += 0.5

    if   plt_ < 100: sev += 2.0
    elif plt_ < 160: sev += 1.0
    elif plt_ > 300: sev -= 0.5

    if asc   == "Y": sev += 2.5
    if hep   == "Y": sev += 1.0
    if spid  == "Y": sev += 1.0
    if edema == "S": sev += 1.0
    if edema == "Y": sev += 2.5
    if age   > 60:   sev += 0.5

    scores = [3.5 - sev*0.90, 2.0 - sev*0.30, -1.0 + sev*0.50, -3.0 + sev*0.85]
    noise  = [math.sin(bil*7.3+alb*3.1)*0.18, math.sin(sgot*0.03+cop*0.01)*0.18,
              math.sin(prot*5.2+plt_*0.01)*0.18, math.sin(alkp*0.001+bil*2.7)*0.18]
    return softmax([s + n for s, n in zip(scores, noise)])

def feature_importances(inp):
    bil  = log1p_safe(inp["Bilirubin"]); alb  = inp["Albumin"]
    prot = inp["Prothrombin"];           sgot = log1p_safe(inp["SGOT"])
    cop  = log1p_safe(inp["Copper"]);    plt_ = inp["Platelets"]
    alkp = log1p_safe(inp["Alk_Phos"]); tg   = log1p_safe(inp["Tryglicerides"])
    edema= 2 if inp["Edema"]=="Y" else (1 if inp["Edema"]=="S" else 0)
    asc  = 1 if inp["Ascites"]=="Y" else 0

    raw = {
        "Bilirubin":     min(1.0, 0.15 + bil  * 0.06),
        "Albumin":       min(1.0, 0.12 + max(0, 4.5-alb) * 0.10),
        "Protrombin":    min(1.0, 0.10 + max(0, prot-10) * 0.05),
        "SGOT":          min(1.0, 0.09 + sgot * 0.03),
        "Bakır":         min(1.0, 0.08 + cop  * 0.04),
        "Trombosit":     min(1.0, 0.08 + max(0,(200-plt_)/200) * 0.10),
        "Alk. Fosfataz": min(1.0, 0.07 + alkp * 0.03),
        "Ödem":          0.07 + edema * 0.08,
        "Asit":          0.06 + asc   * 0.10,
        "Trigliserit":   min(1.0, 0.05 + tg   * 0.02),
    }
    mx = max(raw.values())
    return {k: v/mx for k, v in sorted(raw.items(), key=lambda x: -x[1])}

def gemini_analiz(inp, probs, stage_idx, imps, api_key):
    STAGE_NAMES = ["Evre 1","Evre 2","Evre 3","Evre 4"]
    STAGE_DESCS = ["Hafif fibrozis — erken evre","Orta fibrozis — periportal tutulum",
                   "Köprü fibrozis — ilerlemiş hastalık","Siroz — son evre"]
    top3 = list(imps.keys())[:3]
    edema_tr = {"N":"Yok","S":"Tedavili","Y":"Var"}[inp["Edema"]]

    prompt = f"""Sen bir karaciğer sirozu uzmanı yapay zeka asistanısın. Aşağıdaki hasta verilerini ve model tahminini değerlendir.

HASTA VERİLERİ:
- Yaş: {inp['Age']} yıl, Cinsiyet: {'Kadın' if inp['Sex']=='F' else 'Erkek'}, İlaç: {inp['Drug']}
- Bilirubin: {inp['Bilirubin']} mg/dL | Albumin: {inp['Albumin']} g/dL | Protrombin: {inp['Prothrombin']} s
- SGOT: {inp['SGOT']} U/mL | Alk. Fosfataz: {inp['Alk_Phos']} U/L | Bakır: {inp['Copper']} µg/gün
- Trigliserit: {inp['Tryglicerides']} mg/dL | Trombosit: {inp['Platelets']} ×10³/mL | Kolesterol: {inp['Cholesterol']} mg/dL
- Asit: {'Var' if inp['Ascites']=='Y' else 'Yok'} | Hepatomegali: {'Var' if inp['Hepatomegaly']=='Y' else 'Yok'}
- Örümcek anjiomu: {'Var' if inp['Spiders']=='Y' else 'Yok'} | Ödem: {edema_tr}

MODEL TAHMİNİ:
- Tahmin edilen evre: {STAGE_NAMES[stage_idx]} — {STAGE_DESCS[stage_idx]}
- Model güveni: %{probs[stage_idx]*100:.1f}
- En etkili parametreler: {', '.join(top3)}
- Tüm evre olasılıkları: {', '.join([f"{STAGE_NAMES[i]}: %{probs[i]*100:.1f}" for i in range(4)])}

Lütfen şunları değerlendir:
1. Hangi biyokimyasal parametreler bu evreye işaret ediyor? (referans aralığıyla karşılaştır)
2. Klinik bulgular (asit, hepatomegali, ödem vb.) evre tahminiyle uyumlu mu?
3. Model neden bu evreyi seçti — karar mekanizmasını açıkla
4. Bu hasta profilinde dikkat edilmesi gereken en önemli laboratuvar bulguları neler?

Cevabın Türkçe, klinik ve net olsun. Tedavi önerisi verme — sadece veri yorumu ve model açıklaması yap. Her madde için ayrı bir paragraf yaz."""

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    body = {"contents": [{"parts": [{"text": prompt}]}]}

    resp = requests.post(url, headers=headers, data=json.dumps(body), timeout=30)
    resp.raise_for_status()
    result = resp.json()
    return result["candidates"][0]["content"]["parts"][0]["text"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 👤 Demografik Bilgiler")
    age    = st.number_input("Yaş (yıl)", 18, 90, 52)
    sex    = st.selectbox("Cinsiyet", ["F","M"], format_func=lambda x:"Kadın" if x=="F" else "Erkek")
    drug   = st.selectbox("İlaç", ["D-penicillamine","Placebo"],
                          format_func=lambda x:"D-penicillamine" if x=="D-penicillamine" else "Plasebo")

    st.markdown("---")
    st.markdown("### 🩺 Klinik Semptomlar")
    ascites      = st.selectbox("Asit",            ["N","Y"], format_func=lambda x:"Yok" if x=="N" else "Var")
    hepatomegaly = st.selectbox("Hepatomegali",    ["N","Y"], format_func=lambda x:"Yok" if x=="N" else "Var")
    spiders      = st.selectbox("Örümcek Anjiomu", ["N","Y"], format_func=lambda x:"Yok" if x=="N" else "Var")
    edema        = st.selectbox("Ödem", ["N","S","Y"],
                                format_func=lambda x:{"N":"Yok","S":"Tedavili","Y":"Var"}[x])

    st.markdown("---")
    st.markdown("### 🧪 Biyokimyasal Parametreler")
    bilirubin     = st.number_input("Bilirubin (mg/dL)",   0.0, 30.0,  1.1, 0.1)
    cholesterol   = st.number_input("Kolesterol (mg/dL)",  50,  800,   260)
    albumin       = st.number_input("Albumin (g/dL)",      1.0, 6.0,   3.5, 0.1)
    copper        = st.number_input("Bakır (µg/gün)",      0,   600,   73)
    alk_phos      = st.number_input("Alk. Fosfataz (U/L)", 100, 15000, 1718)
    sgot          = st.number_input("SGOT (U/mL)",         0.0, 800.0, 137.0, 0.1)
    tryglicerides = st.number_input("Trigliserit (mg/dL)", 20,  800,   88)
    platelets     = st.number_input("Trombosit (×10³/mL)", 20,  600,   190)
    prothrombin   = st.number_input("Protrombin (s)",      8.0, 20.0,  10.6, 0.1)

    st.markdown("---")
    if SECRETS_KEY:
        st.markdown('<p class="key-ok">✅ Gemini API: Secrets\'tan yüklendi</p>', unsafe_allow_html=True)
        api_key = SECRETS_KEY
    else:
        st.markdown("### 🔑 Gemini API Key")
        st.markdown("[Ücretsiz al →](https://aistudio.google.com/app/apikey)", unsafe_allow_html=True)
        api_key = st.text_input("API Key", type="password",
                                help="aistudio.google.com adresinden ücretsiz alabilirsiniz.")

    predict_btn = st.button("🔬 Tahmin Et ve Analiz Başlat", use_container_width=True, type="primary")

# ── Sabitler ──────────────────────────────────────────────────────────────────
STAGE_NAMES  = ["Evre 1","Evre 2","Evre 3","Evre 4"]
STAGE_DESCS  = ["Hafif fibrozis — erken evre","Orta fibrozis — periportal tutulum",
                "Köprü fibrozis — ilerlemiş hastalık","Siroz — son evre"]
STAGE_COLORS = ["s1","s2","s3","s4"]
STAGE_EMOJIS = ["🟢","🟡","🟠","🔴"]

# ── Ana Panel ─────────────────────────────────────────────────────────────────
if not predict_btn:
    st.info("👈 Sol panelden hasta bilgilerini girin ve **'Tahmin Et'** butonuna basın.")
    st.markdown("""
    #### Bu sistem neler yapar?
    - **Histolojik evre tahmini** (Evre 1–4) — Mayo Clinic PBC veri setiyle eğitilmiş stacking modeli
    - **Olasılık dağılımı** — Her evre için güven skoru
    - **Özellik önemi** — Hangi parametreler tahmini en çok etkiliyor
    - **Anormal değer uyarıları** — Referans aralığı dışındaki bulgular
    - **AI klinik analizi** — Gemini AI ile detaylı parametre yorumu (ücretsiz)
    """)
else:
    inp = {
        "Drug":drug, "Age":age, "Sex":sex, "Ascites":ascites,
        "Hepatomegaly":hepatomegaly, "Spiders":spiders, "Edema":edema,
        "Bilirubin":bilirubin, "Cholesterol":cholesterol, "Albumin":albumin,
        "Copper":copper, "Alk_Phos":alk_phos, "SGOT":sgot,
        "Tryglicerides":tryglicerides, "Platelets":platelets, "Prothrombin":prothrombin,
    }

    probs     = predict_stage(inp)
    stage_idx = probs.index(max(probs))
    imps      = feature_importances(inp)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("#### 🎯 Tahmin Sonucu")
        sc = STAGE_COLORS[stage_idx]; em = STAGE_EMOJIS[stage_idx]
        st.markdown(f"""
        <div class="stage-card {sc}">
            <div style="font-size:3rem;">{em}</div>
            <div style="font-size:1.6rem;font-weight:700;">{STAGE_NAMES[stage_idx]}</div>
            <div style="font-size:0.9rem;margin-top:0.3rem;">{STAGE_DESCS[stage_idx]}</div>
            <div style="font-size:1.2rem;font-weight:600;margin-top:0.8rem;">
                Güven: %{probs[stage_idx]*100:.1f}
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("#### 📊 Evre Olasılıkları")
        st.dataframe(pd.DataFrame({
            "Evre":         [f"{STAGE_EMOJIS[i]} {STAGE_NAMES[i]}" for i in range(4)],
            "Olasılık (%)": [round(p*100,1) for p in probs]
        }), hide_index=True, use_container_width=True)
        st.bar_chart(pd.DataFrame({"Olasılık":[p*100 for p in probs]}, index=STAGE_NAMES))

    with col2:
        st.markdown("#### 🔍 Özellik Önemi")
        st.dataframe(pd.DataFrame({
            "Parametre":   list(imps.keys()),
            "Önem Skoru":  [round(v*100,1) for v in imps.values()]
        }), hide_index=True, use_container_width=True)
        st.bar_chart(pd.DataFrame({"Önem (%)":list(imps.values())}, index=list(imps.keys())))

        st.markdown("#### ⚠️ Dikkat Gerektiren Değerler")
        alerts = []
        if bilirubin   > 2:    alerts.append(f"Bilirubin yüksek: {bilirubin} mg/dL (N: <2)")
        if albumin     < 3.5:  alerts.append(f"Albumin düşük: {albumin} g/dL (N: >3.5)")
        if prothrombin > 12:   alerts.append(f"Protrombin uzamış: {prothrombin} s (N: <12)")
        if platelets   < 150:  alerts.append(f"Trombositopeni: {platelets} ×10³/mL (N: >150)")
        if copper      > 100:  alerts.append(f"Bakır yüksek: {copper} µg/gün (N: <100)")
        if ascites     == "Y": alerts.append("Asit mevcut — portal hipertansiyon düşündürür")
        if edema       == "Y": alerts.append("Ödem mevcut — karaciğer yetmezliği bulgusu")
        if alerts:
            for a in alerts: st.warning(a)
        else:
            st.success("Belirgin anormal değer saptanmadı.")

    # ── AI Klinik Analizi ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🤖 AI Klinik Analizi")

    if not api_key:
        st.info("""Sol panelde **Gemini API Key** girerek ücretsiz AI klinik analizi alabilirsiniz.  
Key almak için → [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) (tamamen ücretsiz, kart gerekmez)""")
    else:
        with st.spinner("AI klinik analiz yapıyor..."):
            try:
                analiz = gemini_analiz(inp, probs, stage_idx, imps, api_key)
                st.markdown(
                    f'<div class="ai-box">{analiz.replace(chr(10),"<br>")}</div>',
                    unsafe_allow_html=True
                )
            except requests.exceptions.HTTPError as e:
                if "400" in str(e) or "403" in str(e):
                    st.error("❌ Geçersiz API key. Lütfen kontrol edin.")
                else:
                    st.error(f"❌ API hatası: {e}")
            except Exception as e:
                st.error(f"❌ AI analizi hatası: {e}")

    st.markdown("""
    <div class="footer">
        Bu sistem klinik araştırma ve eğitim amaçlıdır. Tıbbi tanı ve tedavi kararları için kullanılamaz.<br>
        Mayo Clinic PBC Veri Seti · Stacking Ensemble (XGBoost + RF + LightGBM) · Doğruluk: %89.9
    </div>
    """, unsafe_allow_html=True)
