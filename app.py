import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib, os, json, warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# ── PAGE CONFIG ──────────────────────────────────────────
st.set_page_config(
    page_title="AQI Prediction System | DRDO",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ───────────────────────────────────────────
st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #0f3d22, #1a6b3c);
    padding: 1.5rem 2rem;
    border-radius: 12px;
    color: white;
    margin-bottom: 1.5rem;
}
.main-header h1 { color: white; font-size: 1.8rem; margin: 0; }
.main-header p  { color: #a8d5b5; margin: 4px 0 0; font-size: 0.9rem; }
.metric-card {
    background: #f8fffe;
    border: 1px solid #d0ead8;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.aqi-box {
    text-align: center;
    padding: 1.5rem;
    border-radius: 12px;
    margin: 1rem 0;
}
.section-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #0f3d22;
    border-left: 4px solid #1a6b3c;
    padding-left: 10px;
    margin: 1.2rem 0 0.8rem;
}
</style>
""", unsafe_allow_html=True)

# ── AQI CATEGORY HELPER ──────────────────────────────────
def get_category(aqi):
    if   aqi <= 50:  return "Good",       "#00c853", "#e8f5e9", "😊 Air quality is satisfactory. Enjoy outdoor activities!"
    elif aqi <= 100: return "Satisfactory","#558b2f", "#f1f8e9", "🙂 Minor discomfort possible for sensitive individuals."
    elif aqi <= 200: return "Moderate",    "#e65100", "#fff3e0", "😐 Sensitive groups may experience breathing discomfort."
    elif aqi <= 300: return "Poor",        "#b71c1c", "#ffebee", "😷 Breathing discomfort for most. Reduce outdoor activity."
    elif aqi <= 400: return "Very Poor",   "#6a1b9a", "#f3e5f5", "🚨 Respiratory illness risk. Avoid outdoor exposure."
    else:            return "Severe",      "#880e4f", "#fce4ec", "⛔ Health emergency. Everyone should avoid outdoors."

# ── LOAD OR TRAIN MODELS ─────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_or_train(data_path):
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.drop_duplicates(inplace=True)
    df.dropna(subset=['AQI'], inplace=True)

    pollutants = [c for c in ['PM2.5','PM10','NO','NO2','NOx','NH3','CO','SO2',
                               'O3','Benzene','Toluene','Xylene'] if c in df.columns]
    for col in pollutants:
        df[col] = df.groupby('City')[col].transform(lambda x: x.fillna(x.median()))
    df[pollutants] = df[pollutants].fillna(df[pollutants].median())

    df['Month']      = df['Date'].dt.month
    df['Season']     = df['Month'].map({12:'Winter',1:'Winter',2:'Winter',
                                         3:'Spring',4:'Spring',5:'Spring',
                                         6:'Summer',7:'Summer',8:'Summer',
                                         9:'Monsoon',10:'Monsoon',11:'Monsoon'})
    le = LabelEncoder()
    df['City_enc']   = le.fit_transform(df['City'])
    df['Season_enc'] = LabelEncoder().fit_transform(df['Season'])

    features = pollutants + ['Month','City_enc','Season_enc']
    X = df[features]; y = df['AQI']
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree':     DecisionTreeRegressor(max_depth=12, random_state=42),
        'Random Forest':     RandomForestRegressor(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42),
    }
    try:
        from xgboost import XGBRegressor
        models['XGBoost'] = XGBRegressor(n_estimators=100, max_depth=6,
                                          learning_rate=0.1, random_state=42, verbosity=0)
    except: pass

    results = {}
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        pred = m.predict(X_te)
        results[name] = {
            'model': m,
            'pred':  pred,
            'y_test':y_te,
            'MAE':   round(mean_absolute_error(y_te, pred),2),
            'RMSE':  round(np.sqrt(mean_squared_error(y_te, pred)),2),
            'R2':    round(r2_score(y_te, pred),4),
        }
    return df, models, results, features, pollutants, le

# ── SIDEBAR ──────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/b/b5/DRDO_logo.png", width=80)
    st.markdown("### AQI Prediction System")
    st.caption("DRDO Internship · Offline ML App")
    st.divider()

    data_file = st.text_input("Dataset path", value="city_day.csv",
                               help="Place city_day.csv in the same folder as app.py")
    st.divider()
    st.markdown("**Navigation**")
    page = st.radio("", ["📊 Dashboard", "🔮 Predict AQI", "🤖 Model Comparison",
                          "📈 EDA & Plots", "📋 Report"], label_visibility="collapsed")

# ── LOAD DATA ────────────────────────────────────────────
if not os.path.exists(data_file):
    st.error(f"Dataset not found at **{data_file}**. "
             f"Copy `city_day.csv` from your pen drive to the same folder as `app.py` and restart.")
    st.stop()

with st.spinner("Loading data and training models (first run only)…"):
    df, models, results, features, pollutants, le = load_or_train(data_file)

# ── HEADER ───────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🌿 AQI Prediction System</h1>
  <p>DRDO Internship Project &nbsp;·&nbsp; Kushagra Saxena, MAIT Delhi &nbsp;·&nbsp;
     Dataset: CPCB India 2015–2020 &nbsp;·&nbsp; Fully Offline</p>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ════════════════════════════════════════════════════════
if page == "📊 Dashboard":
    c1, c2, c3, c4 = st.columns(4)
    best_name = max(results, key=lambda k: results[k]['R2'])
    c1.metric("Total Records",   f"{len(df):,}")
    c2.metric("Cities",           f"{df['City'].nunique()}")
    c3.metric("Mean AQI",         f"{df['AQI'].mean():.1f}")
    c4.metric("Best Model R²",    f"{results[best_name]['R2']:.4f}",
              delta=best_name, delta_color="normal")

    st.markdown('<div class="section-title">City-wise Average AQI</div>', unsafe_allow_html=True)
    city_aqi = df.groupby('City')['AQI'].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(14,5))
    colors_b = ['#b71c1c' if x>200 else '#ff9800' if x>100 else '#4caf50' for x in city_aqi.values]
    ax.bar(city_aqi.index, city_aqi.values, color=colors_b, edgecolor='white')
    ax.axhline(100, color='orange', linestyle='--', lw=1.5, label='Moderate (100)')
    ax.axhline(200, color='red',    linestyle='--', lw=1.5, label='Poor (200)')
    ax.set_title('Average AQI by City (2015–2020)', fontweight='bold')
    ax.tick_params(axis='x', rotation=45); ax.legend()
    plt.tight_layout(); st.pyplot(fig); plt.close()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title">AQI Distribution</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6,4))
        ax.hist(df['AQI'], bins=50, color='#1a6b3c', edgecolor='white', alpha=0.85)
        ax.axvline(df['AQI'].mean(), color='red', linestyle='--',
                   label=f'Mean: {df["AQI"].mean():.1f}')
        ax.set_xlabel('AQI'); ax.set_ylabel('Frequency')
        ax.set_title('AQI Frequency', fontweight='bold'); ax.legend()
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.markdown('<div class="section-title">AQI Category Breakdown</div>', unsafe_allow_html=True)
        cats   = ['Good','Satisfactory','Moderate','Poor','Very Poor','Severe']
        clrs   = ['#00c853','#76ff03','#ff9800','#f44336','#9c27b0','#b71c1c']
        bins_c = [0,50,100,200,300,400,9999]
        df['AQI_Cat'] = pd.cut(df['AQI'], bins=bins_c, labels=cats)
        counts = df['AQI_Cat'].value_counts().reindex(cats).fillna(0)
        fig, ax = plt.subplots(figsize=(6,4))
        ax.bar(cats, counts.values, color=clrs, edgecolor='white')
        ax.set_title('CPCB Category Count', fontweight='bold')
        ax.tick_params(axis='x', rotation=30)
        plt.tight_layout(); st.pyplot(fig); plt.close()

# ════════════════════════════════════════════════════════
# PAGE 2 — PREDICT AQI
# ════════════════════════════════════════════════════════
elif page == "🔮 Predict AQI":
    st.markdown('<div class="section-title">Enter Pollutant Readings</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 1])
    with col1:
        with st.form("predict_form"):
            city_opt = sorted(df['City'].unique().tolist())
            city     = st.selectbox("City", city_opt)
            month    = st.slider("Month", 1, 12, 6)
            st.markdown("**Pollutant Concentrations**")
            c1, c2, c3 = st.columns(3)
            vals = {}
            defaults = {'PM2.5':85,'PM10':140,'NO':30,'NO2':45,
                        'NOx':75,'NH3':25,'CO':8,'SO2':20,'O3':60,
                        'Benzene':2,'Toluene':5,'Xylene':1}
            units    = {'PM2.5':'µg/m³','PM10':'µg/m³','NO':'µg/m³','NO2':'µg/m³',
                        'NOx':'µg/m³','NH3':'µg/m³','CO':'mg/m³','SO2':'µg/m³',
                        'O3':'µg/m³','Benzene':'µg/m³','Toluene':'µg/m³','Xylene':'µg/m³'}
            maxs     = {'PM2.5':500,'PM10':600,'NO':200,'NO2':300,
                        'NOx':500,'NH3':100,'CO':50,'SO2':150,'O3':300,
                        'Benzene':20,'Toluene':40,'Xylene':20}
            for i, p in enumerate([pp for pp in pollutants]):
                col = [c1,c2,c3][i%3]
                with col:
                    vals[p] = st.number_input(f"{p} ({units.get(p,'')})",
                                               0.0, float(maxs.get(p,500)),
                                               float(defaults.get(p,10)), step=0.5)
            model_choice = st.selectbox("Model", list(models.keys()))
            submitted    = st.form_submit_button("🔮 Predict AQI", use_container_width=True)

    with col2:
        if submitted:
            season_map = {12:'Winter',1:'Winter',2:'Winter',
                          3:'Spring',4:'Spring',5:'Spring',
                          6:'Summer',7:'Summer',8:'Summer',
                          9:'Monsoon',10:'Monsoon',11:'Monsoon'}
            season     = season_map[month]
            try:
                city_enc   = le.transform([city])[0]
            except:
                city_enc   = 0
            season_enc = {'Spring':0,'Summer':1,'Monsoon':2,'Winter':3}.get(season,0)

            row = {p: vals.get(p,0) for p in pollutants}
            row['Month']      = month
            row['City_enc']   = city_enc
            row['Season_enc'] = season_enc
            X_inp = pd.DataFrame([row])[features]

            mdl   = models[model_choice]
            pred  = float(mdl.predict(X_inp)[0])
            pred  = max(0, min(pred, 999))
            cat, txt_c, bg_c, msg = get_category(pred)

            st.markdown(f"""
            <div class="aqi-box" style="background:{bg_c}; border: 2px solid {txt_c}30">
                <div style="font-size:0.9rem;color:#555">Predicted AQI</div>
                <div style="font-size:3.5rem;font-weight:700;color:{txt_c}">{pred:.0f}</div>
                <div style="font-size:1.1rem;font-weight:600;color:{txt_c}">{cat}</div>
                <div style="font-size:0.85rem;color:#555;margin-top:8px">{msg}</div>
                <div style="font-size:0.75rem;color:#999;margin-top:12px">
                City: {city} &nbsp;·&nbsp; Month: {month} &nbsp;·&nbsp; Model: {model_choice}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Mini gauge
            fig, ax = plt.subplots(figsize=(5,1.2))
            ax.barh([0], [500], color='#eee', height=0.4)
            gradient = np.linspace(0,1,500).reshape(1,-1)
            ax.imshow(gradient, aspect='auto', cmap='RdYlGn_r',
                      extent=[0,500,-0.2,0.2], origin='lower', vmin=0, vmax=1)
            ax.axvline(pred, color='#111', lw=3)
            ax.set_xlim(0,500); ax.set_yticks([]); ax.set_xlabel('AQI scale (0–500)')
            for v,l in [(50,'Good'),(100,'Sat.'),(200,'Mod.'),(300,'Poor'),(400,'V.Poor')]:
                ax.axvline(v, color='white', lw=0.8, alpha=0.6)
                ax.text(v, 0.22, l, ha='center', fontsize=6, color='#333')
            ax.set_title('AQI Gauge', fontsize=9, fontweight='bold')
            plt.tight_layout(); st.pyplot(fig); plt.close()
        else:
            st.info("Fill in pollutant values and click **Predict AQI**")

# ════════════════════════════════════════════════════════
# PAGE 3 — MODEL COMPARISON
# ════════════════════════════════════════════════════════
elif page == "🤖 Model Comparison":
    st.markdown('<div class="section-title">Performance Metrics</div>', unsafe_allow_html=True)
    rows = []
    for name, r in results.items():
        rows.append({'Model': name, 'R² Score': r['R2'], 'MAE': r['MAE'], 'RMSE': r['RMSE']})
    results_df = pd.DataFrame(rows).sort_values('R² Score', ascending=False).reset_index(drop=True)
    st.dataframe(results_df.style.highlight_max(subset=['R² Score'], color='#d4edda')
                                  .highlight_min(subset=['MAE','RMSE'], color='#d4edda')
                                  .format({'R² Score':'{:.4f}','MAE':'{:.2f}','RMSE':'{:.2f}'}),
                 use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6,4))
        colors_bar = ['#1a6b3c','#2196f3','#ff9800','#e91e63'][:len(results)]
        r2_vals = [results[n]['R2'] for n in results]
        bars = ax.bar(list(results.keys()), r2_vals, color=colors_bar, edgecolor='white')
        for b,v in zip(bars, r2_vals):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.003, f'{v:.4f}',
                    ha='center', fontsize=9)
        ax.set_title('R² Score Comparison', fontweight='bold')
        ax.set_ylabel('R² Score'); ax.tick_params(axis='x', rotation=20)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(6,4))
        mae_vals = [results[n]['MAE'] for n in results]
        bars = ax.bar(list(results.keys()), mae_vals, color=colors_bar, edgecolor='white')
        for b,v in zip(bars, mae_vals):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.3, f'{v:.2f}',
                    ha='center', fontsize=9)
        ax.set_title('MAE Comparison (lower is better)', fontweight='bold')
        ax.set_ylabel('Mean Absolute Error'); ax.tick_params(axis='x', rotation=20)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown('<div class="section-title">Actual vs Predicted</div>', unsafe_allow_html=True)
    selected = st.selectbox("Select model", list(results.keys()))
    r = results[selected]
    fig, axes = plt.subplots(1, 2, figsize=(13,5))
    lim = max(r['y_test'].max(), r['pred'].max()) * 1.05
    axes[0].scatter(r['y_test'], r['pred'], alpha=0.3, color='#1a6b3c', s=12)
    axes[0].plot([0,lim],[0,lim],'r--',lw=1.5,label='Perfect prediction')
    axes[0].set_xlabel('Actual AQI'); axes[0].set_ylabel('Predicted AQI')
    axes[0].set_title(f'{selected}: Actual vs Predicted', fontweight='bold')
    axes[0].set_xlim(0,lim); axes[0].set_ylim(0,lim); axes[0].legend()

    residuals = r['y_test'].values - r['pred']
    axes[1].hist(residuals, bins=50, color='#2196f3', edgecolor='white', alpha=0.85)
    axes[1].axvline(0, color='red', linestyle='--', lw=1.5)
    axes[1].set_xlabel('Residual'); axes[1].set_ylabel('Frequency')
    axes[1].set_title('Residuals Distribution', fontweight='bold')
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # Feature importance
    if 'Random Forest' in models:
        st.markdown('<div class="section-title">Feature Importance (Random Forest)</div>', unsafe_allow_html=True)
        rf = models['Random Forest']
        importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=True).tail(12)
        fig, ax = plt.subplots(figsize=(10,5))
        colors_fi = ['#1a6b3c' if i >= len(importances)-3 else '#90caa8'
                     for i in range(len(importances))]
        importances.plot(kind='barh', color=colors_fi, edgecolor='white', ax=ax)
        ax.set_title('Feature Importance — Random Forest', fontweight='bold')
        ax.set_xlabel('Importance Score')
        plt.tight_layout(); st.pyplot(fig); plt.close()

# ════════════════════════════════════════════════════════
# PAGE 4 — EDA & PLOTS
# ════════════════════════════════════════════════════════
elif page == "📈 EDA & Plots":
    st.markdown('<div class="section-title">Correlation Heatmap</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(13,9))
    corr = df[pollutants+['AQI']].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, linewidths=0.4, annot_kws={'size':8}, ax=ax)
    ax.set_title('Pollutant Correlation Matrix', fontsize=13, fontweight='bold')
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown('<div class="section-title">Pollutant Distributions</div>', unsafe_allow_html=True)
    key_p  = [p for p in ['PM2.5','PM10','NO2','SO2','CO','O3'] if p in df.columns]
    pal    = ['#1E88E5','#E53935','#43A047','#8E24AA','#FB8C00','#00ACC1']
    fig, axes = plt.subplots(2, 3, figsize=(15,9))
    axes = axes.flatten()
    for i, col in enumerate(key_p):
        data = df[col].dropna()
        cap  = data.quantile(0.99)
        axes[i].hist(data[data<=cap], bins=40, color=pal[i], edgecolor='white', alpha=0.85)
        axes[i].axvline(data.mean(), color='red', linestyle='--', lw=1.2,
                        label=f'Mean: {data.mean():.1f}')
        axes[i].set_title(f'{col} Distribution', fontweight='bold')
        axes[i].set_xlabel(col); axes[i].legend(fontsize=8)
    plt.suptitle('Key Pollutant Distributions (99th pctile cap)', fontsize=12, fontweight='bold')
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown('<div class="section-title">Monthly AQI Trend</div>', unsafe_allow_html=True)
    df['YM'] = df['Date'].dt.to_period('M')
    monthly  = df.groupby('YM')['AQI'].mean()
    fig, ax  = plt.subplots(figsize=(14,4))
    ax.plot(monthly.index.astype(str), monthly.values, color='#1a6b3c', lw=2)
    ax.fill_between(range(len(monthly)), monthly.values, alpha=0.15, color='#1a6b3c')
    xticks = range(0, len(monthly), 6)
    ax.set_xticks(list(xticks))
    ax.set_xticklabels([list(monthly.index.astype(str))[i] for i in xticks], rotation=45)
    ax.set_title('Monthly Average AQI (All Cities)', fontweight='bold')
    ax.set_xlabel('Month'); ax.set_ylabel('Avg AQI')
    plt.tight_layout(); st.pyplot(fig); plt.close()

# ════════════════════════════════════════════════════════
# PAGE 5 — REPORT
# ════════════════════════════════════════════════════════
elif page == "📋 Report":
    st.markdown("## Project Report")
    best_name = max(results, key=lambda k: results[k]['R2'])
    best      = results[best_name]

    st.markdown(f"""
**Title:** AQI Prediction System Using Machine Learning  
**Intern:** Kushagra Saxena | MAIT, Delhi  
**Organisation:** DRDO | June–July 2026  
**Dataset:** CPCB India Air Quality Data (2015–2020)

---
### Objective
To predict the Air Quality Index (AQI) of Indian cities using supervised machine learning regression models based on ground-level pollutant sensor readings.

### Dataset Summary
| Parameter | Value |
|---|---|
| Total Records | {len(df):,} |
| Cities Covered | {df['City'].nunique()} |
| Date Range | {df['Date'].min().date()} – {df['Date'].max().date()} |
| Features Used | {', '.join(pollutants)} |
| Target Variable | AQI |

### Methodology
1. **Data Collection:** CPCB India dataset (2015–2020) downloaded from Kaggle
2. **Data Cleaning:** Removed duplicates; city-wise median imputation for missing values
3. **Feature Engineering:** Added Month, Season, City encoding
4. **Models Trained:** Linear Regression, Decision Tree, Random Forest, XGBoost
5. **Evaluation:** MAE, RMSE, R² Score on 20% test split

### Model Performance
| Model | R² Score | MAE | RMSE |
|---|---|---|---|
""")
    for name, r in sorted(results.items(), key=lambda x: -x[1]['R2']):
        st.markdown(f"| {name} | {r['R2']} | {r['MAE']} | {r['RMSE']} |")

    st.markdown(f"""
### Key Findings
- **Best Model:** {best_name} with R² = {best['R2']}, MAE = {best['MAE']}
- **Most Polluted City:** {df.groupby('City')['AQI'].mean().idxmax()}
- **Strongest AQI Predictor:** PM2.5 (highest correlation with AQI)
- **Seasonal Pattern:** AQI peaks in winter months (Nov–Jan) due to low dispersion

### Conclusion
The {best_name} model achieved the best performance with R² = {best['R2']:.4f}, 
indicating it explains {best['R2']*100:.1f}% of variance in AQI. 
The system runs fully offline and can be deployed on air-gapped systems.

### References
1. Kumar, K. & Pande, B.P. (2022). Air Pollution Prediction with Machine Learning: A Case Study of Indian Cities. *Int. J. Environmental Science and Technology.*
2. Gupta et al. (2023). Prediction of AQI Using ML Techniques. *Journal of Environmental and Public Health*, Wiley.
3. IJRASET (2023). Review on AQI Prediction and Management Using Machine Learning.
4. Springer Nature (2025). Air Quality Forecasting: Comparative Analysis and Ensemble Strategies.
""")
