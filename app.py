import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(layout="wide")
st.title("🔋 Battery Sizer & Simulator (95% Rule + Dynamic P80)")

# ==========================================================
# SIDEBAR PARAMETERS (ALL HARD-CODE REMOVED)
# ==========================================================

st.sidebar.header("⚙️ Paramètres")

dt_hours = st.sidebar.number_input("Pas de temps (heures)", value=0.25)

values_are_kw = st.sidebar.checkbox("Valeurs en kW (sinon kWh)", value=True)

tariff_import = st.sidebar.number_input("Tarif import (CHF/kWh)", value=0.32)
tariff_export = st.sidebar.number_input("Tarif export (CHF/kWh)", value=0.08)

roundtrip_eff = st.sidebar.slider("Rendement aller-retour", 0.5, 1.0, 0.92)

cap_min = st.sidebar.number_input("Capacité min (kWh)", 1, 100, 5)
cap_max = st.sidebar.number_input("Capacité max (kWh)", 1, 200, 30)
cap_step = st.sidebar.number_input("Pas capacité (kWh)", 1, 20, 1)

p_min = st.sidebar.number_input("Puissance min (kW)", 1, 50, 3)
p_max = st.sidebar.number_input("Puissance max (kW)", 1, 100, 10)
p_step = st.sidebar.number_input("Pas puissance (kW)", 1, 20, 1)

gain_threshold = st.sidebar.slider("Seuil % du gain max", 0.5, 1.0, 0.95)
daily_percentile = st.sidebar.slider("Percentile export journalier (Pxx)", 0.5, 0.99, 0.8)

# ==========================================================
# FILE UPLOAD
# ==========================================================

uploaded_file = st.file_uploader("📂 Charger fichier Excel", type=["xlsx","xls"])

if uploaded_file:

    df = pd.read_excel(uploaded_file)

    df.columns = df.columns.str.lower()

    # auto detect
    date_col = [c for c in df.columns if "date" in c][0]
    imp_col = [c for c in df.columns if any(x in c for x in ["soutirage","import","achat"] )][0]
    exp_col = [c for c in df.columns if any(x in c for x in ["export","surplus","injection"])][0]

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    if values_are_kw:
        df["import_kwh"] = df[imp_col] * dt_hours
        df["export_kwh"] = df[exp_col] * dt_hours
    else:
        df["import_kwh"] = df[imp_col]
        df["export_kwh"] = df[exp_col]

    # ==========================================================
    # DYNAMIC CAP FROM DAILY EXPORT
    # ==========================================================

    daily_export = df.groupby(df[date_col].dt.date)["export_kwh"].sum()
    cap_max_dyn = np.ceil(np.percentile(daily_export, daily_percentile*100))

    cap_max_dyn = min(cap_max_dyn, cap_max)
    cap_max_dyn = max(cap_max_dyn, cap_min)

    st.sidebar.markdown(f"### Capacité max dynamique: **{cap_max_dyn} kWh**")

    # ==========================================================
    # QUICK SIM FUNCTION
    # ==========================================================

    def simulate(cap_kwh, p_kw):
        eta = np.sqrt(roundtrip_eff)
        soc = 0
        p_step = p_kw * dt_hours

        imp_after = 0
        exp_after = 0
        sum_charge = 0
        sum_dis = 0

        for i,row in df.iterrows():

            imp = row["import_kwh"]
            exp = row["export_kwh"]

            charge = min(exp, p_step, max(cap_kwh - soc,0))
            soc += charge * eta
            exp_after += exp - charge

            discharge = min(imp, p_step, soc)
            soc -= discharge / eta
            imp_after += imp - discharge

            sum_charge += charge
            sum_dis += discharge

        eq_cycles = (sum_charge + sum_dis)/(2*cap_kwh) if cap_kwh>0 else 0

        imp_before = df["import_kwh"].sum()
        exp_before = df["export_kwh"].sum()

        gain = (imp_before - imp_after)*tariff_import - (exp_before - exp_after)*tariff_export

        return gain, imp_after, exp_after, eq_cycles

    # ==========================================================
    # GRID SEARCH
    # ==========================================================

    results = []

    for cap in np.arange(cap_min, cap_max_dyn+1, cap_step):
        for p in np.arange(p_min, p_max+1, p_step):

            gain, imp_a, exp_a, cyc = simulate(cap,p)
            results.append([cap,p,gain,cyc])

    results = pd.DataFrame(results, columns=["Cap_kWh","Power_kW","Gain_CHF","Cycles"])

    gain_max = results["Gain_CHF"].max()
    threshold = gain_threshold * gain_max

    candidates = results[results["Gain_CHF"]>=threshold]
    best = candidates.sort_values(["Cap_kWh","Power_kW"]).iloc[0]

    st.success(f"🔋 Batterie optimale : {best.Cap_kWh} kWh / {best.Power_kW} kW")
    st.write(f"Gain annuel: {round(best.Gain_CHF,2)} CHF")

    # ==========================================================
    # GRAPH 1 – Gain surface
    # ==========================================================

    fig = go.Figure(data=[go.Scatter3d(
        x=results["Cap_kWh"],
        y=results["Power_kW"],
        z=results["Gain_CHF"],
        mode='markers',
        marker=dict(size=4,color=results["Gain_CHF"],colorscale="Viridis")
    )])

    fig.update_layout(title="Surface Gain CHF")
    st.plotly_chart(fig, use_container_width=True)

    # ==========================================================
    # DETAILED SIM BEST
    # ==========================================================

    eta = np.sqrt(roundtrip_eff)
    soc=0
    soc_list=[]

    for i,row in df.iterrows():
        imp=row["import_kwh"]
        exp=row["export_kwh"]

        charge=min(exp,best.Power_kW*dt_hours,max(best.Cap_kWh-soc,0))
        soc+=charge*eta

        discharge=min(imp,best.Power_kW*dt_hours,soc)
        soc-=discharge/eta

        soc_list.append(soc)

    df["SOC"] = soc_list

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df[date_col], y=df["SOC"], name="SoC"))
    fig2.update_layout(title="État de charge batterie")
    st.plotly_chart(fig2, use_container_width=True)

    # ==========================================================
    # SUMMARY
    # ==========================================================

    st.header("📊 Résumé annuel")

    st.dataframe(results.sort_values("Gain_CHF",ascending=False).head(10))
