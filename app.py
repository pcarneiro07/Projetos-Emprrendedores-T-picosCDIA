# app.py
import os
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

import psycopg2
from psycopg2.extras import DictCursor

from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LassoCV

import plotly.graph_objects as go

# =========================
# CONFIG & CREDENCIAIS
# =========================
st.set_page_config(page_title="Monitor Artemis — Anomalias Diárias", layout="wide")

PGUSER     = "postgres.xxjasxfersfkhivepyri"
PGPASSWORD = "J4w2yKW7eTyR$k?"
PGHOST     = "aws-1-us-east-2.pooler.supabase.com"
PGPORT     = "5432"
PGDATABASE = "postgres"

# =========================
# FUNÇÕES AUXILIARES
# =========================

def _connect():
    return psycopg2.connect(
        user=PGUSER, password=PGPASSWORD, host=PGHOST, port=PGPORT,
        dbname=PGDATABASE, sslmode="require", connect_timeout=10,
        options="-c statement_timeout=60000"
    )

def _safe_to_datetime_series(series: pd.Series) -> pd.Series:
    s = pd.to_datetime(series, errors="coerce", utc=True, format="%d/%m/%Y %H:%M")
    if s.notna().sum() == 0:
        s = pd.to_datetime(series, errors="coerce", utc=True, format="%d-%m-%Y %H:%M")
    return s

def _to_float_cols(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = (
                df[c].astype(str)
                .str.replace(",", ".", regex=False)
                .str.replace(" ", "", regex=False)
            )
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def robust_z(x: pd.Series) -> pd.Series:
    med = x.median()
    mad = (x - med).abs().median()
    mad = 1.4826 * (mad if mad != 0 else (x.std() if x.std() > 0 else 1.0))
    return (x - med) / mad

def scale_0_100(vec: np.ndarray, p_low=10, p_high=90) -> np.ndarray:
    v = np.asarray(vec).astype(float)
    lo, hi = np.nanpercentile(v, p_low), np.nanpercentile(v, p_high)
    rng = (hi - lo) if hi > lo else (np.nanstd(v) * 2 or 1.0)
    s = (v - lo) / rng
    s = np.clip(s, 0, 1) * 100.0
    return s

def color_for_level(level: str) -> str:
    return {
        "Normal": "#22c55e",
        "Atenção": "#f59e0b",
        "Crítica": "#ef4444",
    }.get(level, "#94a3b8")

# =========================
# CARREGAMENTO & FEATURES
# =========================

@st.cache_data(show_spinner=True)
def load_daily_dataset():
    conn = _connect()
    cur = conn.cursor(cursor_factory=DictCursor)
    cur.execute('SELECT * FROM public."quali_artemis";')
    quali = pd.DataFrame(cur.fetchall(), columns=[d[0] for d in cur.description])
    cur.execute('SELECT * FROM public."quanti_artemis";')
    quanti = pd.DataFrame(cur.fetchall(), columns=[d[0] for d in cur.description])
    cur.close(); conn.close()

    # --- QUALI
    if "Data hora" in quali.columns:
        quali["Data hora"] = _safe_to_datetime_series(quali["Data hora"])
        quali = quali.dropna(subset=["Data hora"]).sort_values("Data hora")
        quali["date"] = quali["Data hora"].dt.date
    quali_float_cols = [
        "pH",
        "Oxigênio Dissolvido (mg/L)",
        "Condutividade Elétrica (µS/cm)",
        "Turbidez (NTU)",
        "Temperatura (°C)"
    ]
    quali = _to_float_cols(quali, quali_float_cols)
    quali_day = quali.groupby("date", as_index=False)[quali_float_cols].mean()

    # --- QUANTI (SEM "Chuva no ponto (mm)")
    if "Data hora" in quanti.columns:
        quanti["Data hora"] = _safe_to_datetime_series(quanti["Data hora"])
        quanti = quanti.dropna(subset=["Data hora"]).sort_values("Data hora")
        quanti["date"] = quanti["Data hora"].dt.date
    quanti_float_cols = [
        "Leitura de régua (m)",
        "Cota referenciada (m)",
        "Vazão (m³/s)"
    ]
    quanti = _to_float_cols(quanti, quanti_float_cols)
    quanti_day = quanti.groupby("date", as_index=False)[quanti_float_cols].mean()

    daily = pd.merge(quali_day, quanti_day, on="date", how="inner").sort_values("date").reset_index(drop=True)
    return daily, quali_float_cols, quanti_float_cols

# =========================
# SCORES DOS MODELOS
# =========================

def compute_component_scores(Xs: np.ndarray, feature_cols: list, random_state: int = 42):
    """
    Retorna:
      - scores_raw (0-100 por componente usando percentis 5-95, pré-robustificação)
      - scores_robust (z robusto por componente -> 0-100 via percentis 10-90) [USO NO ENSEMBLE]
      - models dict
    """
    iso = IsolationForest(n_estimators=400, contamination=0.02, random_state=random_state)
    iso.fit(Xs); iso_score = scale_0_100(-iso.decision_function(Xs), 5, 95)

    lof = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=0.02)
    lof.fit(Xs); lof_score = scale_0_100(-lof.decision_function(Xs), 5, 95)

    oc = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
    oc.fit(Xs); oc_score = scale_0_100(-oc.decision_function(Xs), 5, 95)

    ee = EllipticEnvelope(contamination=0.02, support_fraction=0.9, random_state=random_state)
    ee.fit(Xs); ee_score = scale_0_100(-ee.score_samples(Xs), 5, 95)

    # Estatístico baseado em RZ médio (dos Xs já padronizados por RobustScaler)
    xs_df = pd.DataFrame(Xs, columns=feature_cols)
    stat_rz = xs_df.apply(robust_z).abs().mean(axis=1)
    stat_score = scale_0_100(stat_rz, 5, 95)

    scores_raw = pd.DataFrame({"stat": stat_score, "iso": iso_score, "lof": lof_score, "oc": oc_score, "ee": ee_score})

    # Robustificação por componente
    scores_robust = scores_raw.copy()
    for c in scores_raw.columns:
        z = robust_z(scores_raw[c])
        scores_robust[c] = scale_0_100(z, 10, 90)  # mais conservador
    
    models = {"iso": iso, "lof": lof, "oc": oc, "ee": ee}
    return scores_raw, scores_robust, models

def weighted_index(scores_df: pd.DataFrame, weights: dict) -> np.ndarray:
    wv = np.array([weights["stat"], weights["iso"], weights["lof"], weights["oc"], weights["ee"]], dtype=float)
    mat = scores_df[["stat","iso","lof","oc","ee"]].values
    idx = mat.dot(wv)
    return np.clip(idx, 0, 100)  # segurança

# =========================
# CLUSTERIZAÇÃO E LIMIARES
# =========================

def kmeans_thresholds_from_index(idx: np.ndarray, k: int = 4, random_state: int = 42):
    km = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
    idx_reshaped = idx.reshape(-1, 1)
    km.fit(idx_reshaped)
    centers = km.cluster_centers_.flatten()
    order = np.argsort(centers)
    centers_sorted = centers[order]
    thresholds = [(centers_sorted[i] + centers_sorted[i+1]) / 2 for i in range(k-1)]
    label_order = {old: np.where(order == old)[0][0] for old in range(k)}
    labels_ordered = np.vectorize(label_order.get)(km.labels_)
    names = ["Normal","Atenção","Crítica"][:k]
    return labels_ordered, centers_sorted, thresholds, names

def classify_by_thresholds(value: float, thresholds: list, names: list):
    if len(thresholds) == 0:
        return names[-1]
    if value <= thresholds[0]:
        return names[0]
    for i in range(1, len(thresholds)):
        if value <= thresholds[i]:
            return names[i]
    return names[-1]

# =========================
# BUSCA DE PESOS (OBJ penalizado)
# =========================

def sample_dirichlet(n_samples: int, dims: int, alpha: float = 1.0, rng=None):
    rng = np.random.default_rng(rng)
    return rng.dirichlet([alpha]*dims, size=n_samples)

def evaluate_with_penalty(idx: np.ndarray, labels: np.ndarray, centers_sorted: np.ndarray, spread_target: float, beta: float) -> float:
    # Silhouette em 1D
    if len(np.unique(labels)) < 2:
        return -np.inf
    sil = silhouette_score(idx.reshape(-1,1), labels, metric="euclidean")
    spread = float(centers_sorted[-1] - centers_sorted[0])  # 0..100
    penalty = max(0.0, spread - spread_target)
    return sil - beta * penalty

def random_search_weights(scores_df: pd.DataFrame, k: int, n_samples: int, spread_target: float, beta: float, random_state: int = 42):
    rng = np.random.default_rng(random_state)
    W = sample_dirichlet(n_samples, 5, alpha=1.0, rng=rng)
    best = (-np.inf, None, None, None, None, None)  # (score, weights, idx, labels, centers, thresholds)

    for w in W:
        weights = {"stat": w[0], "iso": w[1], "lof": w[2], "oc": w[3], "ee": w[4]}
        idx = weighted_index(scores_df, weights)
        labels, centers_sorted, thresholds, names = kmeans_thresholds_from_index(idx, k=k, random_state=int(rng.integers(0,1e9)))
        score = evaluate_with_penalty(idx, labels, centers_sorted, spread_target, beta)
        if score > best[0]:
            best = (score, weights, idx, labels, centers_sorted, thresholds)

    _, weights, idx, labels, centers_sorted, thresholds = best
    return weights, idx, labels, centers_sorted, thresholds, names

# =========================
# PIPELINE COMPLETO (CACHE)
# =========================

@st.cache_resource(show_spinner=True)
def fit_all(daily_df: pd.DataFrame, feature_cols: list, n_weight_samples: int, k_levels: int, spread_target: float, beta: float, random_state: int = 42):
    # Padronização de features de entrada
    scaler = RobustScaler()
    Xs = scaler.fit_transform(daily_df[feature_cols].astype(float))

    # Scores dos modelos
    scores_raw, scores_robust, models = compute_component_scores(Xs, feature_cols, random_state=random_state)

    # Otimização com penalização de spread
    best_w, idx, labels, centers_sorted, thresholds, names = random_search_weights(
        scores_df=scores_robust,  # usamos os scores robustificados
        k=k_levels,
        n_samples=n_weight_samples,
        spread_target=spread_target,
        beta=beta,
        random_state=random_state
    )

    # Monta DataFrame final
    out = daily_df[["date"]].copy()
    out["idx_0_100"] = idx
    out["nivel"] = [classify_by_thresholds(v, thresholds, names) for v in idx]

    # Contribuição relativa de cada componente no índice
    contrib = scores_robust.copy()
    for key in ["stat","iso","lof","oc","ee"]:
        contrib[f"{key}_contrib"] = (best_w[key] * contrib[key]) / np.maximum(idx, 1e-9)

    # Correlação feature × componentes (para contexto)
    corr = pd.DataFrame(index=feature_cols)
    xs_df = pd.DataFrame(Xs, columns=feature_cols)
    for key in ["stat","iso","lof","oc","ee"]:
        corr[key] = xs_df.corrwith(scores_robust[key])

    # Surrogate linear (explicabilidade global)
    lasso = LassoCV(cv=5, random_state=random_state, n_alphas=50).fit(Xs, idx)
    surrogate_coefs = pd.Series(lasso.coef_, index=feature_cols).sort_values(key=lambda s: s.abs(), ascending=False)

    results = {
        "scaler": scaler,
        "models": models,
        "scores_raw": scores_raw,
        "scores_components": scores_robust,   # << usado no ensemble
        "best_weights": best_w,
        "index": idx,
        "labels": labels,
        "centers_sorted": centers_sorted,
        "thresholds": thresholds,
        "names": names,
        "contrib_components": contrib[[c for c in contrib.columns if c.endswith("_contrib")]],
        "corr_features_components": corr,
        "surrogate_lasso": lasso,
        "surrogate_coefs": surrogate_coefs
    }
    return out, results

# =========================
# APP
# =========================

st.title("🌊 Monitor Artemis — Índice Diário de Anomalias (0–100)")

with st.sidebar:
    st.header("Configurações")
    n_samples = st.slider("Amostras p/ busca de pesos (Dirichlet)", 200, 5000, 1500, 100,
                          help="Quanto maior, potencialmente melhor (e mais lento).")
    k_levels = 3
    spread_target = st.slider("Alvo de spread entre centros (0–100)", 20, 100, 60, 5,
                              help="Quanto os centros podem se afastar antes de penalizar.")
    beta = st.slider("Penalidade por spread acima do alvo (β)", 0.000, 0.020, 0.005, 0.001,
                     help="Maior β = menos picos/extremos.")
    show_history = st.checkbox("Mostrar histórico do índice diário", True)
    show_corr = st.checkbox("Mostrar correlação Feature × Componentes", False)
    show_surrogate = st.checkbox("Mostrar explicabilidade global (modelo linear)", True)
    st.caption("Pesos são aprendidos maximizando Silhouette com penalização de spread. Componentes são robustificados antes da combinação.")

with st.spinner("Carregando dados..."):
    daily, quali_cols, quanti_cols = load_daily_dataset()

all_features = quali_cols + quanti_cols

with st.spinner("Treinando modelos e otimizando pesos..."):
    full_labels, results = fit_all(
        daily_df=daily, feature_cols=all_features,
        n_weight_samples=n_samples, k_levels=k_levels,
        spread_target=spread_target, beta=beta, random_state=42
    )

# Junta tudo
full = pd.concat(
    [daily.reset_index(drop=True),
     full_labels[["idx_0_100","nivel"]].reset_index(drop=True),
     results["scores_components"].reset_index(drop=True),
     results["contrib_components"].reset_index(drop=True)],
    axis=1
)

last_date = full["date"].max()
sel_date = st.date_input("Escolha o dia", value=last_date,
                         min_value=full["date"].min(), max_value=full["date"].max())

row = full.loc[full["date"] == sel_date]
if row.empty:
    st.warning("Sem dados para a data selecionada."); st.stop()

# =========================
# PAINEL PRINCIPAL
# =========================
centers_sorted = results["centers_sorted"]
thresholds = results["thresholds"]
names = results["names"]
best_w = results["best_weights"]

col1, col2 = st.columns([1,2], gap="large")

with col1:
    score = float(row["idx_0_100"].iloc[0])
    nivel = str(row["nivel"].iloc[0])

    st.subheader(f"📅 {sel_date.strftime('%d/%m/%Y')}")
    st.metric("Índice do dia (0–100)", f"{score:.1f}")
    st.markdown(
        f"<div style='padding:12px;border-radius:10px;background-color:{color_for_level(nivel)};color:white;font-weight:600;'>Situação: {nivel}</div>",
        unsafe_allow_html=True)

    

with col2:
    if show_history:
        hist = full[["date","idx_0_100","nivel"]].copy()
        hist["date"] = pd.to_datetime(hist["date"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hist["date"], y=hist["idx_0_100"],
            mode="lines+markers", line=dict(color="#1f77b4", width=2),
            marker=dict(size=5),
            customdata=np.stack([hist["nivel"]], axis=-1),
            hovertemplate="<b>%{x|%d/%m/%Y}</b><br>Índice: %{y:.1f}<br>Situação: %{customdata[0]}<extra></extra>",
        ))
        # bandas de nível
        y_min, y_max = float(hist["idx_0_100"].min()), float(hist["idx_0_100"].max())
        bounds = [-np.inf] + thresholds + [np.inf]
        colors = ["rgba(34,197,94,0.08)","rgba(245,158,11,0.10)","rgba(239,68,68,0.10)"]
        for i in range(len(bounds)-1):
            lo_plot = max(y_min, bounds[i] if np.isfinite(bounds[i]) else y_min)
            hi_plot = min(y_max, bounds[i+1] if np.isfinite(bounds[i+1]) else y_max)
            if hi_plot > lo_plot:
                fig.add_shape(type="rect", xref="paper", yref="y",
                              x0=0, x1=1, y0=lo_plot, y1=hi_plot,
                              fillcolor=colors[min(i, len(colors)-1)], line_width=0, layer="below")
        fig.update_layout(
            title="Histórico Diário — Índice de Anomalia (limiares aprendidos)",
            xaxis_title="Data", yaxis_title="Índice (0–100)",
            hovermode="x unified", template="plotly_white", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

def _color_text_zscore(z: float) -> str:
    """
    Coloração com base no Z-score.
    |z| = 0  → verde
    |z| = 3  → vermelho total
    """
    norm = min(abs(z) / 3, 1.0)  # 3 desvios = vermelho total

    r = int(255 * norm)
    g = int(200 * (1 - norm))
    b = int(80  * (1 - norm))

    return f"color: rgb({r},{g},{b}); font-weight:700;"

# =========================
# CONTRIBUINTES DO DIA (vs média histórica)
# =========================

# Média histórica
stats = (
    full[all_features]
    .agg(['mean'])
    .T.rename(columns={'mean': 'media'})
    .reset_index()
    .rename(columns={'index': 'variavel'})
)

# Valores do dia
valores = (
    row[all_features]
    .iloc[0]
    .to_frame("valor_atual")
    .reset_index()
    .rename(columns={"index": "variavel"})
)

contrib = pd.merge(valores, stats, on="variavel", how="left")

# Desvio percentual
contrib["dif_percentual"] = (
    (contrib["valor_atual"] - contrib["media"]) / contrib["media"] * 100
).round(2)

# Desvio padrão histórico
stds = full[all_features].std().replace(0, 1e-9)

# Z-score
contrib["z_score"] = (
    (contrib["valor_atual"] - contrib["media"]) / stds.values
)

# Ordenação por maior desvio
contrib = contrib.sort_values("z_score", key=lambda s: abs(s), ascending=False)

# Estilização
styled_contrib = (
    contrib.style.format({
        "valor_atual": "{:.2f}",
        "media": "{:.2f}",
        "dif_percentual": "{:+.2f}%",
        "z_score": "{:+.2f}"
    })
    .map(lambda z: _color_text_zscore(z), subset=["z_score"])
)

# =========================
# >>>>>> FUNÇÕES DE CRITICIDADE (para cor da fonte nas abas)
# =========================

def _color_text_font(level: float) -> str:
    """
    level ∈ [0,1]: 0 = mais normal (verde), 1 = mais anômalo (vermelho)
    """
    r = int(255 * level)
    g = int(190 * (1 - level))
    b = int(80  * (1 - level))
    return f"color: rgb({r},{g},{b}); font-weight:700;"

def _compute_variable_criticality(results: dict, row_df: pd.DataFrame, feature_subset: list) -> pd.Series:
    """
    Criticidade contextual por variável considerando:
      - valor do dia (padronizado pelo mesmo RobustScaler),
      - direção de influência (correlação de cada feature com os componentes do ensemble),
      - pesos dos componentes (best_weights).
    Retorna: Série em [0,1] por variável do subset.
    """
    corr = results["corr_features_components"].copy()     # index = features, cols = [stat, iso, lof, oc, ee]
    weights = results["best_weights"]
    # direção ponderada
    corr["weighted"] = (
        corr["stat"]*weights["stat"] + corr["iso"]*weights["iso"] +
        corr["lof"]*weights["lof"]   + corr["oc"]*weights["oc"]  +
        corr["ee"]*weights["ee"]
    )

    all_feats = corr.index.tolist()                       # MESMA ordem usada no scaler
    # pega 1 linha (o dia escolhido) com todas as features, padroniza
    current_vals = row_df[all_feats].astype(float).iloc[0].values.reshape(1, -1)
    scaled_all = results["scaler"].transform(current_vals)[0]

    # restringe ao subset (quali ou quanti)
    scaled_subset = pd.Series(scaled_all, index=all_feats).loc[feature_subset].values
    dir_subset    = corr.loc[feature_subset, "weighted"].values

    # influência = valor padronizado × direção ponderada (sinal)
    influence = scaled_subset * dir_subset

    # normalização min-max para [0,1] (0=verde, 1=vermelho)
    crit = (influence - influence.min()) / (influence.max() - influence.min() + 1e-9)
    return pd.Series(crit, index=feature_subset)

from openai import OpenAI

# inicialização correta
client = OpenAI(api_key="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")

# --------------------------
# CLASSIFICAÇÃO QUALITATIVA
# --------------------------
def classify_z(z):
    """
    Converte z-score em descrição qualitativa.
    """
    if abs(z) >= 3:
        return "extremamente anômalo"
    elif abs(z) >= 2:
        return "fortemente anômalo"
    elif abs(z) >= 1:
        return "levemente desviado"
    else:
        return "normal"


# --------------------------
# FUNÇÃO DO LLM
# --------------------------
def llm_daily_summary(day_index, day_imp, date, situation, vars_txt):

    prompt = f"""
Você é um especialista em hidrologia e análise ambiental.

Produza um resumo objetivo e contínuo, sem tópicos, sem listas e sem enumerações, descrevendo o estado do rio no dia **{date}** com base na situação e no comportamento qualitativo das variáveis fornecidas.

Regras obrigatórias:
- Utilize exclusivamente as informações fornecidas.
- O texto deve ser um único parágrafo contínuo, com 5 a 7 linhas.
- Se todas as variáveis estiverem normais, destaque de forma clara que o dia é totalmente estável e que não houve anomalias.
- Se houver variáveis desviadas, priorize apenas aquelas com maior nível de anomalia.
- Mencione explicitamente quando uma variável estiver normal ou sem impacto relevante.
- NÃO use listas, enumerações, tópicos, separações do tipo “Explicação:”.
- NÃO mencione valores numéricos, porcentagens, índices, z-scores ou causas externas.
- Mantenha o tom técnico, direto e descritivo.

Informações do dia:
- Situação: {situation}
- Variáveis monitoradas (ordenadas do maior desvio para o menor):
{vars_txt}

Agora escreva um resumo único e natural descrevendo o estado do rio no dia indicado.

"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Você é um analista ambiental especialista em rios e águas superficiais."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=260,
        temperature=0.25
    )

    return response.choices[0].message.content


# --------------------------
# PREPARAÇÃO DOS DADOS
# --------------------------

# Seleciona linha do dia
row = full.loc[full["date"] == sel_date]

# Obter z-scores e variáveis
contrib = row[all_features].iloc[0].to_frame("valor_atual").reset_index().rename(columns={"index": "variavel"})
media_df = full[all_features].mean()
std_df = full[all_features].std().replace(0, 1e-9)

contrib["media"] = contrib["variavel"].map(media_df)
contrib["z_score"] = (contrib["valor_atual"] - contrib["media"]) / std_df.values

# Classificação qualitativa
contrib["qualitativo"] = contrib["z_score"].apply(classify_z)

# Ordena pela anomalia (maior |z|)
contrib_sorted = contrib.sort_values("z_score", key=lambda s: abs(s), ascending=False)

# Texto enviado ao LLM
vars_txt = "\n".join(
    [f"- {row['variavel']}: {row['qualitativo']}" for _, row in contrib_sorted.iterrows()]
)

# Situação do dia
situation_today = row["nivel"].iloc[0]

# Dicionário das variáveis (não usado diretamente pelo LLM)
variables_today = row[all_features].iloc[0].to_dict()


# --------------------------
# CHAMADA DO LLM
# --------------------------
summary = llm_daily_summary(
    day_index=float(row["idx_0_100"].iloc[0]),
    day_imp=variables_today,
    date=sel_date.strftime("%d/%m/%Y"),
    situation=situation_today,
    vars_txt=vars_txt
)


# --------------------------
# EXIBIÇÃO NO STREAMLIT
# --------------------------
st.subheader("🧠 Resumo do Dia (IA)")
st.write(summary)

# =========================
# ABAS QUALI / QUANTI / EXPLICAÇÃO DO DIA
# =========================
tabs = st.tabs(["Quali (qualidade da água)", "Quanti (nível/vazão)"])

with tabs[0]:
    quali_day = row[quali_cols].copy()
    quali_day.index = [sel_date]

    # Média e desvio padrão históricos
    media_quali = full[quali_cols].mean()
    std_quali   = full[quali_cols].std().replace(0, 1e-9)

    # Z-score por variável
    z_quali = (quali_day.iloc[0] - media_quali) / std_quali

    styled_quali = quali_day.style.format("{:.2f}")

    # Aplica coloração pelo Z-score
    for var in quali_cols:
        z = float(z_quali[var])
        styled_quali = styled_quali.map(lambda v, z=z: _color_text_zscore(z), subset=[var])

    st.dataframe(styled_quali, use_container_width=True)
    st.caption("Cor baseada no Z-score (0 = normal, 3+ = muito anômalo).")



with tabs[1]:
    quanti_day = row[quanti_cols].copy()
    quanti_day.index = [sel_date]

    # Média e desvio padrão históricos
    media_quanti = full[quanti_cols].mean()
    std_quanti   = full[quanti_cols].std().replace(0, 1e-9)

    # Z-score por variável
    z_quanti = (quanti_day.iloc[0] - media_quanti) / std_quanti

    styled_quanti = quanti_day.style.format("{:.2f}")

    # Aplica coloração pelo Z-score
    for var in quanti_cols:
        z = float(z_quanti[var])
        styled_quanti = styled_quanti.map(lambda v, z=z: _color_text_zscore(z), subset=[var])

    st.dataframe(styled_quanti, use_container_width=True)
    st.caption("Quanto mais 'normal', mais verde; Quanto mais 'crítico', mais vermelho")



