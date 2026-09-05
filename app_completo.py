import json

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from shapely.geometry import Point, shape
from shapely.strtree import STRtree

# ==========================================================
# Configuración inicial
# ==========================================================
st.set_page_config(page_title="Arbolado Urbano - Corrientes Capital", layout="wide", page_icon="🌳")

# ==========================================================
# Estética: paleta de alto impacto para exposición (tema oscuro)
# ==========================================================
PALETA = {
    "fondo": "#0E1F17",
    "fondo_secundario": "#153726",
    "verde_neon": "#00E676",
    "verde_oscuro": "#00C853",
    "amarillo": "#FFD600",
    "naranja": "#FF6D00",
    "rojo": "#FF1744",
    "azul": "#00B0FF",
    "texto": "#F1FFF6",
}

# Paleta de colores vivos para gráficos multi-categoría (chi2, MCA)
PALETA_VIVA = ["#00E676", "#00B0FF", "#FFD600", "#FF1744", "#D500F9", "#FF6D00", "#1DE9B6"]

st.markdown(
    f"""
    <style>
    /* Encabezado con degradé de alto contraste */
    .encabezado-app {{
        background: linear-gradient(90deg, #003D1F 0%, #00A152 55%, #00E676 100%);
        padding: 1.6rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.4rem;
        box-shadow: 0 4px 20px rgba(0, 230, 118, 0.25);
    }}
    .encabezado-app h1 {{
        color: white !important;
        margin: 0;
        font-size: 2.2rem;
        font-weight: 800;
        text-shadow: 0 2px 6px rgba(0,0,0,0.35);
    }}
    .encabezado-app p {{
        color: #E8FFF0 !important;
        margin-top: 0.5rem;
        margin-bottom: 0;
        font-size: 1rem;
    }}

    /* Tarjetas de métricas (KPIs) con look "neón" */
    div[data-testid="stMetric"] {{
        background-color: {PALETA['fondo_secundario']};
        border: 1px solid rgba(0, 230, 118, 0.35);
        border-left: 6px solid {PALETA['verde_neon']};
        border-radius: 12px;
        padding: 1rem 1.1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.25);
    }}
    div[data-testid="stMetricValue"] {{
        color: {PALETA['verde_neon']} !important;
        font-size: 1.9rem !important;
        font-weight: 800 !important;
    }}
    div[data-testid="stMetricLabel"] {{
        color: {PALETA['texto']} !important;
        opacity: 0.85;
    }}

    /* Pestañas grandes y bien visibles */
    button[data-baseweb="tab"] {{
        font-weight: 700;
        font-size: 1.02rem;
    }}
    button[data-baseweb="tab"][aria-selected="true"] {{
        color: {PALETA['verde_neon']} !important;
        border-bottom: 3px solid {PALETA['verde_neon']} !important;
    }}

    /* Subtítulos de sección con acento neón */
    h3 {{
        color: {PALETA['verde_neon']};
        border-bottom: 3px solid {PALETA['amarillo']};
        padding-bottom: 0.35rem;
        font-weight: 800;
    }}

    /* Alertas (success/warning/info) más vívidas */
    div[data-testid="stAlertContentSuccess"] {{
        font-size: 1.05rem;
    }}
    </style>

    <div class="encabezado-app">
        <h1>🌳 Análisis Integral del Arbolado Urbano — Corrientes Capital</h1>
        <p>Tablero basado en arboles.csv, Seguimiento_Arboles.csv y barrios_de_la_ciudad.csv
        (Práctica Profesionalizante II - 2026).</p>
    </div>
    """,
    unsafe_allow_html=True,
)


def estilo_oscuro(fig, ax_o_axes):
    """Aplica el tema oscuro a una figura de matplotlib para que combine con la app."""
    fig.patch.set_facecolor(PALETA["fondo"])
    ejes = ax_o_axes if isinstance(ax_o_axes, (list, np.ndarray)) else [ax_o_axes]
    for ax in ejes:
        ax.set_facecolor(PALETA["fondo_secundario"])
        ax.title.set_color(PALETA["texto"])
        ax.xaxis.label.set_color(PALETA["texto"])
        ax.yaxis.label.set_color(PALETA["texto"])
        ax.tick_params(colors=PALETA["texto"])
        for spine in ax.spines.values():
            spine.set_color(PALETA["texto"])
        leg = ax.get_legend()
        if leg is not None:
            leg.get_frame().set_facecolor(PALETA["fondo_secundario"])
            for text in leg.get_texts():
                text.set_color(PALETA["texto"])
    return fig


# ==========================================================
# 1. Carga y preparación de datos
# ==========================================================
@st.cache_data
def cargar_datos(archivo_arboles, archivo_mantenimiento, archivo_barrios):
    arboles = pd.read_csv(archivo_arboles)
    mantenimiento = pd.read_csv(archivo_mantenimiento)
    barrios = pd.read_csv(archivo_barrios)

    # --- Unión espacial árbol -> barrio (equivalente a gpd.sjoin del notebook) ---
    barrios = barrios.copy()
    barrios["geometry"] = barrios["the_geom"].apply(lambda g: shape(json.loads(g)))

    poligonos = list(barrios["geometry"])
    tree_idx = STRtree(poligonos)
    geom_a_barrio = {id(geom): nombre for geom, nombre in zip(poligonos, barrios["nombre_barrio"])}

    def encontrar_barrio(row):
        punto = Point(row["lng"], row["lat"])
        for idx in tree_idx.query(punto):
            geom = poligonos[idx]
            if geom.contains(punto):
                return geom_a_barrio[id(geom)]
        return None

    arboles = arboles.copy()
    arboles["nombre_barrio"] = arboles.apply(encontrar_barrio, axis=1)

    # --- df final: arboles + mantenimiento + barrio ---
    df = pd.merge(arboles, mantenimiento, on="id_arbol", how="left")

    return arboles, mantenimiento, barrios, df


st.sidebar.header("📂 Datos de entrada")
modo = st.sidebar.radio("Origen de los datos", ["Usar archivos locales del repo", "Subir archivos"])

if modo == "Subir archivos":
    file_arboles = st.sidebar.file_uploader("arboles.csv", type="csv")
    file_mantenimiento = st.sidebar.file_uploader("Seguimiento_Arboles.csv", type="csv")
    file_barrios = st.sidebar.file_uploader("barrios_de_la_ciudad.csv", type="csv")
else:
    file_arboles = "arboles.csv"
    file_mantenimiento = "Seguimiento_Arboles.csv"
    file_barrios = "barrios_de_la_ciudad.csv"

if not (file_arboles and file_mantenimiento and file_barrios):
    st.info("⬅️ Cargá los tres archivos CSV desde la barra lateral para ver el tablero.")
    st.stop()

try:
    arboles, mantenimiento, barrios, df = cargar_datos(file_arboles, file_mantenimiento, file_barrios)
except Exception as e:
    st.error(f"No se pudieron cargar/procesar los datos: {e}")
    st.stop()


# ==========================================================
# 2. Filtros globales
# ==========================================================
st.sidebar.header("🔎 Filtros globales")
barrios_disponibles = sorted(arboles["nombre_barrio"].dropna().unique())
barrios_sel = st.sidebar.multiselect("Barrios a incluir", options=barrios_disponibles, default=barrios_disponibles)

arboles_f = arboles[arboles["nombre_barrio"].isin(barrios_sel)]
df_f = df[df["nombre_barrio"].isin(barrios_sel)]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Árboles inventariados", f"{len(arboles_f):,}")
col2.metric("Especies distintas", f"{arboles_f['especie'].nunique()}")
col3.metric("Barrios seleccionados", f"{len(barrios_sel)}")
col4.metric("Registros de mantenimiento", f"{df_f['id_seguimiento'].notna().sum():,}")

st.divider()

tabs = st.tabs(
    [
        "📋 EDA general",
        "🗺️ Distribución espacial",
        "🌳 Barrios con más árboles",
        "🛠️ Tipos de mantenimiento",
        "🧭 Zona Norte / Sur",
        "📊 Riesgo vs. otras variables",
        "🚨 Necesidad de intervención",
        "🔬 Análisis MCA",
    ]
)

# ----------------------------------------------------------
# TAB 0: EDA general
# ----------------------------------------------------------
with tabs[0]:
    st.subheader("Exploración general del arbolado")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Especies más frecuentes**")
        top_especies = arboles_f["especie"].value_counts().head(10)
        st.bar_chart(top_especies, color=PALETA["verde_neon"])
    with c2:
        st.markdown("**Tipo de vereda**")
        st.bar_chart(arboles_f["tipo_vereda"].value_counts(), color=PALETA["azul"])

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**Lado de vereda**")
        st.bar_chart(arboles_f["lado_vereda"].value_counts(), color=PALETA["amarillo"])
    with c4:
        st.markdown("**Árboles activos**")
        st.bar_chart(arboles_f["activo"].value_counts(), color=PALETA["verde_oscuro"])

    if st.checkbox("Ver estadísticas descriptivas completas (describe)"):
        st.dataframe(arboles_f.describe(include="all"))

    st.caption(
        f"Duplicados detectados: {arboles_f.duplicated().sum()} · "
        f"Valores nulos totales: {int(arboles_f.isna().sum().sum())}"
    )

# ----------------------------------------------------------
# TAB 1: Distribución espacial
# ----------------------------------------------------------
with tabs[1]:
    st.subheader("Distribución espacial del arbolado urbano")

    st.map(
        arboles_f.rename(columns={"lat": "latitude", "lng": "longitude"})[["latitude", "longitude"]],
        size=3,
    )

    st.markdown("**Mapa de densidad (equivalente al heatmap del notebook)**")
    fig, ax = plt.subplots(figsize=(8, 7))
    hb = ax.hexbin(arboles_f["lng"], arboles_f["lat"], gridsize=40, cmap="viridis", mincnt=1)
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.set_title("Densidad de árboles por zona")
    cbar = fig.colorbar(hb, ax=ax, label="Cantidad de árboles")
    cbar.ax.yaxis.label.set_color(PALETA["texto"])
    cbar.ax.tick_params(colors=PALETA["texto"])
    estilo_oscuro(fig, ax)
    st.pyplot(fig)

# ----------------------------------------------------------
# TAB 2: Barrios con más árboles
# ----------------------------------------------------------
with tabs[2]:
    st.subheader("¿Qué barrios tienen más árboles plantados?")

    top_n = st.slider("Cantidad de barrios a mostrar", 5, 30, 15, key="topn_barrios")
    arboles_por_barrio = arboles_f["nombre_barrio"].value_counts().head(top_n)
    st.bar_chart(arboles_por_barrio, color=PALETA["verde_neon"])

    if not arboles_por_barrio.empty:
        st.success(
            f"El barrio con más árboles plantados es **{arboles_por_barrio.index[0]}** "
            f"con **{arboles_por_barrio.iloc[0]}** árboles."
        )

    if st.checkbox("Ver tabla completa por barrio"):
        st.dataframe(
            arboles_f["nombre_barrio"].value_counts().rename_axis("Barrio").reset_index(name="Cantidad de árboles")
        )

# ----------------------------------------------------------
# TAB 3: Tipos de mantenimiento
# ----------------------------------------------------------
with tabs[3]:
    st.subheader("¿Qué tipos de mantenimiento aparecen con mayor frecuencia?")

    mant_f = mantenimiento[mantenimiento["id_arbol"].isin(arboles_f["id_arbol"])]
    frecuencia = mant_f["tipo_seguimiento"].value_counts()
    porcentaje = (frecuencia / frecuencia.sum() * 100).round(2)

    st.bar_chart(frecuencia, color=PALETA["amarillo"])

    if not frecuencia.empty:
        st.success(
            f"El tipo de mantenimiento más frecuente es **'{frecuencia.index[0]}'** "
            f"con **{frecuencia.iloc[0]}** registros ({porcentaje.iloc[0]:.1f}% del total)."
        )

    if st.checkbox("Ver tabla de frecuencia y porcentaje"):
        st.dataframe(pd.DataFrame({"Frecuencia": frecuencia, "Porcentaje": porcentaje}))

# ----------------------------------------------------------
# TAB 4: Zona Norte / Sur
# ----------------------------------------------------------
with tabs[4]:
    st.subheader("Distribución del arbolado por zona (Norte / Sur)")
    st.caption(
        "Se usa como línea divisoria la latitud del cruce Av. 3 de Abril y Rioja "
        "(límite sur del casco histórico), igual que en el notebook."
    )

    lat_divisoria = st.slider("Latitud divisoria", -27.50, -27.44, -27.472, step=0.001, format="%.3f")

    df_zona = df_f.copy()
    df_zona["zona"] = df_zona["lat"].apply(lambda x: "Zona Norte" if x >= lat_divisoria else "Zona Sur")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Cantidad de árboles por zona**")
        st.bar_chart(df_zona["zona"].value_counts(), color=PALETA["verde_neon"])

    with c2:
        fig, ax = plt.subplots(figsize=(6, 5))
        colores = {"Zona Norte": PALETA["verde_neon"], "Zona Sur": PALETA["naranja"]}
        for zona, color in colores.items():
            datos = df_zona[df_zona["zona"] == zona]
            ax.scatter(datos["lng"], datos["lat"], s=6, alpha=0.5, label=zona, color=color)
        ax.axhline(y=lat_divisoria, color=PALETA["amarillo"], linestyle="--", linewidth=2, label="Línea divisoria")
        ax.set_xlabel("Longitud")
        ax.set_ylabel("Latitud")
        ax.legend()
        estilo_oscuro(fig, ax)
        st.pyplot(fig)

    variable_zona = st.selectbox(
        "Ver distribución de zona vs.",
        ["riesgo", "estado_salud", "levantamiento_vereda", "ahuecamiento", "inclinacion"],
    )
    st.dataframe(pd.crosstab(df_zona["zona"], df_zona[variable_zona]))

# ----------------------------------------------------------
# TAB 5: Riesgo vs. otras variables (bivariado + chi2)
# ----------------------------------------------------------
with tabs[5]:
    st.subheader("Análisis bivariado: riesgo según otras variables")

    variable = st.selectbox(
        "Elegí la variable a cruzar con 'riesgo'",
        ["estado_salud", "inclinacion", "ahuecamiento", "levantamiento_vereda", "fase_vital"],
    )

    tabla_cruzada = pd.crosstab(df_f["riesgo"], df_f[variable])
    st.dataframe(tabla_cruzada)

    fig, ax = plt.subplots(figsize=(8, 5))
    tabla_cruzada.plot(kind="bar", stacked=True, ax=ax, color=PALETA_VIVA[: len(tabla_cruzada.columns)])
    ax.set_title(f"Riesgo según {variable}")
    ax.set_xlabel("Nivel de riesgo")
    ax.set_ylabel("Cantidad de árboles")
    estilo_oscuro(fig, ax)
    st.pyplot(fig)

    try:
        chi2, p, gl, esperados = chi2_contingency(tabla_cruzada)
        st.markdown(f"**Chi-cuadrado:** {chi2:.2f}  |  **Grados de libertad:** {gl}  |  **Valor p:** {p:.6f}")
        if p < 0.05:
            st.info(
                f"Como el valor p ({p:.6f}) es menor a 0.05, se rechaza la hipótesis nula de independencia: "
                f"existe una asociación estadísticamente significativa entre **riesgo** y **{variable}**."
            )
        else:
            st.info(
                f"Como el valor p ({p:.6f}) no es menor a 0.05, no hay evidencia suficiente de asociación "
                f"entre **riesgo** y **{variable}**."
            )
    except Exception as e:
        st.warning(f"No se pudo calcular el chi-cuadrado para esta combinación: {e}")

# ----------------------------------------------------------
# TAB 6: Necesidad de intervención por barrio
# ----------------------------------------------------------
with tabs[6]:
    st.subheader("¿Hay barrios con mayor necesidad de intervención?")

    min_arboles = st.slider("Mínimo de árboles por barrio para considerarlo", 0, 100, 30)

    riesgo_alto_col = "Con riesgo de caída (alto)"
    estados_criticos = ["Malo", "Muerto"]

    resumen = pd.DataFrame({"total_arboles": df_f.groupby("nombre_barrio")["id_arbol"].count()})
    resumen["riesgo_alto"] = (
        df_f[df_f["riesgo"] == riesgo_alto_col].groupby("nombre_barrio")["id_arbol"].count()
    )
    resumen["riesgo_alto"] = resumen["riesgo_alto"].fillna(0).astype(int)
    resumen["pct_riesgo_alto"] = (resumen["riesgo_alto"] / resumen["total_arboles"] * 100).round(2)

    resumen["estado_critico"] = (
        df_f[df_f["estado_salud"].isin(estados_criticos)].groupby("nombre_barrio")["id_arbol"].count()
    )
    resumen["estado_critico"] = resumen["estado_critico"].fillna(0).astype(int)
    resumen["pct_estado_critico"] = (resumen["estado_critico"] / resumen["total_arboles"] * 100).round(2)

    resumen["indice_intervencion"] = (resumen["pct_riesgo_alto"] + resumen["pct_estado_critico"]) / 2

    resumen_final = resumen[resumen["total_arboles"] >= min_arboles].sort_values(
        "indice_intervencion", ascending=False
    )

    top_n_interv = st.slider("Cantidad de barrios a mostrar", 5, 30, 15, key="topn_interv")
    top_interv = resumen_final.head(top_n_interv)
    st.bar_chart(top_interv["indice_intervencion"], color=PALETA["rojo"])

    if not top_interv.empty:
        st.warning(
            f"El barrio con mayor necesidad de intervención es **{top_interv.index[0]}**, "
            f"con un índice de {top_interv['indice_intervencion'].iloc[0]:.1f}% "
            f"({top_interv['pct_riesgo_alto'].iloc[0]:.1f}% en riesgo alto y "
            f"{top_interv['pct_estado_critico'].iloc[0]:.1f}% en estado crítico)."
        )

    if st.checkbox("Ver tabla completa de intervención por barrio"):
        st.dataframe(resumen_final)

# ----------------------------------------------------------
# TAB 7: Análisis de Correspondencias Múltiples (MCA)
# ----------------------------------------------------------
with tabs[7]:
    st.subheader("Análisis de Correspondencias Múltiples (MCA)")
    st.caption(
        "Visualiza conjuntamente las variables categóricas relacionadas con el estado y riesgo del arbolado."
    )

    try:
        import prince

        variables_mca = df_f[
            ["riesgo", "estado_salud", "inclinacion", "ahuecamiento", "levantamiento_vereda", "fase_vital"]
        ].dropna()

        if len(variables_mca) < 10:
            st.info("No hay suficientes registros con los filtros actuales para calcular el MCA.")
        else:
            mca = prince.MCA(n_components=2, random_state=42)
            mca = mca.fit(variables_mca)

            coord = mca.column_coordinates(variables_mca)
            varianza = mca.eigenvalues_summary if hasattr(mca, "eigenvalues_summary") else None

            colores_mca = {
                "riesgo": PALETA["rojo"],
                "estado_salud": PALETA["azul"],
                "inclinacion": PALETA["verde_neon"],
                "ahuecamiento": PALETA["amarillo"],
                "levantamiento_vereda": "#D500F9",
                "fase_vital": PALETA["naranja"],
            }

            fig, ax = plt.subplots(figsize=(10, 8))
            for nombre in coord.index:
                var_base = nombre.split("__")[0] if "__" in nombre else nombre
                color = colores_mca.get(var_base, PALETA["texto"])
                ax.scatter(coord.loc[nombre, 0], coord.loc[nombre, 1], color=color, s=70, edgecolors="white", linewidths=0.5)
                ax.text(coord.loc[nombre, 0] + 0.02, coord.loc[nombre, 1] + 0.02, nombre, fontsize=8, color=PALETA["texto"])

            ax.axhline(0, color=PALETA["texto"], linestyle="--", alpha=0.4)
            ax.axvline(0, color=PALETA["texto"], linestyle="--", alpha=0.4)
            ax.set_xlabel("Dimensión 1")
            ax.set_ylabel("Dimensión 2")
            ax.set_title("MCA - Categorías de variables del arbolado")
            estilo_oscuro(fig, ax)
            st.pyplot(fig)

            st.markdown("**Autovalores (varianza explicada):**")
            st.write(mca.eigenvalues_)

    except ImportError:
        st.warning(
            "La librería `prince` no está instalada en este entorno. "
            "Agregá `prince` a requirements.txt para habilitar esta pestaña."
        )
    except Exception as e:
        st.error(f"No se pudo calcular el MCA: {e}")
