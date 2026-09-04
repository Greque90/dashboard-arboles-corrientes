import json

import pandas as pd
import streamlit as st
from shapely.geometry import Point, shape
from shapely.strtree import STRtree

# ==========================================================
# Configuración inicial de la página
# ==========================================================
st.set_page_config(page_title="Arbolado Urbano - Corrientes Capital", layout="wide")

st.title("🌳 Tablero de Control del Arbolado Urbano — Corrientes Capital")
st.caption(
    "Basado en el análisis de arboles.csv, Seguimiento_Arboles.csv y "
    "barrios_de_la_ciudad.csv. Responde a: ¿qué barrios tienen más árboles "
    "plantados?, ¿qué tipos de mantenimiento son más frecuentes?, "
    "¿qué barrios necesitan intervención prioritaria?"
)


# ==========================================================
# 1. Carga y preparación de datos (cacheada)
# ==========================================================
@st.cache_data
def cargar_datos(archivo_arboles, archivo_mantenimiento, archivo_barrios):
    arboles = pd.read_csv(archivo_arboles)
    mantenimiento = pd.read_csv(archivo_mantenimiento)
    barrios = pd.read_csv(archivo_barrios)

    # --- Unión espacial árbol -> barrio, con shapely puro (equivalente al gpd.sjoin del notebook) ---
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

    # --- df final: mantenimiento + características del árbol + barrio ---
    # (mismo criterio que el notebook: arboles + mantenimiento + nombre_barrio)
    df = pd.merge(arboles, mantenimiento, on="id_arbol", how="left")

    return arboles, mantenimiento, barrios, df


# ==========================================================
# 2. Origen de datos
# ==========================================================
st.sidebar.header("📂 Datos de entrada")

modo = st.sidebar.radio("Origen de los datos", ["Usar archivos locales del repo", "Subir archivos"])

if modo == "Subir archivos":
    file_arboles = st.sidebar.file_uploader("arboles.csv", type="csv")
    file_mantenimiento = st.sidebar.file_uploader("Seguimiento_Arboles.csv", type="csv")
    file_barrios = st.sidebar.file_uploader("barrios_de_la_ciudad.csv", type="csv")
else:
    # Ajustá estas rutas/nombres según cómo subas los CSV a tu repo de GitHub
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
# 3. Filtros en la barra lateral
# ==========================================================
st.sidebar.header("🔎 Filtros")

barrios_disponibles = sorted(arboles["nombre_barrio"].dropna().unique())
barrios_sel = st.sidebar.multiselect(
    "Barrios a incluir",
    options=barrios_disponibles,
    default=barrios_disponibles,
)

arboles_f = arboles[arboles["nombre_barrio"].isin(barrios_sel)]
df_f = df[df["nombre_barrio"].isin(barrios_sel)]


# ==========================================================
# 4. KPIs generales
# ==========================================================
col1, col2, col3, col4 = st.columns(4)
col1.metric("Árboles inventariados", f"{len(arboles_f):,}")
col2.metric("Especies distintas", f"{arboles_f['especie'].nunique()}")
col3.metric("Barrios seleccionados", f"{len(barrios_sel)}")
col4.metric("Registros de mantenimiento", f"{df_f['id_seguimiento'].notna().sum():,}")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🌳 Barrios con más árboles",
        "🛠️ Tipos de mantenimiento",
        "🚨 Necesidad de intervención",
        "🗺️ Mapa / EDA general",
    ]
)

# ----------------------------------------------------------
# TAB 1: ¿Qué barrios tienen más árboles plantados?
# ----------------------------------------------------------
with tab1:
    st.subheader("¿Qué barrios tienen más árboles plantados?")

    top_n = st.slider("Cantidad de barrios a mostrar", 5, 30, 15, key="topn_barrios")

    arboles_por_barrio = arboles_f["nombre_barrio"].value_counts().head(top_n)
    st.bar_chart(arboles_por_barrio)

    if not arboles_por_barrio.empty:
        st.success(
            f"El barrio con más árboles plantados es **{arboles_por_barrio.index[0]}** "
            f"con **{arboles_por_barrio.iloc[0]}** árboles."
        )

    col_a, col_b = st.columns(2)
    with col_a:
        if st.checkbox("Ver tabla completa por barrio"):
            st.dataframe(
                arboles_f["nombre_barrio"]
                .value_counts()
                .rename_axis("Barrio")
                .reset_index(name="Cantidad de árboles")
            )
    with col_b:
        if st.checkbox("Ver especies más frecuentes"):
            st.dataframe(
                arboles_f["especie"]
                .value_counts()
                .head(10)
                .rename_axis("Especie")
                .reset_index(name="Cantidad")
            )

# ----------------------------------------------------------
# TAB 2: ¿Qué tipos de mantenimiento aparecen con mayor frecuencia?
# ----------------------------------------------------------
with tab2:
    st.subheader("¿Qué tipos de mantenimiento aparecen con mayor frecuencia?")

    mant_f = mantenimiento[mantenimiento["id_arbol"].isin(arboles_f["id_arbol"])]

    frecuencia = mant_f["tipo_seguimiento"].value_counts()
    porcentaje = (frecuencia / frecuencia.sum() * 100).round(2)

    st.bar_chart(frecuencia)

    if not frecuencia.empty:
        st.success(
            f"El tipo de mantenimiento más frecuente es **'{frecuencia.index[0]}'** "
            f"con **{frecuencia.iloc[0]}** registros ({porcentaje.iloc[0]:.1f}% del total)."
        )

    if st.checkbox("Ver tabla de frecuencia y porcentaje"):
        tabla = pd.DataFrame({"Frecuencia": frecuencia, "Porcentaje": porcentaje})
        st.dataframe(tabla)

# ----------------------------------------------------------
# TAB 3: ¿Hay barrios con mayor necesidad de intervención?
# ----------------------------------------------------------
with tab3:
    st.subheader("¿Hay barrios con mayor necesidad de intervención?")

    min_arboles = st.slider(
        "Mínimo de árboles por barrio para considerarlo (evita % poco representativos)",
        0, 100, 30,
    )

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

    st.bar_chart(top_interv["indice_intervencion"])

    if not top_interv.empty:
        st.warning(
            f"El barrio con mayor necesidad de intervención es **{top_interv.index[0]}**, "
            f"con un índice de {top_interv['indice_intervencion'].iloc[0]:.1f}% "
            f"({top_interv['pct_riesgo_alto'].iloc[0]:.1f}% en riesgo alto y "
            f"{top_interv['pct_estado_critico'].iloc[0]:.1f}% en estado crítico)."
        )

    st.caption(
        "El índice de intervención combina el % de árboles con riesgo de caída alto "
        "y el % en estado sanitario crítico (Malo/Muerto). Se filtran barrios con "
        "menos árboles que el mínimo elegido para evitar porcentajes poco representativos."
    )

    if st.checkbox("Ver tabla completa de intervención por barrio"):
        st.dataframe(resumen_final)

# ----------------------------------------------------------
# TAB 4: Mapa / EDA general
# ----------------------------------------------------------
with tab4:
    st.subheader("Distribución espacial del arbolado")

    st.map(
        arboles_f.rename(columns={"lat": "latitude", "lng": "longitude"})[["latitude", "longitude"]],
        size=3,
    )

    st.caption(
        f"Se muestran {len(arboles_f):,} árboles según los barrios seleccionados en el filtro."
    )
