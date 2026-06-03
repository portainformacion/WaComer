from pathlib import Path
from collections import Counter
import os
import re
import unicodedata

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import pydeck as pdk

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except Exception:
    WORDCLOUD_AVAILABLE = False


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

st.set_page_config(
    page_title="WaComer",
    page_icon="🍽️",
    layout="wide"
)

st.title("🍽️ WaComer")
st.subheader("Recomendador de restaurantes cercanos usando reseñas y PLN")

st.markdown(
    """
    **WaComer** recomienda restaurantes cercanos en Lima usando reseñas reales y técnicas de
    Procesamiento de Lenguaje Natural.

    La recomendación considera:

    - ubicación del usuario;
    - distancia al restaurante;
    - sentimiento inferido desde el texto de las reseñas;
    - cantidad de reseñas disponibles;
    - tema más frecuente mencionado por los clientes.

    **No se usa el rating numérico para calcular el sentimiento ni para recomendar.**
    """
)


# ============================================================
# RUTA DE DATOS
# ============================================================

RAW_DIR = Path("data") / "raw"


# ============================================================
# DICCIONARIOS PLN
# ============================================================

palabras_positivas = {
    "bueno", "buena", "buenos", "buenas",
    "excelente", "perfecto", "perfecta", "genial",
    "increible", "maravilloso", "maravillosa",
    "agradable", "recomendado", "recomendada", "recomiendo",
    "favorito", "favorita", "calidad", "satisfecho", "satisfecha",
    "rico", "rica", "ricos", "ricas",
    "delicioso", "deliciosa", "deliciosos", "deliciosas",
    "sabroso", "sabrosa", "sabrosos", "sabrosas",
    "fresco", "fresca", "frescos", "frescas",
    "jugoso", "jugosa", "crocante", "crujiente",
    "caliente", "suave", "contundente",
    "exquisito", "exquisita", "espectacular",
    "amable", "amables", "atento", "atenta", "atentos", "atentas",
    "cordial", "servicial", "rapido", "rapida", "rapidos", "rapidas",
    "eficiente", "paciente", "educado", "educada",
    "limpio", "limpia", "limpios", "limpias",
    "comodo", "comoda", "comodos", "comodas",
    "bonito", "bonita", "bonitos", "bonitas",
    "tranquilo", "tranquila", "acogedor", "acogedora",
    "ordenado", "ordenada",
    "barato", "barata", "economico", "economica",
    "justo", "justa", "accesible", "promocion"
}

palabras_negativas = {
    "malo", "mala", "malos", "malas",
    "pesimo", "pesima", "horrible", "terrible",
    "decepcion", "decepcionante", "fatal",
    "deficiente", "regular", "mediocre",
    "frio", "fria", "frios", "frias",
    "quemado", "quemada", "quemados", "quemadas",
    "crudo", "cruda", "crudos", "crudas",
    "salado", "salada", "salados", "saladas",
    "insipido", "insipida", "duro", "dura",
    "grasoso", "grasosa", "seco", "seca",
    "feo", "fea", "malogrado", "malograda",
    "pequeño", "pequena", "escaso", "escasa",
    "lento", "lenta", "lentos", "lentas",
    "demora", "demoro", "demoraron", "demorado", "demorada",
    "tardo", "tardaron", "espera", "cola",
    "grosero", "grosera", "malcriado", "malcriada",
    "desatento", "desatenta", "indiferente",
    "sucio", "sucia", "sucios", "sucias",
    "ruidoso", "ruidosa", "incomodo", "incomoda",
    "desordenado", "desordenada", "oscuro", "oscura",
    "caluroso", "calurosa",
    "caro", "cara", "caros", "caras",
    "costoso", "costosa", "excesivo", "excesiva",
    "sobrevalorado", "sobrevalorada",
    "incompleto", "incompleta", "derramado", "derramada",
    "roto", "rota", "equivocado", "equivocada",
    "reclamo", "queja", "problema", "problemas"
}

negaciones = {"no", "nunca", "jamas", "tampoco", "ni"}

intensificadores = {
    "muy", "demasiado", "super", "bastante", "tan", "re", "sumamente"
}

temas_keywords = {
    "comida": [
        "comida", "sabor", "rico", "rica", "delicioso", "deliciosa",
        "plato", "menu", "porcion", "pollo", "ceviche", "carne",
        "arroz", "pasta", "postre", "frio", "caliente", "salado",
        "dulce", "sopa", "ensalada", "hamburguesa", "pizza", "cafe",
        "chifa", "anticucho", "parrilla", "criollo"
    ],
    "atencion": [
        "atencion", "mozo", "moza", "mesero", "mesera", "personal",
        "amable", "trato", "atienden", "atendieron", "cordial",
        "servicial", "cajero", "cajera", "recepcion", "atento"
    ],
    "precio": [
        "precio", "caro", "barato", "costoso", "cuenta",
        "pagar", "costo", "promocion", "oferta", "economico",
        "excesivo", "sobrevalorado"
    ],
    "demora": [
        "demora", "demoraron", "demoro", "tardo", "tardaron",
        "espera", "lento", "lenta", "cola", "rapido", "rapida",
        "tiempo"
    ],
    "ambiente": [
        "ambiente", "local", "musica", "ruido", "limpio", "sucio",
        "comodo", "comodidad", "decoracion", "terraza", "espacio",
        "baño", "bano", "mesa", "silla", "vista"
    ],
    "delivery": [
        "delivery", "pedido", "repartidor", "llego", "empaque",
        "derramado", "domicilio", "entrega", "app"
    ]
}


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def cargar_csv_seguro(ruta):
    try:
        return pd.read_csv(ruta)
    except UnicodeDecodeError:
        return pd.read_csv(ruta, encoding="latin1")


def quitar_tildes(texto):
    texto = unicodedata.normalize("NFD", str(texto))
    texto = texto.encode("ascii", "ignore")
    texto = texto.decode("utf-8")
    return texto


def limpiar_texto(texto):
    if pd.isna(texto):
        return ""

    texto = str(texto).lower()
    texto = quitar_tildes(texto)
    texto = re.sub(r"http\S+|www\S+", " ", texto)
    texto = re.sub(r"\d+", " ", texto)
    texto = re.sub(r"[^a-zñ\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()

    return texto


def detectar_archivos_csv(raw_dir):
    if not raw_dir.exists():
        raise FileNotFoundError(
            "No existe la carpeta data/raw. Crea data/raw y coloca ahí tus dos CSV."
        )

    csv_files = [f for f in os.listdir(raw_dir) if f.lower().endswith(".csv")]

    if len(csv_files) < 2:
        raise ValueError(
            f"Se esperaban al menos 2 archivos CSV en data/raw. Encontrados: {csv_files}"
        )

    tablas = {}

    for archivo in csv_files:
        ruta = raw_dir / archivo
        tablas[archivo] = cargar_csv_seguro(ruta)

    return tablas


def score_tabla_reviews(df):
    columnas = [c.lower() for c in df.columns]
    score = 0

    for c in columnas:
        if c == "caption":
            score += 20
        if "review" in c:
            score += 4
        if "comment" in c:
            score += 4
        if "comentario" in c:
            score += 4
        if "text" in c:
            score += 3
        if "resena" in c or "reseña" in c:
            score += 4
        if "id_review" in c:
            score += 5

    score += min(df.shape[0] / 100000, 5)

    return score


def score_tabla_restaurantes(df):
    columnas = [c.lower() for c in df.columns]
    score = 0

    for c in columnas:
        if c == "title":
            score += 20
        if c == "lat":
            score += 10
        if c == "long":
            score += 10
        if "place" in c:
            score += 2
        if "name" in c:
            score += 2
        if "nombre" in c:
            score += 2
        if "restaurant" in c:
            score += 2
        if "lat" in c:
            score += 4
        if "lon" in c or "lng" in c or "long" in c:
            score += 4

    return score


def analizar_sentimiento_diccionario(texto):
    texto_limpio = limpiar_texto(texto)
    tokens = texto_limpio.split()

    if len(tokens) == 0:
        return pd.Series({
            "sentimiento_pln": "sin_texto",
            "score_sentimiento": 0.0,
            "positivas_detectadas": "",
            "negativas_detectadas": ""
        })

    score = 0.0
    positivas_detectadas = []
    negativas_detectadas = []

    for i, token in enumerate(tokens):
        valor = 0.0

        if token in palabras_positivas:
            valor = 1.0
            positivas_detectadas.append(token)

        elif token in palabras_negativas:
            valor = -1.0
            negativas_detectadas.append(token)

        ventana_anterior = tokens[max(0, i - 3):i]

        if valor != 0 and any(neg in ventana_anterior for neg in negaciones):
            valor *= -1

        if valor != 0 and any(intens in ventana_anterior for intens in intensificadores):
            valor *= 1.5

        score += valor

    n_pos = len(positivas_detectadas)
    n_neg = len(negativas_detectadas)

    if n_pos > 0 and n_neg > 0:
        if abs(score) <= 1:
            sentimiento = "mixto"
        elif score > 1:
            sentimiento = "positivo"
        else:
            sentimiento = "negativo"
    else:
        if score > 0:
            sentimiento = "positivo"
        elif score < 0:
            sentimiento = "negativo"
        else:
            sentimiento = "neutro"

    return pd.Series({
        "sentimiento_pln": sentimiento,
        "score_sentimiento": float(score),
        "positivas_detectadas": ", ".join(sorted(set(positivas_detectadas))),
        "negativas_detectadas": ", ".join(sorted(set(negativas_detectadas)))
    })


def clasificar_tema(texto):
    texto_limpio = limpiar_texto(texto)
    tokens = texto_limpio.split()

    if len(tokens) == 0:
        return "otros"

    puntajes = {}

    for tema, palabras in temas_keywords.items():
        puntaje = 0

        for palabra in palabras:
            if palabra in tokens:
                puntaje += 1

        puntajes[tema] = puntaje

    tema_max = max(puntajes, key=puntajes.get)

    if puntajes[tema_max] == 0:
        return "otros"

    return tema_max


def calcular_distancia_km(lat1, lon1, lat2, lon2):
    R = 6371

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c


def asignar_color_componentes(indice_positivo):
    if indice_positivo >= 70:
        return 0, 170, 0
    elif indice_positivo >= 40:
        return 255, 180, 0
    else:
        return 220, 0, 0


def calcular_frecuencias(textos, top_n=20):
    stopwords_basicas = {
        "que", "con", "para", "una", "uno", "unos", "unas",
        "los", "las", "del", "por", "pero", "muy", "mas",
        "como", "esta", "este", "son", "fue", "sin", "sus",
        "sea", "hay", "todo", "todos", "toda", "todas",
        "tambien", "restaurant", "restaurante", "lugar",
        "comida", "servicio", "bien", "mal", "solo", "vez",
        "hacer", "hace", "ser", "era", "fui", "ir"
    }

    texto_total = " ".join(textos.dropna().astype(str)).lower()
    palabras = texto_total.split()
    palabras = [p for p in palabras if len(p) > 2 and p not in stopwords_basicas]

    frecuencia = Counter(palabras)

    return pd.DataFrame(
        frecuencia.most_common(top_n),
        columns=["palabra", "frecuencia"]
    )


# ============================================================
# CARGA Y PROCESAMIENTO COMPLETO
# ============================================================

@st.cache_data(show_spinner=True)
def cargar_y_procesar_base():
    tablas = detectar_archivos_csv(RAW_DIR)

    archivo_reviews = max(tablas.keys(), key=lambda a: score_tabla_reviews(tablas[a]))
    archivo_restaurantes = max(tablas.keys(), key=lambda a: score_tabla_restaurantes(tablas[a]))

    reviews = tablas[archivo_reviews].copy()
    restaurants = tablas[archivo_restaurantes].copy()

    # Columnas reales del dataset
    col_id_place_reviews = "id_place"
    col_id_place_rest = "id_place"
    col_comentario = "caption"
    col_restaurante = "title"
    col_lat = "lat"
    col_lon = "long"

    columnas_reviews_necesarias = [col_id_place_reviews, col_comentario]
    columnas_rest_necesarias = [col_id_place_rest, col_restaurante, col_lat, col_lon]

    faltantes_reviews = [c for c in columnas_reviews_necesarias if c not in reviews.columns]
    faltantes_rest = [c for c in columnas_rest_necesarias if c not in restaurants.columns]

    if faltantes_reviews:
        raise ValueError(
            f"Faltan columnas en reviews: {faltantes_reviews}. "
            f"Columnas disponibles: {reviews.columns.tolist()}"
        )

    if faltantes_rest:
        raise ValueError(
            f"Faltan columnas en restaurantes: {faltantes_rest}. "
            f"Columnas disponibles: {restaurants.columns.tolist()}"
        )

    col_id_review = "id_review" if "id_review" in reviews.columns else None

    df = reviews.merge(
        restaurants,
        left_on=col_id_place_reviews,
        right_on=col_id_place_rest,
        how="left",
        suffixes=("_review", "_rest")
    )

    columnas_utiles = [
        col_id_place_reviews,
        col_restaurante,
        col_comentario,
        col_lat,
        col_lon
    ]

    if col_id_review is not None:
        columnas_utiles.insert(1, col_id_review)

    df = df[columnas_utiles].copy()

    renombres = {
        col_id_place_reviews: "id_place",
        col_restaurante: "nombre_restaurante",
        col_comentario: "comentario",
        col_lat: "latitud",
        col_lon: "longitud"
    }

    if col_id_review is not None:
        renombres[col_id_review] = "id_review"

    df = df.rename(columns=renombres)
    df = df.loc[:, ~df.columns.duplicated()].copy()

    columnas_requeridas = [
        "comentario",
        "nombre_restaurante",
        "latitud",
        "longitud"
    ]

    faltantes = [c for c in columnas_requeridas if c not in df.columns]

    if faltantes:
        raise ValueError(
            f"Faltan columnas después del renombrado: {faltantes}. "
            f"Columnas actuales: {df.columns.tolist()}"
        )

    df = df.dropna(
        subset=["comentario", "nombre_restaurante", "latitud", "longitud"]
    ).copy()

    df["comentario"] = df["comentario"].astype(str)
    df["nombre_restaurante"] = df["nombre_restaurante"].astype(str)

    df["latitud"] = pd.to_numeric(df["latitud"], errors="coerce")
    df["longitud"] = pd.to_numeric(df["longitud"], errors="coerce")

    df = df.dropna(subset=["latitud", "longitud"]).copy()

    # Coordenadas razonables para Lima/Perú
    df = df[
        (df["latitud"].between(-13.5, -10.5)) &
        (df["longitud"].between(-78.5, -75.0))
    ].copy()

    # Limitar para que no demore demasiado en Streamlit
    if len(df) > 50000:
        df = df.sample(n=50000, random_state=42).copy()

    df["texto_limpio"] = df["comentario"].apply(limpiar_texto)

    resultados_sentimiento = df["comentario"].apply(analizar_sentimiento_diccionario)
    df = pd.concat([df, resultados_sentimiento], axis=1)

    df["tema"] = df["comentario"].apply(clasificar_tema)

    return df, archivo_reviews, archivo_restaurantes


try:
    df, archivo_reviews_detectado, archivo_restaurantes_detectado = cargar_y_procesar_base()
except Exception as e:
    st.error("No se pudo cargar o procesar la base.")
    st.write(e)
    st.markdown(
        """
        Verifica que tu proyecto tenga esta estructura:

        ```text
        app.py
        requirements.txt
        data/
            raw/
                places.csv
                reviews.csv
        ```

        En este dataset se espera:

        - En restaurantes: `id_place`, `title`, `lat`, `long`
        - En reviews: `id_place`, `caption`
        """
    )
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Configuración")

st.sidebar.success(f"Reseñas procesadas: {len(df):,}")

with st.sidebar.expander("Archivos detectados"):
    st.write("Reviews:", archivo_reviews_detectado)
    st.write("Restaurantes:", archivo_restaurantes_detectado)

st.sidebar.markdown(
    """
    Coordenadas referenciales:

    - Centro de Lima: `-12.0464`, `-77.0428`
    - Miraflores: `-12.1211`, `-77.0297`
    - La Molina: `-12.0817`, `-76.9286`
    - San Isidro: `-12.0970`, `-77.0365`
    """
)


# ============================================================
# RESUMEN POR RESTAURANTE
# ============================================================

df["es_positivo"] = (df["sentimiento_pln"] == "positivo").astype(int)
df["es_negativo"] = (df["sentimiento_pln"] == "negativo").astype(int)
df["es_mixto"] = (df["sentimiento_pln"] == "mixto").astype(int)

resumen = df.groupby(
    ["nombre_restaurante", "latitud", "longitud"],
    as_index=False
).agg(
    total_resenas=("sentimiento_pln", "count"),
    positivas=("es_positivo", "sum"),
    negativas=("es_negativo", "sum"),
    mixtas=("es_mixto", "sum"),
    score_promedio=("score_sentimiento", "mean"),
    tema_mas_frecuente=("tema", lambda x: x.value_counts().index[0])
)

resumen["indice_positivo"] = (
    resumen["positivas"] / resumen["total_resenas"] * 100
).round(2)

resumen["indice_negativo"] = (
    resumen["negativas"] / resumen["total_resenas"] * 100
).round(2)


# ============================================================
# INPUT DE UBICACIÓN
# ============================================================

st.write("## 📍 ¿Dónde estás?")

col1, col2, col3 = st.columns(3)

with col1:
    user_lat = st.number_input("Latitud", value=-12.0817, format="%.6f")

with col2:
    user_lon = st.number_input("Longitud", value=-76.9286, format="%.6f")

with col3:
    radio_km = st.slider("Radio de búsqueda en km", 1, 20, 5, 1)

col4, col5 = st.columns(2)

with col4:
    min_resenas = st.slider("Mínimo de reseñas por restaurante", 1, 100, 5, 1)

with col5:
    top_n = st.slider("Número de recomendaciones", 3, 30, 10, 1)


# ============================================================
# RECOMENDADOR
# ============================================================

resumen["distancia_km"] = calcular_distancia_km(
    user_lat,
    user_lon,
    resumen["latitud"],
    resumen["longitud"]
).round(2)

recomendados = resumen[
    (resumen["distancia_km"] <= radio_km) &
    (resumen["total_resenas"] >= min_resenas)
].copy()

if len(recomendados) == 0:
    st.warning("No se encontraron restaurantes cercanos con los filtros seleccionados.")
    st.stop()

recomendados["score_recomendacion"] = (
    recomendados["indice_positivo"]
    - recomendados["distancia_km"] * 5
    + np.log1p(recomendados["total_resenas"]) * 2
).round(2)

recomendados = recomendados.sort_values("score_recomendacion", ascending=False)
top_recomendados = recomendados.head(top_n).copy()


# ============================================================
# RESULTADO PRINCIPAL
# ============================================================

st.write("## 🍽️ Restaurantes recomendados cerca de ti")

mejor = top_recomendados.iloc[0]

st.success(
    f"Te recomendamos **{mejor['nombre_restaurante']}**, ubicado a "
    f"**{mejor['distancia_km']} km**, con **{mejor['indice_positivo']}%** "
    f"de reseñas positivas inferidas por PLN."
)

st.dataframe(
    top_recomendados[
        [
            "nombre_restaurante",
            "distancia_km",
            "indice_positivo",
            "indice_negativo",
            "total_resenas",
            "score_promedio",
            "tema_mas_frecuente",
            "score_recomendacion"
        ]
    ],
    use_container_width=True
)


# ============================================================
# MAPA
# ============================================================

st.write("## 🗺️ Mapa de recomendaciones")

top_recomendados["color_r"] = top_recomendados["indice_positivo"].apply(
    lambda x: asignar_color_componentes(x)[0]
)
top_recomendados["color_g"] = top_recomendados["indice_positivo"].apply(
    lambda x: asignar_color_componentes(x)[1]
)
top_recomendados["color_b"] = top_recomendados["indice_positivo"].apply(
    lambda x: asignar_color_componentes(x)[2]
)

top_recomendados["radio"] = top_recomendados["total_resenas"].apply(
    lambda x: max(80, min(int(x) * 10, 700))
)

top_recomendados["latitud"] = pd.to_numeric(top_recomendados["latitud"], errors="coerce")
top_recomendados["longitud"] = pd.to_numeric(top_recomendados["longitud"], errors="coerce")
top_recomendados = top_recomendados.dropna(subset=["latitud", "longitud"]).copy()

df_usuario = pd.DataFrame({
    "lat": [float(user_lat)],
    "lon": [float(user_lon)],
    "nombre": ["Tu ubicación"],
    "color_r": [0],
    "color_g": [0],
    "color_b": [255],
    "radio": [150]
})

if len(top_recomendados) > 0:
    layer_restaurantes = pdk.Layer(
        "ScatterplotLayer",
        data=top_recomendados,
        get_position="[longitud, latitud]",
        get_fill_color="[color_r, color_g, color_b]",
        get_radius="radio",
        pickable=True,
    )

    layer_usuario = pdk.Layer(
        "ScatterplotLayer",
        data=df_usuario,
        get_position="[lon, lat]",
        get_fill_color="[color_r, color_g, color_b]",
        get_radius="radio",
        pickable=True,
    )

    view_state = pdk.ViewState(
        latitude=float(user_lat),
        longitude=float(user_lon),
        zoom=13,
        pitch=0,
    )

    tooltip = {
        "html": """
        <b>Restaurante:</b> {nombre_restaurante}<br/>
        <b>Distancia:</b> {distancia_km} km<br/>
        <b>Índice positivo PLN:</b> {indice_positivo}%<br/>
        <b>Índice negativo PLN:</b> {indice_negativo}%<br/>
        <b>Total reseñas:</b> {total_resenas}<br/>
        <b>Tema frecuente:</b> {tema_mas_frecuente}<br/>
        <b>Score recomendación:</b> {score_recomendacion}
        """,
        "style": {
            "backgroundColor": "white",
            "color": "black"
        }
    }

    try:
        st.pydeck_chart(
            pdk.Deck(
                map_style=None,
                layers=[layer_restaurantes, layer_usuario],
                initial_view_state=view_state,
                tooltip=tooltip
            )
        )
    except Exception as e:
        st.warning("No se pudo renderizar el mapa interactivo. Se muestra un mapa simple.")
        st.write(e)

        st.map(
            top_recomendados.rename(
                columns={"latitud": "lat", "longitud": "lon"}
            )[["lat", "lon"]]
        )

st.markdown(
    """
    **Leyenda del mapa:**

    - 🔵 Punto azul: ubicación del usuario.
    - 🟢 Verde: 70% o más de reseñas positivas.
    - 🟡 Amarillo: entre 40% y 69% de reseñas positivas.
    - 🔴 Rojo: menos de 40% de reseñas positivas.
    """
)


# ============================================================
# EXPLORAR RESTAURANTE
# ============================================================

st.write("## 🔎 Explorar restaurante recomendado")

restaurante_elegido = st.selectbox(
    "Selecciona un restaurante",
    top_recomendados["nombre_restaurante"].tolist()
)

detalle = df[df["nombre_restaurante"] == restaurante_elegido].copy()

st.write(f"### {restaurante_elegido}")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Reseñas analizadas", len(detalle))
c2.metric("Positivas", int((detalle["sentimiento_pln"] == "positivo").sum()))
c3.metric("Negativas", int((detalle["sentimiento_pln"] == "negativo").sum()))
c4.metric("Tema frecuente", detalle["tema"].value_counts().index[0])

col_a, col_b = st.columns(2)

with col_a:
    st.write("### Sentimientos")
    conteo_sent = detalle["sentimiento_pln"].value_counts()

    fig, ax = plt.subplots()
    conteo_sent.plot(kind="bar", ax=ax)
    ax.set_xlabel("Sentimiento")
    ax.set_ylabel("Cantidad")
    st.pyplot(fig)

with col_b:
    st.write("### Temas")
    conteo_temas = detalle["tema"].value_counts()

    fig, ax = plt.subplots()
    conteo_temas.plot(kind="bar", ax=ax)
    ax.set_xlabel("Tema")
    ax.set_ylabel("Cantidad")
    st.pyplot(fig)

st.write("### Palabras frecuentes")

freq_detalle = calcular_frecuencias(detalle["texto_limpio"], top_n=15)
st.dataframe(freq_detalle, use_container_width=True)

if WORDCLOUD_AVAILABLE:
    st.write("### Nube de palabras")

    texto_wc = " ".join(detalle["texto_limpio"].dropna().astype(str))

    if texto_wc.strip():
        try:
            wc = WordCloud(
                width=900,
                height=400,
                background_color="white"
            ).generate(texto_wc)

            fig, ax = plt.subplots()
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)
        except Exception as e:
            st.warning("No se pudo generar la nube de palabras.")
            st.write(e)

st.write("### Muestra de reseñas")

st.dataframe(
    detalle[
        [
            "comentario",
            "sentimiento_pln",
            "score_sentimiento",
            "positivas_detectadas",
            "negativas_detectadas",
            "tema"
        ]
    ].head(20),
    use_container_width=True
)


# ============================================================
# ANALIZADOR INDIVIDUAL
# ============================================================

st.write("## ✍️ Analizar una reseña nueva")

comentario_usuario = st.text_area(
    "Escribe una reseña:",
    "La comida estuvo rica, pero la atención fue muy lenta."
)

if st.button("Analizar reseña"):
    resultado = analizar_sentimiento_diccionario(comentario_usuario)
    tema = clasificar_tema(comentario_usuario)
    texto_limpio = limpiar_texto(comentario_usuario)

    a, b, c = st.columns(3)

    a.metric("Sentimiento", resultado["sentimiento_pln"])
    b.metric("Score", resultado["score_sentimiento"])
    c.metric("Tema", tema)

    st.write("**Texto limpio:**", texto_limpio)
    st.write("**Palabras positivas detectadas:**", resultado["positivas_detectadas"])
    st.write("**Palabras negativas detectadas:**", resultado["negativas_detectadas"])


# ============================================================
# METODOLOGÍA
# ============================================================

with st.expander("📌 ¿Cómo funciona WaComer?"):
    st.markdown(
        """
        WaComer realiza el procesamiento completo dentro de la aplicación:

        1. Carga dos archivos CSV ubicados en `data/raw/`.
        2. Detecta cuál corresponde a restaurantes y cuál a reseñas.
        3. Une ambas tablas mediante `id_place`.
        4. Usa `title` como nombre del restaurante.
        5. Usa `caption` como texto de la reseña.
        6. Limpia el texto de las reseñas.
        7. Aplica un enfoque léxico de PLN para inferir sentimiento desde el texto.
        8. Clasifica el tema principal de cada reseña.
        9. Agrupa resultados por restaurante.
        10. Calcula la distancia desde la ubicación del usuario.
        11. Recomienda restaurantes cercanos con mejor percepción textual.

        El score de recomendación usado es:

        `score = índice positivo - 5 × distancia_km + 2 × ln(1 + número de reseñas)`

        Este score prioriza restaurantes cercanos, con mayor porcentaje de reseñas positivas 
        y con suficiente evidencia textual.
        """
    )