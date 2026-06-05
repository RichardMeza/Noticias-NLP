"""
Analizador de Noticias Peruanas con PLN
PC3 - Procesamiento de Lenguaje Natural
========================================
Técnicas PLN aplicadas:
  1. NER (Reconocimiento de Entidades)
  2. Análisis de Sentimiento
  3. Resumen Automático (Sumy LSA)

Fuentes de entrada:
  - URL de artículo web
  - Archivo PDF (subido directamente)
"""

# ─── INSTALACIÓN (correr una sola vez) ────────────────────────────────────────
# pip install newspaper3k spacy pysentimiento sumy nltk lxml[html_clean] pymupdf
# python -m spacy download es_core_news_sm

import re
import string
import html
import streamlit as st
import nltk
import spacy
import fitz  # PyMuPDF
from newspaper import Article
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from pysentimiento import create_analyzer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from collections import Counter

# ─── DESCARGA DE RECURSOS NLTK ────────────────────────────────────────────────
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# ─── CARGA DE MODELOS (se cachean para no recargar) ───────────────────────────
@st.cache_resource
def cargar_modelos():
    nlp        = spacy.load("es_core_news_sm")
    analizador = create_analyzer(task="sentiment", lang="es")
    return nlp, analizador

# ══════════════════════════════════════════════════════════════════════════════
# FASE 1 — EXTRACCIÓN DE TEXTO
# ══════════════════════════════════════════════════════════════════════════════
def extraer_texto(url: str) -> dict:
    """Descarga y parsea el artículo desde la URL."""
    articulo = Article(url, language="es")
    articulo.download()
    articulo.parse()
    return {
        "titulo": articulo.title,
        "texto":  articulo.text,
        "fecha":  articulo.publish_date.strftime("%d/%m/%Y") if articulo.publish_date else "No disponible",
        "autores": ", ".join(articulo.authors) if articulo.authors else "No disponible",
    }

# ══════════════════════════════════════════════════════════════════════════════
# FASE 1B — EXTRACCIÓN DESDE PDF
# ══════════════════════════════════════════════════════════════════════════════
def extraer_texto_pdf(archivo_bytes: bytes, nombre_archivo: str) -> dict:
    """Extrae texto de un PDF usando PyMuPDF (fitz)."""
    doc = fitz.open(stream=archivo_bytes, filetype="pdf")
    paginas_texto = []
    for num_pagina in range(len(doc)):
        pagina = doc[num_pagina]
        paginas_texto.append(pagina.get_text())
    texto_completo = "\n".join(paginas_texto).strip()
    doc.close()
    return {
        "titulo":  nombre_archivo.replace(".pdf", "").replace("_", " ").title(),
        "texto":   texto_completo,
        "fecha":   "No disponible",
        "autores": "No disponible",
    }

# ══════════════════════════════════════════════════════════════════════════════
# FASE 2 — PREPROCESAMIENTO
# ══════════════════════════════════════════════════════════════════════════════
def limpiar_texto(texto: str) -> str:
    """Limpieza básica: URLs, caracteres especiales, espacios extra."""
    texto = re.sub(r"http\S+|www\S+", "", texto)           # quitar URLs
    texto = re.sub(r"\S+@\S+", "", texto)                  # quitar emails
    texto = re.sub(r"\d+", "", texto)                      # quitar números
    texto = re.sub(r"[^\w\sáéíóúñüÁÉÍÓÚÑÜ]", " ", texto)  # quitar puntuación
    texto = re.sub(r"\s+", " ", texto).strip()             # espacios extra
    return texto

def tokenizar(texto: str) -> list:
    """Divide el texto en tokens (palabras)."""
    return word_tokenize(texto.lower(), language="spanish")

def eliminar_stopwords(tokens: list) -> list:
    """Elimina palabras vacías en español."""
    stops = set(stopwords.words("spanish"))
    return [t for t in tokens if t not in stops and t not in string.punctuation]

def lematizar(texto: str, nlp) -> list:
    """Reduce palabras a su forma base usando spaCy."""
    doc = nlp(texto)
    return [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]

def stemming(tokens: list) -> list:
    """Reduce palabras a su raíz con Snowball Stemmer."""
    stemmer = SnowballStemmer("spanish")
    return [stemmer.stem(t) for t in tokens]

def normalizar(texto: str) -> str:
    """Normaliza tildes y mayúsculas."""
    reemplazos = {"á":"a","é":"e","í":"i","ó":"o","ú":"u","ü":"u","ñ":"n"}
    for k, v in reemplazos.items():
        texto = texto.replace(k, v)
    return texto.lower()

def preprocesar(texto: str, nlp) -> dict:
    """Pipeline completo de preprocesamiento."""
    limpio    = limpiar_texto(texto)
    tokens    = tokenizar(limpio)
    sin_stops = eliminar_stopwords(tokens)
    lemas     = lematizar(limpio, nlp)
    stems     = stemming(sin_stops)
    normal    = normalizar(limpio)
    return {
        "texto_limpio":        limpio,
        "tokens":              tokens,
        "tokens_sin_stopwords": sin_stops,
        "lemas":               lemas,
        "stems":               stems,
        "texto_normalizado":   normal,
        "frecuencia":          Counter(sin_stops).most_common(10),
    }

# ══════════════════════════════════════════════════════════════════════════════
# FASE 3 — TÉCNICAS PLN
# ══════════════════════════════════════════════════════════════════════════════

# ── Técnica 1: NER ────────────────────────────────────────────────────────────
ETIQUETAS_ES = {
    "PER": "Persona",
    "ORG": "Organización",
    "LOC": "Lugar",
    "MISC": "Miscelánea",
}

def aplicar_ner(texto: str, nlp) -> dict:
    """Extrae entidades nombradas del texto."""
    doc       = nlp(texto)
    entidades = {}
    for ent in doc.ents:
        label = ETIQUETAS_ES.get(ent.label_, ent.label_)
        entidades.setdefault(label, [])
        if ent.text not in entidades[label]:
            entidades[label].append(ent.text)
    return entidades

# ── Técnica 2: Análisis de sentimiento ────────────────────────────────────────
ETIQUETAS_SENTIMIENTO = {
    "POS": "😊 Positivo",
    "NEG": "😠 Negativo",
    "NEU": "😐 Neutro",
}

def analizar_sentimiento(texto: str, analizador) -> dict:
    """Clasifica el sentimiento del texto."""
    # pysentimiento tiene límite de tokens, usamos los primeros 512 chars
    fragmento  = texto[:512]
    resultado  = analizador.predict(fragmento)
    etiqueta   = ETIQUETAS_SENTIMIENTO.get(resultado.output, resultado.output)
    confianza  = round(max(resultado.probas.values()) * 100, 1)
    return {
        "sentimiento": etiqueta,
        "confianza":   confianza,
        "detalle":     {ETIQUETAS_SENTIMIENTO.get(k, k): round(v * 100, 1)
                        for k, v in resultado.probas.items()},
    }

# ── Técnica 3: Resumen automático (LSA) ───────────────────────────────────────
def resumir_texto(texto: str, num_oraciones: int = 4) -> str:
    """Genera un resumen extractivo con el algoritmo LSA."""
    parser    = PlaintextParser.from_string(texto, Tokenizer("spanish"))
    resumidor = LsaSummarizer()
    resumen   = resumidor(parser.document, num_oraciones)
    return " ".join(str(oracion) for oracion in resumen)

# ══════════════════════════════════════════════════════════════════════════════
# INTERFAZ STREAMLIT
# ══════════════════════════════════════════════════════════════════════════════
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Paleta base ─────────────────────────────────────────── */
:root {
    --ink:      #0f0f0f;
    --paper:    #f5f0e8;
    --accent:   #c0392b;
    --muted:    #7a7060;
    --border:   #d4c9b0;
    --card-bg:  #fffdf8;
}

/* ── Reset general ───────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--paper) !important;
    font-family: 'DM Sans', sans-serif;
    color: var(--ink);
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] { display: none; }
section.main > div { padding-top: 0 !important; }

/* ── Inputs con contraste correcto ───────────────────────── */
[data-testid="stTextInput"] input {
    background-color: #ffffff !important;
    color: var(--ink) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 2px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
}
[data-testid="stTextInput"] > div > div {
    background-color: #ffffff !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 2px !important;
}
[data-testid="stTextInput"] input::placeholder { color: var(--muted) !important; }
[data-testid="stTextInput"] input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(192,57,43,.10) !important;
}

/* ── Tabs ────────────────────────────────────────────────── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background-color: transparent !important;
    border-bottom: 2px solid var(--border) !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background-color: transparent !important;
    color: var(--muted) !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: var(--ink) !important;
    border-bottom: 2px solid var(--accent) !important;
}

/* ── Alert / info box ────────────────────────────────────── */
[data-testid="stAlert"] {
    background-color: #fffdf8 !important;
    border: 1px solid var(--border) !important;
    color: var(--ink) !important;
    border-radius: 2px !important;
}

/* ── File uploader ───────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background-color: #ffffff !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: 2px !important;
}

/* ── Caption ─────────────────────────────────────────────── */
[data-testid="stCaptionContainer"], small { color: var(--muted) !important; }

/* ── Header del periódico ────────────────────────────────── */
.newspaper-header {
    border-top: 5px solid var(--ink);
    border-bottom: 3px double var(--ink);
    text-align: center;
    padding: 18px 0 12px;
    margin-bottom: 28px;
    background: var(--paper);
}
.newspaper-header .kicker {
    font-family: 'DM Sans', sans-serif;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 6px;
}
.newspaper-header h1 {
    font-family: 'Playfair Display', serif;
    font-size: clamp(28px, 5vw, 52px);
    font-weight: 900;
    line-height: 1;
    color: var(--ink);
    margin: 0;
    letter-spacing: -1px;
}
.newspaper-header .sub {
    font-size: 12px;
    color: var(--muted);
    margin-top: 8px;
    letter-spacing: 1px;
}
.rule-thin  { height: 1px; background: var(--border); margin: 10px 0; }

/* ── Tarjetas de contenido ───────────────────────────────── */
.card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 2px;
    padding: 22px 24px;
    margin-bottom: 18px;
    box-shadow: 2px 2px 0 rgba(0,0,0,.06);
}
.card-title {
    font-family: 'Playfair Display', serif;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
}

/* ── Metadatos del artículo ──────────────────────────────── */
.article-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(18px, 3vw, 26px);
    font-weight: 700;
    line-height: 1.3;
    color: var(--ink);
    margin: 0 0 10px;
}
.meta-row {
    display: flex;
    gap: 24px;
    flex-wrap: wrap;
    margin-top: 10px;
}
.meta-item {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    color: var(--muted);
    font-weight: 500;
    letter-spacing: .5px;
}
.meta-label { text-transform: uppercase; font-size: 10px; letter-spacing: 1.5px; }

/* ── Texto limpio ────────────────────────────────────────── */
.clean-text-box {
    background: #fff;
    border-left: 3px solid var(--accent);
    padding: 16px 18px;
    font-size: 14px;
    line-height: 1.75;
    color: #333;
    max-height: 200px;
    overflow-y: auto;
    font-family: 'DM Sans', sans-serif;
    border-radius: 0 2px 2px 0;
}

/* ── Sección NER ─────────────────────────────────────────── */
.ner-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
    gap: 14px;
}
.ner-group { }
.ner-type {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 6px;
    padding-bottom: 4px;
    border-bottom: 1px solid var(--border);
}
.ner-tag {
    display: inline-block;
    background: var(--paper);
    border: 1px solid var(--border);
    border-radius: 2px;
    padding: 3px 8px;
    font-size: 12px;
    margin: 3px 2px;
    color: var(--ink);
}

/* ── Sentimiento ─────────────────────────────────────────── */
.sentiment-big {
    font-family: 'Playfair Display', serif;
    font-size: 32px;
    font-weight: 700;
    text-align: center;
    margin: 8px 0 4px;
}
.sentiment-conf {
    text-align: center;
    font-size: 12px;
    color: var(--muted);
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 16px;
}
.bar-row { margin-bottom: 10px; }
.bar-label {
    display: flex;
    justify-content: space-between;
    font-size: 11px;
    color: var(--muted);
    margin-bottom: 3px;
    text-transform: uppercase;
    letter-spacing: .5px;
}
.bar-track {
    height: 6px;
    background: var(--border);
    border-radius: 3px;
    overflow: hidden;
}
.bar-fill {
    height: 100%;
    background: var(--accent);
    border-radius: 3px;
    transition: width .6s ease;
}

/* ── Resumen ─────────────────────────────────────────────── */
.resumen-text {
    font-family: 'Playfair Display', serif;
    font-size: 16px;
    line-height: 1.8;
    color: var(--ink);
    font-style: italic;
}

/* ── Tabs de streamlit ───────────────────────────────────── */
[data-testid="stTabs"] button {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 12px !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    font-weight: 500 !important;
}

/* ── Footer ──────────────────────────────────────────────── */
.footer {
    border-top: 2px solid var(--ink);
    margin-top: 40px;
    padding-top: 12px;
    text-align: center;
    font-size: 11px;
    color: var(--muted);
    letter-spacing: 1px;
    text-transform: uppercase;
}
</style>
"""

def main():
    st.set_page_config(
        page_title="Analizador de Noticias Peruanas · PLN",
        page_icon="📰",
        layout="wide",
    )

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # ── Cabecera estilo periódico ──────────────────────────────────────────────
    st.markdown("""
    <div class="newspaper-header">
        <div class="kicker">Procesamiento de Lenguaje Natural</div>
        <h1>Analizador de Noticias Peruanas</h1>
        <div class="sub">Herramienta de análisis · NER · Sentimiento · Resumen</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Selección de fuente ────────────────────────────────────────────────────
    tab_url, tab_pdf = st.tabs(["🔗 Desde URL", "📄 Desde PDF"])

    datos = None

    with tab_url:
        url = st.text_input("URL de la noticia", placeholder="https://rpp.pe/...", label_visibility="collapsed")
        st.caption("Pega la URL de cualquier noticia (RPP, El Comercio, La República, Gestión, etc.)")
        if url:
            with st.spinner("Cargando modelos PLN..."):
                nlp, analizador = cargar_modelos()
            with st.spinner("Extrayendo texto de la noticia..."):
                try:
                    datos = extraer_texto(url)
                except Exception as e:
                    st.error(f"No se pudo extraer el texto: {e}")
            if datos and not datos["texto"]:
                st.error("No se encontró texto en esa URL. Prueba con otra noticia.")
                datos = None

    with tab_pdf:
        archivo = st.file_uploader("Sube un archivo PDF", type=["pdf"])
        if archivo is not None:
            with st.spinner("Cargando modelos PLN..."):
                nlp, analizador = cargar_modelos()
            with st.spinner("Extrayendo texto del PDF..."):
                try:
                    datos = extraer_texto_pdf(archivo.read(), archivo.name)
                except Exception as e:
                    st.error(f"No se pudo leer el PDF: {e}")
            if datos and not datos["texto"]:
                st.error("El PDF no contiene texto extraíble (puede ser un PDF escaneado).")
                datos = None

    if datos is None:
        st.info("Usa una de las pestañas de arriba para comenzar el análisis.")
        return

    # ── Metadatos del artículo ─────────────────────────────────────────────────
    titulo_html  = html.escape(datos["titulo"] or "Sin título")
    fecha_html   = html.escape(datos["fecha"])
    autores_html = html.escape(datos["autores"])

    st.markdown(f"""
    <div class="card">
        <div class="card-title">Artículo analizado</div>
        <div class="article-title">{titulo_html}</div>
        <div class="meta-row">
            <div class="meta-item">
                <span>📅</span>
                <span><span class="meta-label">Fecha&nbsp;</span>{fecha_html}</span>
            </div>
            <div class="meta-item">
                <span>✍️</span>
                <span><span class="meta-label">Autor&nbsp;</span>{autores_html}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Ver texto completo extraído"):
        st.write(datos["texto"])

    # ── Preprocesamiento — solo texto limpio ───────────────────────────────────
    with st.spinner("Preprocesando..."):
        prep = preprocesar(datos["texto"], nlp)

    texto_limpio_truncado = prep["texto_limpio"][:1200].replace("<", "&lt;").replace(">", "&gt;")
    st.markdown(f"""
    <div class="card">
        <div class="card-title">⚙️ Texto preprocesado</div>
        <div class="clean-text-box">{texto_limpio_truncado}…</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Técnicas PLN ───────────────────────────────────────────────────────────
    st.markdown("<div class='card-title' style='font-size:14px;letter-spacing:2px;color:#c0392b;margin-top:8px;'>🧠 Técnicas de PLN</div>", unsafe_allow_html=True)

    col_ner, col_sent, col_res = st.columns([1.1, 0.9, 1], gap="medium")

    # ── NER ────────────────────────────────────────────────────────────────────
    with col_ner:
        with st.spinner("Reconociendo entidades..."):
            entidades = aplicar_ner(datos["texto"], nlp)

        grupos_html = ""
        if entidades:
            for tipo, lista in entidades.items():
                tags = "".join(f'<span class="ner-tag">{html.escape(e)}</span>' for e in lista[:6])
                grupos_html += f'<div class="ner-group"><div class="ner-type">{tipo}</div>{tags}</div>'
        else:
            grupos_html = "<p style='color:var(--muted);font-size:13px;'>No se encontraron entidades.</p>"

        st.markdown(f"""
        <div class="card" style="height:100%">
            <div class="card-title">Reconocimiento de Entidades</div>
            <div class="ner-grid">{grupos_html}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Sentimiento ────────────────────────────────────────────────────────────
    with col_sent:
        with st.spinner("Analizando sentimiento..."):
            sentimiento = analizar_sentimiento(datos["texto"], analizador)

        barras_html = ""
        for etiq, prob in sentimiento["detalle"].items():
            barras_html += (
                '<div class="bar-row">'
                '<div class="bar-label">'
                f'<span>{etiq}</span><span>{prob}%</span>'
                '</div>'
                '<div class="bar-track">'
                f'<div class="bar-fill" style="width:{prob}%"></div>'
                '</div>'
                '</div>'
            )

        sent_html = (
            '<div class="card" style="height:100%">'
            '<div class="card-title">Análisis de Sentimiento</div>'
            f'<div class="sentiment-big">{sentimiento["sentimiento"]}</div>'
            f'<div class="sentiment-conf">Confianza: {sentimiento["confianza"]}%</div>'
            + barras_html +
            '</div>'
        )
        st.markdown(sent_html, unsafe_allow_html=True)

    # ── Resumen ────────────────────────────────────────────────────────────────
    with col_res:
        with st.spinner("Generando resumen..."):
            resumen = resumir_texto(datos["texto"])

        resumen_texto = html.escape(resumen if resumen else "No se pudo generar el resumen.")
        st.markdown(f"""
        <div class="card" style="height:100%">
            <div class="card-title">Resumen Automático</div>
            <div class="resumen-text">"{resumen_texto}"</div>
        </div>
        """, unsafe_allow_html=True)



if __name__ == "__main__":
    main()
