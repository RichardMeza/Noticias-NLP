"""
Analizador de Noticias Peruanas con PLN
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
def main():
    st.set_page_config(
        page_title="Analizador de Noticias Peruanas",
        page_icon="📰",
        layout="wide"
    )

    st.title("📰 Analizador de Noticias Peruanas")
    st.write("Aplicación de PLN: extracción, preprocesamiento, NER, sentimiento y resumen automático.")

    tab_url, tab_pdf = st.tabs(["🔗 Desde URL", "📄 Desde PDF"])

    datos = None

    with tab_url:
        url = st.text_input("Ingrese la URL de la noticia")

        if st.button("Analizar noticia desde URL"):
            if url:
                with st.spinner("Cargando modelos..."):
                    nlp, analizador = cargar_modelos()

                with st.spinner("Extrayendo texto..."):
                    datos = extraer_texto(url)

    with tab_pdf:
        archivo = st.file_uploader("Sube un archivo PDF", type=["pdf"])

        if archivo is not None:
            if st.button("Analizar PDF"):
                with st.spinner("Cargando modelos..."):
                    nlp, analizador = cargar_modelos()

                with st.spinner("Extrayendo texto del PDF..."):
                    datos = extraer_texto_pdf(archivo.read(), archivo.name)

    if datos is None:
        st.info("Ingrese una URL o suba un PDF para iniciar el análisis.")
        return

    st.subheader("📌 Artículo analizado")
    st.write(f"**Título:** {datos['titulo']}")
    st.write(f"**Fecha:** {datos['fecha']}")
    st.write(f"**Autor(es):** {datos['autores']}")

    with st.expander("Ver texto completo"):
        st.write(datos["texto"])

    with st.spinner("Preprocesando texto..."):
        prep = preprocesar(datos["texto"], nlp)

    st.subheader("⚙️ Texto preprocesado")
    st.text_area("Texto limpio", prep["texto_limpio"][:1500], height=200)

    st.subheader("🧠 Técnicas de PLN")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🏷️ Entidades")
        entidades = aplicar_ner(datos["texto"], nlp)

        if entidades:
            for tipo, lista in entidades.items():
                st.write(f"**{tipo}:**")
                st.write(", ".join(lista[:10]))
        else:
            st.write("No se encontraron entidades.")

    with col2:
        st.markdown("### 😊 Sentimiento")
        sentimiento = analizar_sentimiento(datos["texto"], analizador)

        st.metric("Sentimiento", sentimiento["sentimiento"])
        st.metric("Confianza", f"{sentimiento['confianza']}%")

        st.write("Detalle:")
        st.write(sentimiento["detalle"])

    with col3:
        st.markdown("### 📝 Resumen")
        resumen = resumir_texto(datos["texto"])
        st.write(resumen)


if __name__ == "__main__":
    main()
