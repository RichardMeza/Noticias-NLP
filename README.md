# Analizador de Noticias Peruanas con PLN

Aplicación web desarrollada con Streamlit para analizar noticias peruanas usando Procesamiento de Lenguaje Natural.

## Técnicas utilizadas

- Extracción de texto desde URL con `newspaper3k`
- Extracción de texto desde PDF con `PyMuPDF`
- Preprocesamiento de texto con `NLTK` y `spaCy`
- Reconocimiento de entidades nombradas, NER, con `spaCy`
- Análisis de sentimiento con `pysentimiento`
- Resumen automático extractivo con `Sumy LSA`

## Estructura recomendada del repositorio

```text
nlp-noticias-peruanas/
│── app.py
│── requirements.txt
│── README.md
└── .streamlit/
    └── config.toml
```

## Ejecución local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Despliegue en Streamlit Community Cloud

1. Crear un repositorio en GitHub.
2. Subir `app.py`, `requirements.txt`, `README.md` y la carpeta `.streamlit`.
3. Entrar a Streamlit Community Cloud.
4. Elegir el repositorio.
5. Seleccionar como archivo principal: `app.py`.
6. Presionar Deploy.

## Nota importante

El modelo de spaCy en español se instala desde el archivo `requirements.txt` usando el enlace directo al modelo `es_core_news_sm`.
