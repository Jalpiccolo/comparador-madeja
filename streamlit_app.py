import streamlit as st
import pandas as pd
import numpy as np
import cv2
from sklearn.cluster import KMeans
from skimage import color
import requests
from io import BytesIO
from PIL import Image

# Configuración de la página
st.set_page_config(
    page_title="Descubre los Hilos de Madeja o Perlé Ideales",
    page_icon="🧵",
    layout="wide"
)

# Estilos CSS personalizados para coincidir con la referencia
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background-color: #ffffff;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #f0f7f9;
        padding-top: 2rem;
    }
    
    .sidebar-subtitle {
        color: #2e7d32;
        font-weight: bold;
        margin-top: -10px;
        margin-bottom: 20px;
    }
    
    .how-it-works {
        background-color: #e3f2fd;
        padding: 20px;
        border-radius: 10px;
        font-size: 0.9em;
    }
    
    .how-it-works h3 {
        color: #0d47a1;
        font-size: 1.1em;
        margin-top: 0;
    }

    /* Main Content Styling */
    .section-header {
        color: #2e7d32;
        font-weight: bold;
        font-size: 1.5em;
        margin-bottom: 20px;
    }
    
    .color-label {
        font-weight: bold;
        color: #333;
        margin-bottom: 15px;
        display: block;
    }

    .detected-color-box {
        width: 100%;
        height: 120px;
        border-radius: 10px;
        border: 1px solid #eee;
        background-color: #fff;
    }
    
    .rgb-text {
        color: #888;
        font-size: 0.8em;
        margin-top: 5px;
    }

    /* Suggestion Cards */
    .suggestion-card {
        background-color: white;
        border: 1px solid #e0e0e0;
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        transition: transform 0.2s;
    }
    
    .suggestion-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }

    .thread-img {
        width: 120px;
        height: 120px;
        border-radius: 50%;
        object-fit: cover;
        margin: 0 auto 15px auto;
        display: block;
    }

    .thread-id {
        font-size: 1.8em;
        font-weight: bold;
        color: #333;
        margin-bottom: 5px;
    }
    
    .thread-name {
        font-weight: bold;
        color: #555;
        margin-bottom: 5px;
    }
    
    .match-pct {
        color: #888;
        font-size: 0.85em;
        margin-bottom: 20px;
    }

    .shop-button {
        display: inline-block;
        width: 100%;
        padding: 10px;
        background-color: white;
        color: #0d47a1;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        text-decoration: none;
        font-weight: bold;
        font-size: 0.9em;
        transition: all 0.3s;
    }
    
    .shop-button:hover {
        background-color: #f8f9fa;
        border-color: #0d47a1;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar con Estilo de la Imagen
with st.sidebar:
    try:
        st.image("logo-piccolo.png", width="stretch")
    except:
        st.title("Piccolo")
    
    st.markdown('<p class="sidebar-subtitle">🧵 Tu asistente de costura</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="how-it-works">
        <h3>¿Cómo funciona?</h3>
        <ol>
            <li>Sube una foto de tu proyecto.</li>
            <li>Detectamos los colores clave.</li>
            <li>Te sugerimos las mejores madejas de nuestro inventario.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

# Función para cargar datos
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Copia de base_datos_madeja.csv")
    except:
        try:
            df = pd.read_csv("base_datos_madeja.csv")
        except:
            return None
    
    required_columns = ["ID", "Nombre_Lana", "Ruta_Foto", "Color_Hex", "Color_RGB"]
    for col in required_columns:
        if col not in df.columns:
            st.error(f"Falta la columna necesaria: {col}")
            return None
            
    def parse_rgb(rgb_str):
        try:
            return [int(x.strip()) for x in rgb_str.replace("(", "").replace(")", "").split(",")]
        except:
            return [0, 0, 0]
            
    df['RGB_tuple'] = df['Color_RGB'].apply(parse_rgb)
    
    rgb_array = np.array(df['RGB_tuple'].tolist()).reshape(-1, 1, 3) / 255.0
    lab_array = color.rgb2lab(rgb_array).reshape(-1, 3)
    df['LAB'] = list(lab_array)
    
    return df

# Función para extraer colores dominantes usando K-Means
def extract_colors(image, n_clusters=10):
    img = np.array(image)
    img = cv2.resize(img, (200, 200), interpolation=cv2.INTER_AREA)
    img_flatten = img.reshape((-1, 3))
    
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    kmeans.fit(img_flatten)
    
    colors = kmeans.cluster_centers_.astype(int)
    return colors

# Función para encontrar las mejores coincidencias
def find_matches(target_rgb, df_madejas, top_n=2):
    target_lab = color.rgb2lab(np.array(target_rgb).reshape(1, 1, 3) / 255.0).reshape(3)
    
    distances = []
    for idx, row in df_madejas.iterrows():
        dist = np.linalg.norm(target_lab - np.array(row['LAB']))
        distances.append(dist)
    
    df_temp = df_madejas.copy()
    df_temp['dist'] = distances
    
    matches = df_temp.sort_values('dist').head(top_n)
    return matches

# App UI
df_madejas = load_data()

st.title("Descubre los Hilos de Madeja o Perlé Ideales")
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file is not None and df_madejas is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    col_img, col_info = st.columns([1, 2])
    with col_img:
        st.image(image, caption="Tu proyecto", width="stretch")
    with col_info:
        st.markdown('<p class="section-header">Colores que armonizan perfectamente con tu foto</p>', unsafe_allow_html=True)
        
        progress_bar = st.progress(0)
        
        progress_bar.progress(30, text="Analizando imagen...")
        extracted_colors = extract_colors(image, n_clusters=10)
        
        progress_bar.progress(65, text="Extrayendo paleta de colores...")
        
        progress_bar.progress(100, text="Buscando coincidencias Piccolo...")
        st.success("¡Análisis completado!")

    st.markdown("---")

    # Resultados
    for i, target_rgb in enumerate(extracted_colors):
        hex_color = '#{:02x}{:02x}{:02x}'.format(*target_rgb)
        
        # Grid para Color Detectado + Sugerencia 1 + Sugerencia 2
        cols = st.columns([1, 1.5, 1.5])
        
        with cols[0]:
            st.markdown(f'<span class="color-label">Color Detectado #{i+1}</span>', unsafe_allow_html=True)
            st.markdown(f'<div class="detected-color-box" style="background-color: {hex_color};"></div>', unsafe_allow_html=True)
            st.markdown(f'<p class="rgb-text">RGB: ({target_rgb[0]}, {target_rgb[1]}, {target_rgb[2]})</p>', unsafe_allow_html=True)
            
        matches = find_matches(target_rgb, df_madejas)
        
        for m_idx, (mid, match) in enumerate(matches.iterrows()):
            with cols[m_idx + 1]:
                st.markdown(f'<span class="color-label">Sugerencia {m_idx+1}</span>', unsafe_allow_html=True)
                
                # Intentamos extraer un ID corto o código del nombre
                thread_id = match['Nombre_Lana'].split()[-1] if len(match['Nombre_Lana'].split()) > 1 else match['ID']
                
                # Calculamos un Match % visual (Delta-E de 0 a 50 aprox para buen match)
                distance = match['dist']
                match_pct = max(0, 100 - (distance * 1.5))
                
                st.markdown(f"""
                <div class="suggestion-card">
                    <img src="{match['Ruta_Foto']}" class="thread-img">
                    <div class="thread-id">{thread_id}</div>
                    <div class="thread-name">{match['Nombre_Lana']}</div>
                    <div class="match-pct">Match: {match_pct:.1f}%</div>
                    <a href="https://piccolo.com.co/categoria-producto/bordado/hilos/" target="_blank" class="shop-button">🛍️ Ver Producto</a>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)

else:
    if df_madejas is None:
        st.warning("No se pudo cargar la base de datos.")
    else:
        st.markdown('<p class="section-header">Sube una imagen para encontrar tus hilos</p>', unsafe_allow_html=True)
