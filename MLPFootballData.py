# IMPORTS 
import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# CONFIGURACIÓN DE PÁGINA 
st.set_page_config(page_title="MLPFootballData", layout="wide", page_icon="⚽")

LOGO_URL = "https://i.ibb.co/LdPLCmJG/Logo-mlpfootballdata.png"

# CONTENIDO PRINCIPAL 
st.title("⚽ Scouting Profesional ⚽")
#st.caption("Datos Wyscout")
st.divider()

# CACHE FICHERO CSV
@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

# CACHE FICHERO EXCEL
@st.cache_data
def load_excel(path: str) -> pd.DataFrame:
    return pd.read_excel(path)

# LIMPIA DE DATOS
# Carga dataframe y limpia columnas
# Temp. actual
df = load_csv('Data/s_Scout_TempActual_Jugadores.csv')
df.drop(columns=['Unnamed: 0', 'Equipo'], inplace=True)
df.rename(columns={'Equipo durante el período seleccionado': 'Equipo'}, inplace=True)
df['Pais-Comp'] = df['Pais_Comp'] + " - " + df['Competición']
df = df.fillna(0)
# Historico
df2 = load_excel('Data/s_Scout_Historico_Jugadores.xlsx')
df2.drop(columns=['Unnamed: 0', 'Equipo'], inplace=True)
df2.rename(columns={'Equipo durante el período seleccionado': 'Equipo'}, inplace=True)
df2['Pais-Comp'] = df2['Pais_Comp'] + " - " + df2['Competición']
df2 = df2.fillna(0)

df_total = pd.concat([df, df2], ignore_index=True)

metricas = ['Acciones de ataque exitosas/90','Acciones defensivas realizadas/90','Aceleraciones/90','Asistencias','Asistencias/90',
        'Ataque en profundidad/90','Carreras en progresión/90','Centros al área pequeña/90','Centros desde el último tercio/90',
        'Centros desde la banda derecha/90','Centros desde la banda izquierda/90','Centros/90','Córneres/90','Desmarques/90',
        'Duelos aéreos/90','Duelos aéreos ganados %','Duelos atacantes ganados %','Duelos atacantes/90','Duelos defensivos ganados %','Duelos defensivos/90',
        'Duelos ganados %','Duelos/90','Entradas/90','Faltas recibidas/90','Faltas/90','Goles','Goles (excepto los penaltis)',
        'Goles de cabeza','Goles de cabeza/90','Goles hechos %','Goles, excepto los penaltis/90','Goles/90','Interceptaciones/90','Jugadas claves/90',
        'Longitud media pases largos, m','Longitud media pases, m','Paradas %','Pases al área de penalti/90','Pases cortos / medios/90',
        'Pases en el último tercio/90','Pases en profundidad/90','Pases hacia adelante/90','Pases hacía atrás recibidos del arquero/90',
        'Pases hacia atrás/90','Pases hacía el área pequeña %','Pases largos recibidos/90','Pases largos/90','Pases laterales/90',
        'Pases progresivos/90','Pases recibidos/90','Pases/90','Penaltis a favor','Penaltis realizados %',
        'Posesión conquistada después de una entrada',
        'Posesión conquistada después de una interceptación','Precisión centros %',
        'Precisión centros desde la banda derecha %','Precisión centros desde la banda izquierda %',
        'Precisión desmarques %','Precisión pases %','Precisión pases cortos / medios %',
        'Precisión pases en el último tercio %','Precisión pases en profundidad %',
        'Precisión pases hacia adelante %','Precision pases hacia atrás %','Precisión pases largos %',
        'Precisión pases laterales %','Precisión pases progresivos %','Regates realizados %',
        'Regates/90','Remates','Remates/90',
        'Second assists/90','Tarjetas amarillas','Tarjetas amarillas/90','Tarjetas rojas','Tarjetas rojas/90',
        'Third assists/90','Tiros a la portería %','Tiros interceptados/90','Tiros libres directos %',
        'Tiros libres directos/90','Tiros libres/90','Toques en el área de penalti/90',
        'xA','xA/90','xG','xG/90']

Fase_Defensiva = ['Acciones defensivas realizadas/90','Duelos aéreos/90',
                'Duelos aéreos ganados %','Duelos defensivos ganados %','Duelos defensivos/90','Entradas/90',
                'Faltas/90','Interceptaciones/90','Posesión conquistada después de una entrada',
                'Posesión conquistada después de una interceptación','Tarjetas amarillas','Tarjetas amarillas/90',
                'Tarjetas rojas','Tarjetas rojas/90','Tiros interceptados/90']

ABP = ['Córneres/90','Penaltis a favor','Penaltis realizados %','Tiros libres directos %',
       'Tiros libres directos/90','Tiros libres/90']

Acc_Ofensivas = ['Acciones de ataque exitosas/90','Aceleraciones/90','Asistencias','Asistencias/90',
                'Carreras en progresión/90','Centros al área pequeña/90','Centros desde la banda derecha/90',
                'Centros desde la banda izquierda/90','Centros/90','Duelos atacantes ganados %',
                'Duelos atacantes/90','Faltas recibidas/90','Goles','Goles (excepto los penaltis)',
                'Goles de cabeza','Goles de cabeza/90','Goles hechos %','Goles, excepto los penaltis/90',
                'Goles/90','Pases largos recibidos/90','Pases recibidos/90','Precisión centros %',
                'Precisión centros desde la banda derecha %','Precisión centros desde la banda izquierda %',
                'Regates realizados %','Regates/90','Remates','Remates/90','Tiros a la portería %',
                'Toques en el área de penalti/90','xG','xG/90']

#fisica = ['Total Distance per 90','Running Distance per 90 (15-20 km/h)','HSR Distance per 90 (20-25 km/h)',
#          'Sprinting Distance per 90 (+25 km/h)','HI Distance per 90 (+20 km/h)','Meter/Min','Max Speed (km/h)',
#          'Count Medium Acceleration per 90 (1.5 m/s² to 3 m/s²)','Count High Acceleration per 90 (+3 m/s²)',
#          'Count Medium Deceleration per 90 (-1.5 m/s² to -3 m/s²)','Count High Deceleration per 90 (-3 m/s²)',
#          'Count HSR per 90 (20-25 km/h)','Count Sprint per 90 (+25 km/h)','Count HI per 90 (+20 km/h)']

Jugadas_Clave = ['Asistencias','Asistencias/90',
                'Ataque en profundidad/90','Centros desde el último tercio/90','Desmarques/90',
                'Jugadas claves/90','Pases al área de penalti/90','Pases en el último tercio/90',
                'Pases en profundidad/90','Pases hacía el área pequeña %','Pases progresivos/90',
                'Precisión desmarques %','Precisión pases en el último tercio %',
                'Precisión pases en profundidad %','Precisión pases progresivos %','Second assists/90',
                'Third assists/90','xA','xA/90']

Organizacion = ['Longitud media pases largos, m','Longitud media pases, m','Pases cortos / medios /90',
                'Pases hacia adelante/90','Pases hacia atrás/90','Pases largos/90','Pases laterales/90',
                'Pases/90','Precisión pases %','Precisión pases cortos / medios %',
                'Precisión pases hacia adelante %','Precision pases hacia atrás %','Precisión pases largos %',
                'Precisión pases laterales %']

Portero = ['Acciones defensivas realizadas/90','Duelos aéreos/90',
          'Duelos aéreos ganados %','Duelos defensivos ganados %','Duelos ganados %','Goles evitados',
          'Goles evitados/90','Goles recibidos','Goles recibidos/90','Longitud media pases largos, m',
          'Longitud media pases, m','Paradas %','Pases hacia adelante/90',
          'Pases hacía atrás recibidos del arquero/90','Pases hacia atrás/90','Pases largos/90',
          'Pases laterales/90','Pases/90','Porterías imbatidas en los 90','Precisión pases %',
          'Precisión pases cortos / medios %','Precisión pases hacia adelante %',
          'Precision pases hacia atrás %','Precisión pases largos %','Precisión pases laterales %',
          'Remates en contra','Remates en contra/90','Salidas/90','xG en contra','xG en contra/90']

Finalizacion = ['Goles', 'Goles (excepto los penaltis)', 'Goles de cabeza', 'Goles de cabeza/90',
                'Goles hechos %', 'Goles, excepto los penaltis/90', 'Goles/90', 'Remates', 'Remates/90',
                'Tiros a la portería %', 'xG/90']

#Delantero = ['Goles', 'Goles/90', 'Duelos atacantes ganados %', 'Duelos aéreos ganados %',
#             'Remates/90', 'Regates/90', 'Toques en el área de penalti/90', 'Tiros a la portería %',
#             'xG/90', 'xA/90', 'Regates realizados %']

#Extremo = ['Regates realizados %', 'Centros desde la banda izquierda/90', 'Centros desde la banda derecha/90',
#           'Precisión centros desde la banda derecha %','Precisión centros desde la banda izquierda %',
#           'Duelos atacantes ganados %', 'Goles', 'xG/90', 'Aceleraciones/90']

col_principales = ["Jugador", "Equipo", "Posición específica", "Edad", "Valor de mercado (Transfermarkt)", "Vencimiento contrato",
                   "Partidos jugados", "Minutos jugados"]

# Mapa de metricas
mapa_metricas = {
    "Todas": metricas,
    #"Delantero": Delantero,
    #"Extremo": Extremo,
    "Fase Defensiva": Fase_Defensiva,  
    "Acciones Ofensivas": Acc_Ofensivas,  
    "Organizacion": Organizacion,
    "Jugadas Clave": Jugadas_Clave,
    "Finalización": Finalizacion,
    "ABP": ABP,                        
    "Portero": Portero
    }

# MENÚ LATERAL
with st.sidebar:
    st.image(LOGO_URL, width='stretch')

    st.header("Selector jugador modelo")

    # Temporada
    temporadas = sorted(df['Temporada'].dropna().unique().tolist())
    temp = st.selectbox("Temporada", temporadas)

    # selectbox — JUGADOR MODELO
    jugadores = [""] + sorted(df['Jugador'].dropna().unique().tolist())
    jugador_modelo = st.selectbox("Jugador Modelo", jugadores)

    if jugador_modelo == "":
        st.warning("⚠️ Ningún jugador modelo seleccionado.")
        st.stop()

    df_modelo = df[df['Jugador'] == jugador_modelo]

    #Posición principal, me quedo con la primera
    pos_ppal_modelo = df_modelo['Posición específica'].str.split(',').str[0].item()

    # Perfiles de MÉTRICAS
    perfil_metrica = list(mapa_metricas.keys())

    opcion_elegida = st.selectbox(
                            label = "Selecciona perfil de métricas:",
                            options = perfil_metrica)
    
    metricas_perfil = mapa_metricas[opcion_elegida]
    
    # Filtrar el DataFrame para quedarnos solo con las métricas seleccionadas
    X = df[metricas_perfil]

    # Calcular el promedio de las métricas para el jugador modelo a través de las temporadas
    media_modelo = df_modelo[metricas_perfil].mean().values.reshape(1, -1)

    # Normalizar las métricas del DataFrame Modelo
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)

    # Normalizar el vector de métricas promedio del jugador modelo usando el mismo scaler
    media_modelo_normalizada = scaler.transform(media_modelo)

    # Calcular la similitud de coseno entre el promedio del jugador modelo y todos los jugadores
    similaridades = cosine_similarity(media_modelo_normalizada, X_normalized)[0]

    # Añadir las similitudes al DataFrame y ordenar por similitud
    df['Similitud'] = similaridades * 100
    df_similares = df.sort_values(by='Similitud', ascending=False)
    #quitar duplicados
    df_similares.drop_duplicates(inplace=True)

    # AQUI FALTA UNIR LAS DOS VARIABLES DE LAS COLUMNAS Y MOSTRAR ESAS COLUMNAS
    col_deseadas = col_principales + metricas_perfil + ['Similitud']

    st.divider()
    st.caption("Datos Wyscout")

# APLICAR FILTROS para obtener los datos del jugador modelo
df_modelo= df.copy()

if temp != None:
    df_modelo = df_modelo[df_modelo['Temporada'] == temp]

if jugador_modelo != None:
    df_modelo = df_modelo[df_modelo['Jugador'] == jugador_modelo]

st.markdown("**Datos Jugador Modelo**")
st.dataframe(df_modelo, width='stretch', hide_index=True)

st.divider()
st.markdown(f"**Filtros para buscar los Jugadores similares -** Pos.Específica {pos_ppal_modelo}")

# BÚSQUEDA DE JUGADORES SIMILARES AL JUGADOR MODELO
col1, col2, col3, col4 = st.columns(4)

with col1:
    # selectbox — PAIS
    paises = ["Todas"] + sorted(df['Pais-Comp'].dropna().unique().tolist())
    pais_liga = st.selectbox("País - Liga", paises)

    if pais_liga != "Todas":
        df_similares = df_similares[df_similares['Pais-Comp'] == pais_liga]

with col2:
    # slider — minutos mínimos jugados
    minjug_min = int(df['Minutos jugados'].min())
    minjug_max = int(df['Minutos jugados'].max())
    rango_minutos = st.slider(label="Rango Minutos jugados:",
                                    min_value=minjug_min,
                                    max_value=minjug_max,
                                    value=(minjug_min, minjug_max)  )

    # Rango_seleccionado ahora guarda una tupla: (min_elegido, max_elegido)
    min_minjug_elegido, max_minjug_elegido = rango_minutos

    # Filtrar tu DataFrame existente con los dos extremos seleccionados
    df_similares = df_similares[df_similares['Minutos jugados'].between(min_minjug_elegido, max_minjug_elegido)]

with col3:
    # slider — Edad
    edad_min = int(df['Edad'].min())
    edad_max = int(df['Edad'].max())
    rango_edad = st.slider(label="Rango de Edad:",
                                    min_value=edad_min,
                                    max_value=edad_max,
                                    value=(edad_min, edad_max)  
)
    # Rango_seleccionado ahora guarda una tupla: (min_elegido, max_elegido)
    min_edad_elegido, max_edad_elegido = rango_edad

    # Filtrar tu DataFrame existente con los dos extremos seleccionados
    df_similares = df_similares[df_similares['Edad'].between(min_edad_elegido, max_edad_elegido)]

    
st.divider()

# Añadir filtros fijos TEMPORADA, SIMILITUD y POS. ESPECÍFICA
df_similares1 = df_similares[(df_similares['Temporada'] == temp) &
                            (df_similares['Similitud'] > 0) &
                            (df_similares['Posición específica'].str.contains(pos_ppal_modelo))
                            ]

liga = ""
if pais_liga == "ESP - 1DIVI":
    liga = "1ª División"
elif pais_liga == "ESP - 2DIVI":
    liga = "2ª División"
elif pais_liga == "ESP - 1RFEF":
    liga = "1ª RFEF"
elif pais_liga == "ESP - 2RFEF":
    liga = "2ª RFEF"
elif pais_liga == "POR - LIGA1":
    liga = "LIGA 1"
elif pais_liga == "POR - LIGA2":
    liga = "LIGA 2"
elif pais_liga == "POR - LIGA3":
    liga = "LIGA 3"

if liga != "":
    st.markdown(f"**Nº de jugadores similares: {len(df_similares1)}**")
    st.write(f"{liga}")
else:
    st.markdown(f"**Nº de jugadores similares: {len(df_similares1)}**")
            
df_similares1.reset_index(drop=True, inplace=True)
df_similares1 = df_similares1[df_similares1['Jugador'] != jugador_modelo]

df_similares1[col_deseadas]

with col4:
    st.download_button(
        label = "⬇️ Descargar tabla jugadores similares (CSV)",
        data = df_similares1.to_csv(index = False).encode("utf-8"),
        file_name = "Similares_a_" + jugador_modelo + "_" + liga + ".csv",
        mime = "text/csv",
        width = 'stretch',
    )

st.divider()

# GRAFICO CON EL TOP 10 ----------------------------------------------------------------------

# ESTILO
bg = "#2d3137" #fondo
tit = "#f4cf7f" #amarillo crema resto
key = "#c1ff72" #verde lima num2-3
impact = "#ef233c" #rojo num1
white = "#ffffff"

# Crear un gráfico de barras para los 10 jugadores más similares
top_n = 10

df_top_similares1 = df_similares1[df_similares1['Jugador'] != jugador_modelo].head(top_n)

# Crear un nuevo campo en el DataFrame con el nombre completo para la etiqueta en el gráfico
df_top_similares1['jugador_equipo'] = df_top_similares1['Jugador'] + ' (' + df_top_similares1['Equipo'] + ')'

plt.style.use("default")
plt.figure(figsize=(10, 6), facecolor=bg)

#plt.rcParams['figure.facecolor'] = black   # fondo exterior
plt.rcParams['axes.facecolor'] = bg  # fondo interior de los ejes
plt.rcParams['text.color'] = white
plt.rcParams['axes.labelcolor'] = white
plt.rcParams['xtick.color'] = white
plt.rcParams['ytick.color'] = white

# Reducir el tamaño de los nombres de los jugadores (Eje Y)
plt.tick_params(axis='y', labelsize=8)

# Opcional: Si también quieres hacer más pequeños los números de abajo (Eje X)
plt.tick_params(axis='x', labelsize=8)

plt.xlabel('Similitud', size=9)
plt.xlim(0, 110.50)

#plt.title(f'Top {top_n} perfiles similares a {jugador_modelo}',
#        fontsize=10,
#        fontweight="bold",
#        x = 0.5,
#        y = 0.96,
        #loc="center",
#        color=tit,
#        pad=25)  # Espacio entre título y gráfico

# Subtítulo debajo del título (pero arriba del gráfico)
plt.suptitle(f'Top {top_n} Perfiles similares a {jugador_modelo} - {liga} - Métricas: {opcion_elegida}',
    fontsize=10,
    color=tit,
    x=0.50,
    ha = 'center',
    y=0.96  # controlas la posición vertical del subtítulo
)

# Texto en la esquina superior derecha (datos)
#plt.text(
#    1.0,
#    1.023,
#    "Datos Wyscout",
#    ha="right", va="bottom", transform=plt.gca().transAxes, fontsize=9, color="gray"
#    )

#FIRMA
plt.figtext(0.99, 0.01, "Hecho por @ManuelFLP", ha="right", fontsize=9, color="gray")
#marca de agua
plt.text(0.5, 0.5, "Hecho por @ManuelFLP",
    transform=plt.gca().transAxes, fontsize=30, color="gray",
    alpha=0.2, ha="center", va="center", rotation=30)

bars = plt.barh(df_top_similares1["jugador_equipo"], df_top_similares1["Similitud"], color=tit)
for bar in bars:
    width = bar.get_width()
    plt.text(width + 1.00, bar.get_y() + bar.get_height()/2,
            f'{width:.2f} %', va='center', color = white, fontsize=8)

# Ajustar e invertir el eje y
plt.gca().invert_yaxis()

# Línea vertical con el promedio de los jugadores de su posición
promedio = df_top_similares1["Similitud"].mean()
plt.axvline(promedio, color="grey", linestyle="--", linewidth=1, label="Promedio")

#plt.legend()

#Resaltar los 3 primeros
for i, b in enumerate(bars):
    if i == 0:
        b.set_color(impact)
    elif i < 3:
        b.set_color(key)

plt.tight_layout()
#plt.show()
# Capturas la figura actual y la mandas a Streamlit:
st.pyplot(plt.gcf())

# Limpias el lienzo para que no se mezcle con el siguiente gráfico:
plt.clf()

