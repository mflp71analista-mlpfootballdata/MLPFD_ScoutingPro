# IMPORTS 
import streamlit as st
import pandas as pd
import numpy as np
import re
import time
import matplotlib.pyplot as plt
from azure.storage.blob import BlobServiceClient
from io import BytesIO, StringIO

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# desactiva la descarga de datos directamente del dataframe
st.markdown(
    """
    <style>
    /* Oculta la barra de herramientas flotante superior en todos los dataframes */
    [data-testid="stDataFrameToolbar"] {
        display: none !important;
    }
    
    /* Oculta cualquier botón flotante de descarga residual */
    button[title*="Download"], [data-testid="stElementToolbar"] {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# CACHE FICHERO EXCEL
@st.cache_data
def load_excel(path: str) -> pd.DataFrame:
    return pd.read_excel(path)

# CACHE FICHERO CSV
@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

# CONFIGURACIÓN DE PÁGINA 
st.set_page_config(page_title="MLPFootballData", layout="wide", page_icon="⚽")

# CONTENIDO PRINCIPAL 
st.title("⚽ Scouting Profesional⚽")
st.divider()

# CONEXION CON LA NUBE MICROSOFT - CREDENCIALES
correct_key= st.secrets["nube microsoft"]["key_contendor"]

ACCOUNT_NAME = 'mlpfd' # cuenta azure mlpfd
ACCOUNT_KEY = correct_key
CONTAINER_NAME = 'wyscout' # nombre DEL CONTENEDPROR

# SET UP
blob_service_client = BlobServiceClient(account_url=f"https://{ACCOUNT_NAME}.blob.core.windows.net",credential=ACCOUNT_KEY)
container_client = blob_service_client.get_container_client(CONTAINER_NAME)

# leer todos los ficheros
#blobs = []
#for blob in container_client.list_blobs():
#    blobs.append(blob.name)

# LEER METRICAS
BLOB_NAME_METRICAS = 'metricas_grupos.xlsx'
blob_client = container_client.get_blob_client(BLOB_NAME_METRICAS)
data = blob_client.download_blob().readall()
df_metricas = load_excel(BytesIO(data))

# obtener los valores unicos de GRUPO_ROL, POS_ESPECÍFICA y METRIC
# quitamos las físicas
df_metricas = df_metricas[df_metricas["GRUPO_ROL"] != "Físico"]

all_grupos = ["TODAS"] + sorted(df_metricas[df_metricas["POS_ESPECÍFICA"] != "GK"]["GRUPO_ROL"].unique())
all_grupos_GK = ["TODAS"] + sorted(df_metricas[df_metricas["POS_ESPECÍFICA"] == "GK"]["GRUPO_ROL"].unique())

all_pos_espec = sorted(df_metricas['POS_ESPECÍFICA'].unique())

all_metricas = sorted(df_metricas[df_metricas["POS_ESPECÍFICA"] != "GK"]["METRICA"].unique())
all_metricas_GK = sorted(df_metricas[df_metricas["POS_ESPECÍFICA"] == "GK"]["METRICA"].unique())

# LEER JUGADORES
BLOB_NAME_JUGADORES = 's_Scout_TempActual_Jugadores.csv'
blob_client = container_client.get_blob_client(BLOB_NAME_JUGADORES)
data = blob_client.download_blob().readall()
df = load_csv(BytesIO(data))
df.drop(columns=['Unnamed: 0', 'Equipo'], inplace=True)
df.rename(columns={'Equipo durante el período seleccionado': 'Equipo'}, inplace=True)
df['Pais-Comp'] = df['Pais_Comp'] + " - " + df['Competición']
df['Jugador-Posiciones'] = df['Jugador'] + " (" + df['Posición específica'] + ")"
df = df.fillna(0)
col_principales = ["Jugador", "Equipo", "Posición específica", "Edad", "Valor de mercado (Transfermarkt)", "Vencimiento contrato",
                   "Partidos jugados", "Minutos jugados"]

# ══════════════════════════════════════════════════════
# MENÚ LATERAL
# ══════════════════════════════════════════════════════
with st.sidebar:
    LOGO_URL = "https://i.ibb.co/LdPLCmJG/Logo-mlpfootballdata.png"
    st.image(LOGO_URL, width='stretch')
    st.caption("Datos Wyscout")

    st.header("Selector jugador modelo")

    # Temporada
    temporadas = sorted(df['Temporada'].dropna().unique().tolist())
    temp = st.selectbox("Temporada", temporadas)

    # Elgir entre portero o el resto
    portero = st.radio("Portero:", ["Sí", "No"])

    # defino posicion, roles y metricas dependiendo si es portero o no
    if portero == "Sí":
        df = df[df['Posición específica'] == "GK"]
        perfil_grupo_rol = all_grupos_GK
        perfil_metrica = all_metricas_GK
    else:
        df = df[df['Posición específica'] != "GK"]
        perfil_grupo_rol = all_grupos
        perfil_metrica = all_metricas

    # selectbox — JUGADOR MODELO
    jugadores = [""] + sorted(df['Jugador-Posiciones'].dropna().unique().tolist())
    jugador_modelo = st.selectbox("Jugador Modelo", jugadores)

    if jugador_modelo == "":
        st.warning("⚠️ Ningún jugador modelo seleccionado.")
        st.stop()

    df_modelo = df[df['Jugador-Posiciones'] == jugador_modelo]

    # Extraer todas las posiciones del jugador modelo
    pos_jugador = (
        df_modelo["Posición específica"]
        .dropna()
        .astype(str)
        .str.split(",")
        .explode()
        .str.strip()
    )

    # Obtener la lista única de valores específicos de este jugador
    lista_pos_jugador = sorted(pos_jugador.unique())

    # Mostrar el combo con los valores filtrados
    pos_ppal_modelo = st.selectbox("Selecciona posición específica:", options = lista_pos_jugador)

    nombre_jugador = df_modelo.iloc[0]["Jugador"] 

    # Mostrar el combo con los GRUPOS
    
    grupo_rol_sel = st.selectbox("Selecciona grupo métricas:", options = perfil_grupo_rol)

    # Obtengo las metricas a analizar según lo seleccionado
    # DataFrame ya filtrado por posición y grupo
    df_filtrado = df_metricas[df_metricas["POS_ESPECÍFICA"] == pos_ppal_modelo].copy()

    if grupo_rol_sel != "TODAS":
        df_filtrado = df_filtrado[df_filtrado["GRUPO_ROL"] == grupo_rol_sel]

    # Guardar el listado de métricas en una variable tipo lista
    metricas_perfil = sorted(df_filtrado["METRICA"].unique())

    st.session_state["metricas_perfil"] = metricas_perfil

    # *********************** SIMILITUD ****************************************************************
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

    #st.divider()

    #st.caption("Datos Wyscout")

# FIN BARRA LATERAL **************************************************************

# INICIO TAB
tab1, tab2 = st.tabs(["🔍 Búsqueda por Similitud", "👕 Detalle Jugador"])

# ══════════════════════════════════════════════════════
# TAB 1 — BÚSQUEDA POR SIMILITUD
# ══════════════════════════════════════════════════════

with tab1:

    # APLICAR FILTROS para obtener los datos del jugador modelo
    df_modelo= df.copy()

    if temp != None:
        df_modelo = df_modelo[df_modelo['Temporada'] == temp]

    if jugador_modelo != None:
        df_modelo = df_modelo[df_modelo['Jugador-Posiciones'] == jugador_modelo]

    df_modelo.drop(columns=['Jugador-Posiciones'], inplace=True)
    st.markdown("**Datos Jugador Modelo**")
    # Ocultar el botón de descarga dentro de la barra de herramientas de st.dataframe

    st.dataframe(df_modelo, width='stretch', hide_index=True)

    st.divider()
    st.markdown("**Filtros para buscar los Jugadores similares**")

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
                                        value=(minjug_min, minjug_max,),
                                        step=100  )

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

    # obtener cada posición especifica individual, para poder hacer busqueda exacta
    patron = r"(?<!\w)" + re.escape(pos_ppal_modelo) + r"(?!\w)"

    # Añadir filtros fijos TEMPORADA, SIMILITUD y POS. ESPECÍFICA
    df_similares1 = df_similares[(df_similares['Temporada'] == temp) &
                                (df_similares['Similitud'] > 0) &
                                (df_similares['Posición específica'].str.contains(patron, regex=True, na=False))
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
        st.markdown(f"**Pos. específica  {pos_ppal_modelo} - Nº de jugadores similares: {len(df_similares1)}**")
        st.write(f"{liga}")
    else:
        st.markdown(f"**Pos. específica  {pos_ppal_modelo} - Nº de jugadores similares: {len(df_similares1)}**")
                
    df_similares1.reset_index(drop=True, inplace=True)
    df_similares1 = df_similares1[df_similares1['Jugador-Posiciones'] != jugador_modelo]

    # MUESTRO LISTADO JUGADORES SIMILARES
    df_similares1.index = range(1, len(df_similares1) + 1)
    clave_tabla = f"tabla_jugadores_{pais_liga}_{pos_ppal_modelo}_{grupo_rol_sel}"
    evento_tabla = st.dataframe(
                            df_similares1[col_deseadas],
                            width = 'stretch',
                            hide_index=True,
                            on_select="rerun",
                            selection_mode="single-row",
                            key=clave_tabla,
                            )

    # GUARDAR EL NOMBRE DEL JUGADOR CUANDO SE HAGA CLICK EN UNA FILA DEL LISTADO
    # Comprobamos si el usuario ha seleccionado un jugador de la tabla
    fila_seleccionada = evento_tabla.selection.rows
    if fila_seleccionada:
        indice_fila = fila_seleccionada[0]
        jugador_clicado = df_similares1.iloc[indice_fila]["Jugador"]

        # Guardamos en variables de sesion
        # jugador seleccionado
        st.session_state["jugador_seleccionado"] = jugador_clicado
        # grupo metricas
        st.session_state["metricas"] = grupo_rol_sel
        # posiciona analizada
        st.session_state["pos_seleccionado"] = pos_ppal_modelo
        
        # --- BOTÓN PARA IR A LA PESTAÑA 2 ---
        # Al hacer clic en este botón, puedes informar al usuario o guiarle
        if st.button(
            f"🔍 Ir a la pestaña **Detalle Jugador** para ver ficha completa de {jugador_clicado}",
            type="primary",
            key="btn_ir_detalles",
        ):
            st.success(
                f"¡Jugador {jugador_clicado} seleccionado! Dirígete a la pestaña"
                " **Detalles Jugador** arriba para ver su información y radar."
        )
    else:
        # SI EL USUARIO DESMARCA LA FILA: Limpiamos la sesión al instante
        if st.session_state.get("jugador_seleccionado") is not None:
            st.session_state["jugador_seleccionado"] = None
            st.rerun()

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
    bg = "#899BA0" #fondo 2d3137
    tit = "#f4cf7f" #amarillo crema resto
    key = "#c1ff72" #verde lima s

    #ranking jugadores
    impact = "#ef233c" #rojo num 1
    seg_ter = "#c1ff72" #verde lima num2-3
    resto = "#FFF7B1" #resto hasta el 10 

    white = "#ffffff"

    # Crear un gráfico de barras para los 10 jugadores más similares
    top_n = 10

    df_top_similares1 = df_similares1[df_similares1['Jugador-Posiciones'] != jugador_modelo].head(top_n)

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

    # Título principal (más grande, colocado arriba del gráfico)
    plt.text(
        0.5,
        1.08,  # Por encima del borde superior (1.0 es el límite del gráfico)
        f"Top {top_n} Perfiles más similares a {nombre_jugador} ({pos_ppal_modelo})",
        transform=plt.gca().transAxes,  # Activado para usar coordenadas relativas
        fontsize=11,
        fontweight="bold",
        color=tit,
        ha="center",
        va="bottom",
    )

    # Subtítulo (más pequeño, justo debajo del título principal)
    plt.text(
        0.5,
        1.03,  # Un poco más abajo que el título principal pero aún fuera del gráfico
        f"{liga}  —  Métricas: {grupo_rol_sel}",
        transform=plt.gca().transAxes,  # Activado
        fontsize=9,
        fontweight="normal",
        color=tit,
        ha="center",
        va="bottom",
    )

    #FIRMA
    #plt.figtext(0.99, 0.01, "Hecho por @ManuelFLP", ha="right", fontsize=9, color="grey")
    #marca de agua
    plt.text(0.5, 0.5, "Hecho por @ManuelFLP",
        transform=plt.gca().transAxes, fontsize=30, color="grey",
        alpha=0.2, ha="center", va="center", rotation=30)

    bars = plt.barh(df_top_similares1["jugador_equipo"], df_top_similares1["Similitud"], color=resto)
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
            b.set_color(seg_ter)

    plt.tight_layout()
    #plt.show()
    # Capturas la figura actual y la mandas a Streamlit:
    st.pyplot(plt.gcf())

    # Limpias el lienzo para que no se mezcle con el siguiente gráfico:
    plt.clf()

    # Limpiar memoria
    plt.close()

# ══════════════════════════════════════════════════════
# TAB 2 — DATOS JUGADOR SELECCIONADO
# ══════════════════════════════════════════════════════

with tab2: 
    #st.subheader("Ficha Técnica y Análisis Individual")

    # 1. Comprobamos si hay un jugador seleccionado en la memoria de sesión
    if (
        "jugador_seleccionado" in st.session_state
        and st.session_state["jugador_seleccionado"] is not None
        ):
            jugador_activo = st.session_state["jugador_seleccionado"]

            st.markdown(f"##### 👤 Analizando a: **{jugador_activo} - {st.session_state["pos_seleccionado"]}**")
            #st.session_state["metricas"] = grupo_rol_sel
            st.caption(f"Grupo de métricas - {st.session_state["metricas"]}")

            # 2. Filtramos el DataFrame general para extraer los datos de este jugador
            # (Asegúrate de cambiar 'df_completo' por el nombre real de tu dataframe principal con todos los jugadores)
            df_ficha = df_similares1[df_similares1["Jugador"] == jugador_activo]

            if not df_ficha.empty:
                # Mostramos sus datos tabulares de forma limpia
                st.dataframe(df_ficha[col_deseadas], width = 'stretch', hide_index=True)

                st.markdown("---")
                st.markdown("##### 🎯 Gráfico Radar de Rendimiento")

                # Rankear todas las columnas en 'params' (percentil, mejor valor = rank más alto)
                col_metricas = st.session_state["metricas_perfil"] 
                df_similares[col_metricas] = df_similares[col_metricas].rank(pct=True, ascending=True) * 100

                player1_values = df_similares.loc[df_similares['Jugador'] == st.session_state["jugador_seleccionado"], col_metricas].values.flatten().tolist()

                title = st.session_state["jugador_seleccionado"] + " - " + st.session_state["pos_seleccionado"] + f" - {grupo_rol_sel}"
 
                player1_color = "#c1ff72" #verde lima s # df_similares.loc[df_similares['player_name'] == st.session_state["jugador_seleccionado"], 'Team_Color'].values[0]
                #player2_color = df_similares.loc[df_similares['player_name'] == player2, 'Team_Color'].values[0]

                title_color = '#00171f'
                fig_bg_color = "#899BA0" #fondo 2d3137
                radar_bg_color = '#f8f9fa'

                # Etiquetas del radar (usamos los parámetros como etiquetas)
                labels = col_metricas

                # Número de variables
                num_vars = len(labels)

                # Ángulos del radar
                angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
                angles += angles[:1]

                # Cerrar los polígonos
                player1_values += [player1_values[0]]
                #player2_values += [player2_values[0]]

                # Crear figura
                fig, ax = plt.subplots(figsize=(8, 8), dpi=110, subplot_kw=dict(polar=True))

                # Texto personalizado (jugadores y equipos)
                ax.text(0.5, 1.15, title, transform=ax.transAxes, ha='center', fontsize=10, color=title_color)
                #ax.text(0.4, 1.1, f"{st.session_state["jugador_seleccionado"]}", transform=ax.transAxes, ha='right', fontsize=14, color=player1_color)
                #ax.text(0.6, 1.1, f"{player2}", transform=ax.transAxes, ha='left', fontsize=14, color=player2_color,fontproperties=font_title.prop)
                #ax.text(0.4, 1.07, f"{Team1}", transform=ax.transAxes, ha='right', fontsize=12, color=player1_color,fontproperties=font_title.prop)
                #ax.text(0.6, 1.07, f"{Team2}", transform=ax.transAxes, ha='left', fontsize=12, color=player2_color,fontproperties=font_title.prop)

                #add_image_from_url(logo_team1, x_pos=0.45, y_pos=1.1, size=0.2, relative_to='axes')
                #add_image_from_url(logo_team2, x_pos=0.55, y_pos=1.1, size=0.2, relative_to='axes')

                # Dibujar las líneas
                ax.plot(angles, player1_values, color=player1_color, linewidth=2)
                ax.fill(angles, player1_values, color=player1_color, alpha=0.4)

                #ax.plot(angles, player2_values, color=player2_color, linewidth=2)
                #ax.fill(angles, player2_values, color=player2_color, alpha=0.4)


                # Añadir los valores numéricos al radar
                for i in range(num_vars):
                    angle = angles[i]
                    # Jugador 1
                    ax.text(angle, player1_values[i] + 3, f"{int(player1_values[i])}", color="black", fontsize=10, ha='center', va='center')
                    # Jugador 2
                    #ax.text(angle, player2_values[i] + 3, f"{int(player2_values[i])}", color=player2_color, fontsize=10, ha='center', va='center',fontproperties=font_title.prop)

                # Añadir etiquetas
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(labels, fontsize=8)

                # Estilo del radar
                ax.set_yticks([20, 40, 60, 80, 100])
                ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=7)
                ax.yaxis.grid(True, linestyle='dashed', color='gray', alpha=0.5)
                ax.xaxis.grid(True, linestyle='dashed', color='gray', alpha=0.5)

                # Ajustes finales
                ax.spines['polar'].set_visible(False)

                # Fondo gris del área polar
                ax.set_facecolor(radar_bg_color)
                fig.set_facecolor(fig_bg_color)

                plt.tight_layout()

                st.pyplot(fig, width = 'stretch')

                # Un pequeño botón por si quieres resetear la selección y volver a empezar
                #if st.button("🔄 Cambiar de jugador"):
                #    st.session_state["jugador_seleccionado"] = None
                #    st.rerun()

            else:
                st.warning(
                    f"No se han encontrado registros detallados para {jugador_activo}."
                )

    else:
        # Mensaje de guía si entra a la pestaña sin haber seleccionado nada antes
        st.info(
            "👈 Ve a la primera pestaña (**Búsqueda por Similitud**), haz clic en una"
            " fila de la tabla de jugadores y pulsa ver ficha para cargar sus"
            " datos aquí automáticamente."
        )