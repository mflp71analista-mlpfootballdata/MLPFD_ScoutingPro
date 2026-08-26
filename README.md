# ⚽ MLPScoutingPro: Scouting Profesional 

**MLP Scouting Pro** es una aplicación web interactiva desarrollada en **Python** y **Streamlit** orientada a la identificación y análisis comparativo de futbolistas. La herramienta permite a directores deportivos, analistas y secretarías técnicas encontrar sustitutos o perfiles similares a un **Jugador Modelo** de referencia, aplicando algoritmos de distancia e índice de coincidencia estadística sobre métricas ajustadas por 90 minutos.

---

## 🛠️ Stack Tecnológico & Librerías

* **Lenguaje:** Python
* **Framework Web / Interfaz:** Streamlit
* **Procesamiento de Datos:** Pandas, NumPy
* **Métricas Estadísticas & Algoritmos:** Scikit-Learn (Estandarización, métricas de similitud y distancia)
* **Visualización de Datos:** Matplotlib, Seaborn
* **Data Sources:** Datos avanzados de rendimiento técnico-táctico (Wyscout)

---

## 🚀 Funcionalidades Principales

### 1. Selección y Ficha del Jugador Modelo
* Permite seleccionar la temporada de análisis, seleccionar si es portero o no, jugador de referencia, una de las posibles posiciones del jugador modelo y el grupo de métricas a analizar.
* Genera una ficha técnica inmediata con datos de filiación (edad, posición específica, valor de mercado, vencimiento de contrato, pie, altura, peso) y sus métricas avanzadas por 90 minutos (xG, xA, duelos ganados, entradas, pases progresivos, etc.).

### 2. Algoritmo de Similitud y Filtrado Dinámico
* **Segmentación por Posición Específica:** Ajuste automático de la muestra según la demarcación seleccionada del jugador objetivo.
* **Filtros de Mercado:**
  * **País / Liga:** Filtrado por competiciones de destino (ej. *1ª RFEF, LaLiga, etc.*).
  * **Rango de Minutos Jugados:** Control continuo mediante *sliders* para evitar distorsiones por muestras pequeñas.
  * **Rango de Edad:** Filtro interactivo para acotar búsquedas por perfil de proyección o veteranía.
* **Selección por Perfil de Métricas:** Evaluaciones adaptables sobre el total de variables o sobre bloques específicos (Ataque, Creación, Pases Progresivos, Defensa, etc.).

### 3. Resultados & Salida de Datos
* **Tabla de Resultados Exportable:** Desglose métrico completo con el número exacto de jugadores coincidentes encontrados. Incluye funcionalidad directa para **descargar la tabla en CSV**.
* **Ranking de Similitud (Top 10):** Visualización mediante barras con el porcentaje exacto de coincidencia estadística respecto al *Jugador Modelo*.

### 4. Detalles jugador
* **Detalles del jugador:** Listado de todas sus métricas correspondientes al grupo seleccionado**.
* **RadarChart:** Visualización mediante radar chart de los datos de las métricas previamente pasadas todas a percentiles para poder medir las /90 y los porcentajes.

---

## 💡 Aplicación Real en la Dirección Deportiva

1. **Búsqueda de Sustitutos Directos (Replacement Search):** Identificación de reemplazos de garantías ante salidas imprevistas, detectando rendimiento equivalente en ligas de menor presupuesto (ej. captar perfiles de rendimiento élite en 1ª RFEF).
2. **Mitigación del Riesgo en Fichajes:** Evaluación basada en datos objetivos y métricas *per 90* normalizadas, reduciendo el sesgo en el scouting visual.
3. **Scouting de Proyección:** Filtrado combinado por edad y minutos para detectar talentos jóvenes con métricas similares a futbolistas consolidados.
4. **Url App:** Abre el fichero Url_App y dentro está la url de la app. Si quieres probarla ponte en contacto conmigo al email: mflp71.analista@gmail.com
