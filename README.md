# ⚽ MLPFootballData: Scouting Profesional & Jugadores Similares

**MLPFootballData** es una aplicación web interactiva desarrollada en **Python** y **Streamlit** orientada a la identificación y análisis comparativo de futbolistas. La herramienta permite a directores deportivos, analistas y secretarías técnicas encontrar sustitutos o perfiles similares a un **Jugador Modelo** de referencia, aplicando algoritmos de distancia e índice de coincidencia estadística sobre métricas ajustadas por 90 minutos.

---

## 🛠️ Stack Tecnológico & Librerías

* **Lenguaje:** Python
* **Framework Web / Interfaz:** Streamlit
* **Procesamiento de Datos:** Pandas, NumPy
* **Métricas Estadísticas & Algoritmos:** Scikit-Learn (Estandarización, métricas de similitud y distancia)
* **Visualización de Datos:** Matplotlib, Seaborn
* **Data Sources:** Datos avanzados de rendimiento técnico-táctico (Wyscout / FBref)

---

## 🚀 Funcionalidades Principales

### 1. Selección y Ficha del Jugador Modelo
* Permite seleccionar la temporada de análisis y el jugador de referencia (ej. *Nico Williams - Athletic Club*).
* Genera una ficha técnica inmediata con datos de filiación (edad, posición específica, valor de mercado, vencimiento de contrato, pie, altura, peso) y sus métricas avanzadas por 90 minutos (xG, xA, duelos ganados, entradas, pases progresivos, etc.).

### 2. Algoritmo de Similitud y Filtrado Dinámico
* **Segmentación por Posición Específica:** Ajuste automático de la muestra según la demarcación del jugador objetivo (ej. *LAMF - Extremo Izquierdo*).
* **Filtros de Mercado:**
  * **País / Liga:** Filtrado por competiciones de destino (ej. *1ª RFEF, LaLiga, etc.*).
  * **Rango de Minutos Jugados:** Control continuo mediante *sliders* para evitar distorsiones por muestras pequeñas.
  * **Rango de Edad:** Filtro interactivo para acotar búsquedas por perfil de proyección o veteranía.
* **Selección por Perfil de Métricas:** Evaluaciones adaptables sobre el total de variables o sobre bloques específicos (Ataque, Creación, Pases Progresivos, Defensa, etc.).

### 3. Resultados & Salida de Datos
* **Tabla de Resultados Exportable:** Desglose métrico completo con el número exacto de jugadores coincidentes encontrados. Incluye funcionalidad directa para **descargar la tabla en CSV**.
* **Ranking de Similaridad (Top 10):** Visualización mediante barras con el porcentaje exacto de coincidencia estadística respecto al *Jugador Modelo* (ej. *A. Ramírez - Juventud Torremolinos: 83.11%*).

---

## 💡 Aplicación Real en la Dirección Deportiva

1. **Búsqueda de Sustitutos Directos (Replacement Search):** Identificación de reemplazos de garantías ante salidas imprevistas, detectando rendimiento equivalente en ligas de menor presupuesto (ej. captar perfiles de rendimiento élite en 1ª RFEF).
2. **Mitigación del Riesgo en Fichajes:** Evaluación basada en datos objetivos y métricas *per 90* normalizadas, reduciendo el sesgo en el scouting visual.
3. **Scouting de Proyección:** Filtrado combinado por edad y minutos para detectar talentos jóvenes con métricas similares a futbolistas consolidados.
