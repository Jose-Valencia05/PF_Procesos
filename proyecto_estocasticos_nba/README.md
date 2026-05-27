# NBA-Poisson-TimeOut: Sistema Estocástico de Detección de Rachas y Optimización de Tiempos Fuera 🏀📈

Un framework modular en Python diseñado para auditar la dinámica de juego en la NBA en tiempo real. Utiliza la teoría de **Procesos Estocásticos de Poisson** aplicados a datos jugada a jugada (*Play-by-Play*) para identificar rachas de anotación del rival estadísticamente improbables y recomendar de forma óptima el momento exacto para pedir un tiempo fuera (*Timeout*) para "romper el momentum".

El sistema está diseñado bajo el principio de **Separation of Concerns (SoC)**, dividiendo la interfaz gráfica reactiva (**Capa Prefrontal / Dashboard**) de las operaciones matemáticas pesadas (**Capa Subyacente / Motor Estocástico**).

---

## 🏗️ 1. Arquitectura y Topología del Sistema

El diseño del software se fundamenta en un desacoplamiento estricto de responsabilidades. La lógica de negocio y cálculo matemático no tiene conocimiento del cliente que solicita los datos, mientras que la interfaz de visualización ignora las minucias del cálculo estocástico.

### A. La Capa de Función Ejecutiva (Corteza Prefrontal / Dashboard)
El archivo `dashboard.py` actúa como el **hypervisor** del sistema. Implementa una **Topología Monolítica Reactiva** mediante Streamlit, imitando la reactividad de los árboles de estado basados en el **Virtual DOM**. Cuando el usuario modifica un slider (e.g. el umbral crítico $\alpha$ o la ventana $t$) en el panel de control, el estado se actualiza y la capa reactiva intercepta la interrupción de software, reinyectando los parámetros en el motor matemático subyacente y repintando la vista de forma inmediata sin necesidad de reiniciar la aplicación.

### B. La Capa Autónoma (Tallo Cerebral / Motor Estocástico)
La lógica pesada y los cálculos computacionales se encapsulan en el directorio `/src`:
*   `extract.py` (**Extracción e I/O de Red**): Se comunica con la API de la NBA para obtener datos a nivel de jugada (*PBP V3*) con retrasos exponenciales adaptativos para evadir bloqueos de red (*Rate Limit*).
*   `clean.py` (**Normalización y Vectorización**): Traduce el reloj ISO-8601 del partido en minutos de juego continuos e interpole los puntos anotados para crear una serie temporal regular.
*   `analysis.py` (**Unidad Aritmética Lógica - ALU**): Contiene los algoritmos de la distribución acumulada de Poisson y la generación de gráficos.

### C. Topología Lógica del Flujo de Datos

```text
[Interrupción del Usuario] ---> (dashboard.py / Capa Reactiva)
      (Cambio de Sliders)             |
                                      v
                      +---------------------------------------+
                      | 1. src.extract.extract(team, limit)   | -> (I/O Red API / Caché)
                      | 2. src.clean.clean(team)              | -> (Normalización de Tiempos)
                      | 3. src.analysis.detectar_time_outs()  | -> (Cálculo de Poisson)
                      +---------------------------------------+
                                      |
[Renderizado Visual] <----------------+ (Retorno de DataFrames, Métricas y Gráficos)
```

---

## 🧮 2. Fundamento Matemático y Modelado Estocástico

### A. El Proceso de Poisson para Anotaciones Deportivas
Asumimos que el número de puntos que anota el oponente a lo largo del partido se comporta como un proceso de conteo estocástico continuo $\{N(t), t \ge 0\}$. Bajo la hipótesis nula ($H_0$), las anotaciones siguen un **Proceso de Poisson Homogéneo** con una tasa media constante de intensidad $\lambda$ (puntos concedidos por minuto) calculada de forma empírica a partir del histórico de la temporada actual:

$$\lambda = \frac{\text{Total de Puntos Recibidos en la Temporada}}{\text{Total de Minutos Jugados (Partidos} \times 48\text{ min)}}$$

Para una ventana de tiempo móvil de tamaño $t$ (minutos), la variable aleatoria $X$, que representa los puntos anotados por el oponente en ese intervalo, sigue una distribución de Poisson:
$$X \sim \text{Poisson}(\lambda_t) \quad \text{donde} \quad \lambda_t = \lambda \cdot t$$

La función de masa de probabilidad (PMF) para recibir exactamente $y$ puntos en un intervalo $t$ es:
$$P(X = y) = \frac{e^{-\lambda_t} \lambda_t^y}{y!}$$

### B. Análisis de Equidispersión y Selección de la Ventana Óptima ($t$)
Bajo la distribución de Poisson, se cumple la propiedad de equidispersión: $E[X] = \text{Var}[X] = \lambda_t$. Para validar la consistencia del modelo y seleccionar la ventana óptima $t$, se calcula el **Índice de Dispersión de Fisher** ($D$):

$$D = \frac{\text{Var}[X]}{E[X]}$$

*   **$t = 2$ minutos ($D \approx 1.03 > 1$)**: Presenta **sobredispersión**. A escalas muy cortas de juego, las fluctuaciones aleatorias (ruido de alta frecuencia, e.g., dos triples consecutivos) inflan artificialmente la varianza, lo que generaría falsos positivos si se usa para decidir tiempos fuera.
*   **$t = 3$ minutos ($D \approx 0.94 \approx 1$)**: Representa el **punto de equilibrio óptimo**. El índice está sumamente cercano a 1, lo que valida que el proceso se comporta de manera estrictamente Poissoniana en esta escala, filtrando el ruido corto pero reaccionando a tiempo.
*   **$t \ge 5$ minutos ($D \approx 0.78 < 1$)**: Presenta **subdispersión**. A escalas largas, las anotaciones convergen a su media histórica (debido al carácter estacionario y los límites físicos del juego). Deportivamente, esperar 5 minutos para detectar una racha es **tácticamente inútil**, pues el oponente ya habrá consolidado su ventaja.

*Nota: El dashboard interactivo calcula estos índices en tiempo real para el equipo seleccionado y sugiere dinámicamente al usuario la ventana de tiempo que mejor se ajusta a la equidispersión.*

### C. Lógica de Decisión del Timeout (p-valor)
Si el rival anota $k$ puntos en una ventana de $t$ minutos finalizando en el minuto $m$, calculamos la probabilidad acumulada de la cola derecha (la probabilidad de observar una racha de $k$ o más puntos bajo condiciones normales de juego):

$$p\text{-valor} = P(X \ge k) = 1 - P(X \le k - 1) = 1 - \sum_{i=0}^{k-1} \frac{e^{-\lambda_t} \lambda_t^i}{i!}$$

Fijamos un nivel de significancia de **$\alpha$** (configurable dinámicamente de $0.01$ a $0.10$ desde la interfaz). Si:
$$p\text{-valor} < \alpha$$
Rechazamos $H_0$ (concluimos que la racha del oponente no es una fluctuación aleatoria común, sino un evento de alta intensidad) y el algoritmo genera la señal de **`🔥 TIMEOUT`**.

Para acoplarse a las restricciones reales del deporte, se incorpora un **cooldown técnico** (configurable en el panel). Si se sugiere un tiempo fuera en el minuto $m$, no se podrá sugerir otro antes del minuto $m + m_{\text{cooldown}}$, evitando así la saturación de alertas en ventanas solapadas.

---

## 🏗️ 3. Estructura del Repositorio

```
proyecto_estocasticos_nba/
├── dashboard.py            # Hypervisor de control reactivo (Streamlit)
├── main.py                 # Orquestador alternativo de consola (CLI)
├── requirements.txt        # Dependencias del sistema
├── src/
│   ├── extract.py          # Extracción y rate limiting con nba_api (PBP V3)
│   ├── clean.py            # Parseo de tiempo ISO-8601 y estructuración temporal
│   └── analysis.py         # Lógica estocástica de Poisson y motor gráfico
├── data/
│   ├── raw/                # Archivos CSV crudos descargados de la API
│   └── processed/          # Series temporales limpias por minuto y resumen final
└── plots/
    └── timeout_analysis.png # Visualización del partido showcase seleccionado
```

---

## 🚀 4. Instalación y Uso

### Requisitos Previos
*   Python 3.10 o superior instalado.
*   Conexión a internet (para la primera extracción de datos de la API).

### Instalación de Dependencias
Instala los paquetes requeridos usando:
```bash
pip install -r requirements.txt
```
*(Nota: Si detectas que falta Streamlit en tu entorno, instálalo manualmente con `pip install streamlit`)*.

### Ejecución del Dashboard (Interfaz Principal)
Para inicializar la corteza prefrontal reactiva del sistema y levantar el servidor web local:
```bash
streamlit run dashboard.py
```
El hypervisor abrirá automáticamente una pestaña en tu navegador en `http://localhost:8501`.

### Parámetros Configurables en Tiempo Real:
Desde la barra lateral del Dashboard, el usuario puede manipular los siguientes vectores de entrada del sistema sin tocar una sola línea de código:
*   **Selección de Equipo**: Selector dinámico para cualquiera de las **30 franquicias de la NBA** (usando sus IDs oficiales).
*   **Límite de Partidos**: Cantidad de juegos históricos recientes a evaluar (1 a 15).
*   **Filtro de Red (Checkbox de Caché)**: Permite elegir entre forzar descargas directas de NBA.com o utilizar la caché local de CSVs en `data/raw/` para evitar límites de tráfico de API.
*   **Ventana Móvil ($t$)**: Ajuste dinámico de 2 a 5 minutos.
*   **Umbral Crítico ($\alpha$)**: Ajuste del p-valor límite de $0.01$ a $0.10$.
*   **Cooldown Técnico**: Lapso en minutos de separación obligatoria entre alertas.

---

## 📊 5. Interpretación de Resultados en el Dashboard

El Dashboard reactivo organiza los resultados calculados por el motor estocástico en cuatro secciones funcionales:

1.  **Métricas Clave (KPI Cards)**:
    *   **Tasa Base Rival (λ)**: Puntos promedio que nuestro equipo concede al oponente por cada minuto de juego.
    *   **Tiempos Fuera Sugeridos**: Total de alertas `🔥 TIMEOUT` detectadas en el encuentro.
    *   **Max Racha del Rival**: Cantidad máxima de puntos anotados por el oponente en el tamaño de ventana móvil actual.
2.  **Visualización Táctica (Pestaña 1)**:
    Muestra de forma gráfica la puntuación acumulada. Las líneas verticales rojas punteadas indican el momento exacto en el que el *coach* debió presionar el botón de tiempo fuera para cortar la racha estadística del oponente.
3.  **Registro de Tiempos Críticos (Pestaña 2)**:
    Aísla las alertas del partido en una tabla, mostrando el minuto y el p-valor exacto con 5 decimales de precisión, acompañado de tarjetas de advertencia de alta prioridad visual.
4.  **Auditoría Matemática (Pestaña 3)**:
    Permite visualizar la tabla comparativa de equidispersión y te sugiere de manera dinámica cuál es la ventana temporal óptima para ese equipo de acuerdo con los datos de su temporada regular.
5.  **Línea de Tiempo Completa (Pestaña 4)**:
    Permite auditar paso a paso el historial bruto de puntuaciones del partido resampleado en intervalos regulares de un minuto.
