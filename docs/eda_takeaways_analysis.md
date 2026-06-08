# Análisis de las Conclusiones del EDA y Diseño del Agente (Bellman Capital)

Este documento responde a las cinco conclusiones y preguntas clave ("Key Takeaways") planteadas al final del Análisis Exploratorio de Datos del profesor (`notebooks/eda.ipynb`), alineando las respuestas con las reglas de ingeniería y la tesis metodológica descritas en la guía del proyecto (`GEMINI.md`).

---

## 1. La volatilidad depende del régimen.
> **EDA:** "Tu agente se entrenará en algunos regímenes y será evaluado en otros. ¿Cómo debe capturar esto tu estado (state)?"

**Respuesta (Tesis y Diseño del Espacio de Observación):**
Para lidiar con los cambios de régimen y el desafío intrínseco de la no estacionariedad de los mercados (violación del supuesto perfecto de Markov), el espacio de estado (`_obs`) no puede depender únicamente de los retornos crudos. Basándonos en las features ya implementadas en `src/data.py`, nuestro estado captura esto integrando descriptores de la volatilidad actual:
*   **`vol_21`** (Volatilidad rodante a 21 periodos).
*   **`atr_14`** (Average True Range normalizado).

Estas características contextualizan el mercado: un movimiento del 2% en un régimen calmo tiene implicaciones distintas que en uno turbulento. De vital importancia metodológica: al normalizar estas métricas con un `StandardScaler`, **sólo** aplicaremos `fit` sobre el conjunto de entrenamiento, preservando de manera absoluta la restricción de **Cero Lookahead**.

---

## 2. Los activos están correlacionados pero no son idénticos.
> **EDA:** "La diversificación tiene valor, pero no tanto como con activos verdaderamente no correlacionados."

**Respuesta (Tesis y Espacio de Acción):**
Debido a la correlación en mercados de riesgo, las caídas suelen darse en bloque, lo que limita la protección de una simple diversificación "Equal weight". Esto define la estructura que tendrá nuestro portafolio y nuestras acciones:
El agente requiere capacidad táctica agresiva. Se habilitarán **posiciones cortas** (pesos negativos limitados a -1.0) y la posibilidad de refugio total en **Cash** (que nunca puede ser negativo). Cuando el bloque de activos de riesgo se mueva a la baja, el agente debe aprender a migrar su exposición a efectivo o ponerse en corto para defender el portafolio.

---

## 3. Las caídas (Drawdowns) son severas.
> **EDA:** "Una estrategia ingenua de retención puede perder más del 80%. Tu función de recompensa debe tener esto en cuenta."

**Respuesta (Diseño de la Función de Recompensa - `_reward`):**
En perfecta sintonía con las *Expectativas Profundas* expuestas en `GEMINI.md`, el agente no debe ser premiado puramente por los retornos alcistas generados. Como nuestra métrica de evaluación principal será el **Ratio de Sortino** (el cual penaliza fuertemente la volatilidad a la baja), la función de recompensa paso a paso debe reflejar esta asimetría.
En la implementación de `_reward(prev, curr)` se penalizarán explícitamente los *drawdowns* relativos. Esto se puede lograr con funciones de recompensa ajustadas por riesgo que resten a la rentabilidad generada una penalización proporcional a las pérdidas o la varianza negativa (e.g., $Reward_t = Retorno_t - \lambda \times RiesgoA_t$).

---

## 4. El volumen y el Taker Buy Ratio contienen información.
> **EDA:** "Si incluirlos o no en tu estado es una decisión de diseño — justifícala."

**Respuesta (Justificación e Inclusión de Features):**
Siguiendo la regla de *Diseño Basado en Hipótesis* (cada variable debe tener una justificación), **sí incluiremos ambos**.
*   **`vol_ratio`:** El mercado accionado presenta rupturas falsas (ruido). El volumen relativo sirve para confirmar la convicción detrás de un movimiento de precio. Si un movimiento ocurre sin un pico de volumen relativo, el agente puede inferir que es mero ruido de baja liquidez.
*   **`taker_buy_ratio` (TBR):** El TBR actúa como un proxy del flujo institucional de órdenes (un estado oculto o *hidden state* del mercado). Proporciona la presión direccional y nos acerca a mitigar el problema de variables ocultas que rompen el proceso puro de Markov.

---

## 5. Los datos son ruidosos.
> **EDA:** "No esperes que tu agente encuentre una señal limpia. Enfócate en la robustez sobre el rendimiento máximo."

**Respuesta (Robustez, Fricción y Transparencia):**
Este es el pilar central de nuestra evaluación: **La Metodología Supera al Resultado**.
*   **Impacto de Costos:** El ruido de mercado engaña a los agentes y provoca exceso de trading (*churn*). Dado que incluiremos innegociablemente **10 bps de costo por transacción**, el agente se verá forzado a operar sólo cuando sus redes predigan un movimiento que exceda ampliamente esta fricción.
*   **Principio KISS y Transparencia:** El ruido conduce al sobreajuste si usamos modelos excesivamente profundos o no defendibles. Preferiremos arquitecturas de red sencillas, robustas y comprensibles. Adicionalmente, sabemos de antemano que el agente cometerá errores y sufrirá frente al ruido; será nuestra labor documentar científica y explícitamente estos fallos en nuestro informe final, tal y como lo requiere el proyecto.
