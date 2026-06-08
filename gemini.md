# Guía de Referencia y Restricciones del Proyecto (Bellman Capital)

Este documento es una brújula para el desarrollo continuo del agente de Reinforcement Learning. Resume los requisitos innegociables del profesor extraídos del `README.md` y las directrices de ingeniería establecidas para el código.

## 1. Reglas de Ingeniería y Filosofía de Código

*   **Principio KISS (Keep It Simple, Stupid):** Evitar sobreingeniería. Las soluciones deben ser directas, comprensibles y fáciles de depurar.
*   **Principio UNIX:** Crear funciones pequeñas que hagan una sola cosa y la hagan bien. Favorecer la composabilidad.
*   **Programación Funcional:** Evitar estados mutables innecesarios, usar funciones puras donde sea posible y preferir el paso de parámetros explícitos.
*   **Tipado Estricto (Type Hints):** Todo el código Python debe llevar anotaciones de tipo (`-> int`, `List[str]`, etc.).
*   **Uso de Pydantic:** Utilizar Pydantic para la validación de datos, configuraciones y estructuras donde sea adecuado y aporte seguridad.
*   **INMUTABILIDAD DE `src/`:** Queda terminantemente prohibido modificar cualquier archivo dentro de la carpeta `src/`. Todo el desarrollo debe ocurrir en `agent.py` o en scripts auxiliares.

## 2. Requisitos y Restricciones del Profesor (README.md)

### A. Restricciones Innegociables
1.  **Cero Lookahead (Fuga de Datos):** Las características (features) en el instante $t$ solo pueden usar información disponible en $t$ o antes. Especial cuidado con escaladores (`StandardScaler`) que deben ajustarse (fit) únicamente con datos de entrenamiento.
2.  **Costos de Transacción:** Se deben modelar al menos **10 basis points (bps)** por trade. El agente debe demostrar ser robusto a esta fricción (correr simulaciones a 0 bps y 10 bps).
3.  **Reproducibilidad:** Los resultados deben ser 100% reproducibles a partir del código entregado.

### B. Diseño del Entorno (TradingEnv)
*   Debe heredar de `BaseTradingEnv` (que reside en `src/env.py`).
*   Debe implementarse obligatoriamente en `agent.py`.
*   Métodos requeridos a implementar:
    *   `_obs(self) -> np.ndarray`: El vector de estado/observación en cada paso.
    *   `_weights_from_action(self, action) -> np.ndarray`: Mapeo de la acción a pesos del portafolio.
    *   `_reward(self, prev, curr) -> float`: Señal escalar de recompensa.

### C. Espacio de Acción y Portafolio
*   Se permiten posiciones en corto (pesos negativos en activos de riesgo).
*   El peso en Cash **no puede ser negativo** (no se puede pedir prestado cash para apalancarse).
*   La suma de todos los pesos (activos + cash) debe ser exactamente **1.0**.
*   El peso de cada activo de riesgo individual está acotado al rango **[-1.0, 1.0]** (sin apalancamiento superior a 1x).

### D. Métricas y Evaluación
*   **Métrica Principal:** Sortino Ratio (penaliza solo volatilidad a la baja).
*   **Métricas Secundarias:** Retorno acumulado, Max Drawdown, Total de comisiones pagadas.
*   **Protocolo:** Split cronológico (fijado en `configs/default.yaml`). Entrenar en train, evaluar en test. **Prohibido ajustar hiperparámetros mirando el test set.**
*   **Comparativas (Baselines):** Random policy, Hold cash, Hold Asset 0, Equal weight (rebalanced), SMA crossover.

### E. Entregable Final
*   **Archivo Único:** Se entregará exclusivamente el archivo `agent.py`.
*   **Clases Obligatorias:** Debe contener las clases `TradingEnv` y `Agent` sin cambiarles el nombre.
*   **Validación:** Debe pasar todas las pruebas ejecutando `uv run pytest tests/test_submission.py -v`. Una entrega que no pase los tests será descalificada.

---

## 3. Expectativas Profundas (Lo que realmente evalúa el proyecto)

Más allá de la mera implementación del código, el `README.md` expone claramente la verdadera naturaleza del proyecto: **no se evalúa la rentabilidad del agente, sino el rigor metodológico y científico del equipo.**

1. **La Metodología Supera al Resultado ("Rigorous methodology over high returns")**
   * El profesor especifica que "Un agente que fracasa en el conjunto de prueba (held-out period) pero demuestra una evaluación disciplinada puede obtener la máxima puntuación". 
   * **Implicación:** No debemos caer en el sobreajuste (overfitting) ni en la desesperación por lograr retornos positivos falsos. Es preferible documentar un fracaso honesto con un análisis profundo, que presentar un resultado "suertudo" sin justificación. El resultado final en el conjunto de prueba *no forma parte de la rúbrica de calificación*.

2. **Diseño Basado en Hipótesis (La Tesis del Agente)**
   * El proyecto requiere definir una "Tesis": *¿Qué cree tu agente sobre estos mercados y cómo esa creencia moldea su diseño?*
   * **Implicación:** Cada línea de código, cada variable en el espacio de estado y la elección del espacio de acción deben responder a esta tesis. Si incluimos una variable, debemos poder justificar en una sola frase por qué está allí basándonos en el Análisis Exploratorio de Datos (EDA).

3. **Defensabilidad y Comprensión Profunda**
   * "Se espera que cada miembro del equipo entienda y pueda defender la implementación completa".
   * **Implicación:** Refuerza la necesidad del principio KISS. No debemos usar arquitecturas súper complejas (como PPO) a menos que podamos justificar matemáticamente y conceptualmente por qué es necesario frente a un algoritmo más simple (como DQN o un heurístico básico).

4. **Transparencia en el Fracaso y Reflexión**
   * El informe debe incluir explícitamente "una figura documentando un fracaso o comportamiento anómalo" y exige discutir la brecha entre la teoría de DRL (Deep Reinforcement Learning) y este problema aplicado.
   * **Implicación:** Se espera que el agente sufra y cometa errores (por ejemplo, "Reward Hacking" o que no aprenda a operar). Nuestro trabajo principal es **identificar, explicar y documentar esos errores**, entendiendo conceptos clave como la falta de estacionariedad (regime change), la asignación de crédito a largo plazo (long-horizon credit assignment) y la eficiencia de muestreo.

5. **El Peligro del Lookahead y la Robustez**
   * Cualquier filtración del futuro (incluso sutil al usar un `StandardScaler` de forma global) descalifica el trabajo científico. El agente tiene que enfrentarse a fricciones reales (10 bps de costo). Si no se le obliga a lidiar con el impacto real de las transacciones, la evaluación no tiene valor metodológico.
6. **El Desafío Intrínseco de Markov y la No Estacionariedad**
   * El trading algorítmico viola fundamentalmente la suposición de que estamos ante un Proceso de Decisión de Markov (MDP) perfecto. En juegos como el ajedrez, el estado es determinista. En los mercados financieros, dos "estados" idénticos (mismos precios y portafolio) no siempre darán el mismo resultado porque el mercado depende de *estados ocultos* (flujos de órdenes institucionales, macroeconomía) y de *cambios de régimen*.
   * **Implicación:** Si solo modelamos el precio pasado y el portafolio, el agente fallará en ser verdaderamente markoviano. Es imperativo tener esto en mente durante el diseño del entorno y en la discusión teórica del fracaso: el agente debe lidiar con un entorno donde la misma acción sobre la misma observación puede tener recompensas drásticamente distintas dependiendo del régimen subyacente.

*Nota: Este archivo guiará todas las implementaciones de código futuras para asegurar que no se desvíe del objetivo científico del proyecto.*
