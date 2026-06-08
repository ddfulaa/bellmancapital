# Resumen Completo del EDA Propio (`notebooks/own_eda.ipynb`)

> **Nota:** Todo el análisis se realizó exclusivamente sobre el conjunto de entrenamiento (80/20 split cronológico). Ninguna decisión de diseño se tomó mirando el test set.

---

## 0. Datos y Configuración

| Campo | Valor |
|-------|-------|
| Intervalo | 15 minutos |
| Shape | 279,907 filas × 16 columnas |
| Rango temporal | 2018-01-01 → 2025-12-31 |
| Train | 223,925 filas (2018-01-01 → 2024-05-26) |
| Test | 55,982 filas (2024-05-26 → 2025-12-31) |
| Activos | 3 (asset_0, asset_1, asset_2) + cash |
| Features por activo | close, high, low, volume, taker_buy_ratio |

- **Sin valores faltantes ni infinitos.** Todas las columnas tienen varianza.

---

## 1. Distribución de Retornos (Close-to-Close)

| Métrica | Asset 0 | Asset 1 | Asset 2 |
|---------|---------|---------|---------|
| Media por barra | ~1 bp | ~1 bp | ~1 bp |
| Desv. estándar | 84 bps | 113 bps | 95 bps |
| Skewness | −0.64 | +0.40 | −0.29 |
| Kurtosis | 137.9 | 36.7 | 51.2 |
| Percentil 1% | ~−2% | ~−2% | ~−2% |
| Percentil 99% | ~+2% | ~+2% | ~+2% |
| Mínimo extremo | hasta −20% en una sola barra |

**Hallazgo clave:** La media de ~1 bp por barra es absurdamente pequeña frente al costo de comisiones (10 bps por trade, 20 bps ida y vuelta). Cualquier edge que provenga de perseguir movimientos promedio queda destruido por fricción. Las distribuciones son fuertemente leptokúrticas (kurtosis >> 3): la mayoría del P&L se concentra en eventos raros de cola.

---

## 2. Excursión Intra-Barra

La excursión promedio (high−low)/close es de **80–120 bps por barra**, mientras que el desplazamiento neto close-to-close es de ~1 bp. El mercado oscila **80–120× más** de lo que se desplaza neto dentro de cada vela de 15 min. Asset 1 presenta excursiones extremas > 40%.

> **Directriz:** La función `_reward` **no puede** depender solo del precio de cierre. Debe integrar high/low para penalizar la inviabilidad de supervivencia ante excursiones adversas extremas. Un agente que ignora esto sobrevive en simulación pero colapsa en producción.

---

## 3. Autocorrelación de Retornos

La autocorrelación del retorno close-to-close **muere instantáneamente** después de lag 1. La única autocorrelación estadísticamente significativa es lag-1, negativa (~−0.02), consistente con mean-reversion de microestructura.

> **Directriz:** No usar retornos crudos como features predictivas; la señal está en volatilidad, volumen y flujo de órdenes.

---

## 4. Evolución de Precios Normalizados

- **Asset 1** es el más volátil: sube ~60× en 2021 y cae ~95% después.
- **Asset 0 y Asset 2** están altamente correlacionados (se mueven juntos en la mayoría de periodos).
- Régimen Bull extremo (2020–2021) seguido de Bear extremo (2022).
- El agente será entrenado en regímenes que **no existirán en test**.

---

## 5. Correlación entre Activos

| Ventana | Correlación promedio |
|---------|---------------------|
| General | ~0.55–0.70 |
| En crashes | sube a ~0.90+ |
| En periodos calmos | cae a ~0.20 |

La diversificación funciona a veces pero **falla exactamente cuando más se necesita** (en crashes, todos los activos caen juntos).

---

## 6. Clustering de Volatilidad

La volatilidad tiene **memoria larga**: periodos de alta volatilidad persisten semanas a meses. La volatilidad rodante (rolling std) es un predictor razonable de la volatilidad futura.

> **Directriz:** `vol_21` debe ser feature obligatoria en `_obs()` porque el régimen de volatilidad actual persiste y condiciona el riesgo futuro.

---

## 7. Volumen y Taker Buy Ratio (TBR)

- El volumen presenta explosiones seguidas de calma (similar a volatilidad).
- TBR oscila alrededor de 0.50; los extremos (>0.65 o <0.35) son informativos.
- **Señal de absorción:** TBR extremo + alto volumen + retorno plano = reversión inminente.

> **Directriz:** Incluir `vol_ratio` y `taker_buy_ratio` como features. El TBR aislado es ruidoso; la **divergencia TBR-precio** es la verdadera señal.

---

## 8. Causalidad de Granger (Transmisión de Volatilidad)

| Relación | Lag 1 (15 min) | Lags superiores |
|----------|----------------|-----------------|
| `abs_ret_1 → abs_ret_0` | p < 0.001 ✓ | Pierde significancia |
| `abs_ret_1 → abs_ret_2` | p < 0.001 ✓ | Pierde significancia |
| `abs_ret_0 → abs_ret_1` | p < 0.001 ✓ | Pierde significancia |
| `ret_1 → ret_0` (direccional) | p < 0.05 | No significativo |

**Asset 1 es el epicentro del riesgo sistémico.** La transmisión de volatilidad es bidireccional pero más fuerte desde asset_1. La señal predictiva es exclusivamente de **baja latencia** (lag 1 = primeros 15 min). Para lag > 1, el alpha se evapora.

> **Directriz:** El agente debe recibir `abs_ret` con lag 1 y reaccionar inmediatamente a shocks de volatilidad. Señales retardadas son inútiles contra 10 bps de comisión.

---

## 9. Análisis Espectral (PSD)

- **Retornos direccionales:** PSD esencialmente plano (ruido blanco). No hay patrón periódico explotable en dirección de precio.
- **Volatilidad:** Picos espectrales claros en periodos específicos:

| Ciclo | Periodo | Interpretación |
|-------|---------|----------------|
| Ultra-macro | ~410h (~17 días) | Ritmo de mercado profundo |
| Macro | ~171h (~7 días) | Ciclo semanal |
| Meso | ~85h (~3.5 días) | Ciclo inter-semanal |
| Micro | ~24h | Ciclo diario |

> **Directriz:** Inyectar transformaciones de Fourier sincronizadas a 410h, 171h, 85h y 24h en `_obs()`. Obligar a la red a inferir este reloj desde precios puros desperdiciaría capacidad paramétrica.

---

## 10. Drawdowns

| Activo | Max Drawdown |
|--------|-------------|
| Asset 0 | ~−80% |
| Asset 1 | ~−95% |
| Asset 2 | ~−90% |

Los drawdowns duran **meses a años**, no solo días. La recuperación desde drawdowns profundos es extremadamente lenta.

> **Directriz:** La función de recompensa debe penalizar drawdowns severamente. Un agente que mantiene posición durante un drawdown del 80% ha fracasado catastróficamente, incluso si eventualmente recupera.

---

## 11. Impacto de Costos de Transacción (10 bps)

| Frecuencia de trading | Costo anual aprox. |
|------------------------|--------------------|
| Cada barra (96 trades/día) | ~350% del capital |
| Cada 4 horas (6 trades/día) | ~22% del capital |

> **Directriz:** El agente debe ser extremadamente selectivo sobre cuándo operar. Una restricción de turnover o penalización es esencial.

---

## 12. Detección de Regímenes de Mercado

| Régimen | % del tiempo | Características |
|---------|-------------|-----------------|
| Baja volatilidad | ~50% | Retornos ~planos, mejor para mantener posiciones |
| Media volatilidad | ~35% | Transiciones peligrosas |
| Alta volatilidad | ~15% | Pérdidas concentradas, reducir exposición |

Las transiciones de régimen son persistentes: una vez en alta volatilidad, tiende a permanecer por periodos extendidos.

> **Directriz:** `vol_21` e indicadores de régimen deben ser parte de `_obs()`.

---

## 13. Features Candidatas (Análisis de Poder Predictivo)

- La mayoría de indicadores técnicos tradicionales (RSI, MACD, Bollinger Bands) tienen correlación cercana a cero con retornos futuros.
- RSI tiene poder predictivo marginal solo en extremos (<20 o >80).
- MACD crossovers son demasiado lentos (la señal llega después del movimiento).
- **Features simples** (vol_21, atr_14, vol_ratio, taker_buy_ratio) superan a indicadores complejos.

> **Directriz:** Principio KISS. Usar features simples e interpretables en lugar de indicadores complejos que sobreajustan.

---

## 14. Tests de Estacionariedad (ADF y KPSS)

| Serie | ADF (p-value) | Estacionaria? |
|-------|---------------|---------------|
| Precios | p > 0.05 | ❌ No |
| Retornos | p < 0.001 | ✅ Sí |
| Retornos absolutos | p < 0.001 | ✅ Sí (con memoria larga) |

> **Directriz:** Nunca usar precios crudos como features. Siempre usar retornos o transformaciones normalizadas.

---

## 15. Síntesis: Lista Maestra de Features para `_obs()`

| # | Feature | Justificación |
|---|---------|---------------|
| 1 | `ret` (close-to-close return, 3 activos) | Señal direccional básica (estacionaria) |
| 2 | `abs_ret` (retorno absoluto) | Proxy de volatilidad instantánea |
| 3 | `vol_21` (volatilidad rodante 21 periodos) | Régimen de volatilidad actual |
| 4 | `atr_14` (ATR normalizado) | Excursión intra-barra esperada |
| 5 | `vol_ratio` (volumen relativo a media rodante) | Confirmación de convicción |
| 6 | `taker_buy_ratio` (TBR) | Proxy de flujo institucional |
| 7 | `fourier_sin/cos_24h` | Ciclo diario de volatilidad |
| 8 | `fourier_sin/cos_85h` | Ciclo meso de volatilidad |
| 9 | `fourier_sin/cos_171h` | Ciclo macro de volatilidad |
| 10 | Pesos actuales del portafolio | Conciencia de posición |
| 11 | Drawdown desde pico | Conciencia de riesgo |

---

## 16. Notas Metodológicas Críticas

> [!CAUTION]
> - Todo el análisis se realizó **exclusivamente sobre datos de entrenamiento**.
> - Ningún hiperparámetro se ajustó mirando el test set.
> - Todas las features deben computarse **sin lookahead** (solo información causal).
> - `StandardScaler` debe hacer **fit SOLO en datos de entrenamiento**.

---

## 17. Resumen Ejecutivo de Directrices para el Agente

1. **Recompensa:** Integrar high/low (no solo close-to-close). Penalizar drawdowns severamente.
2. **Latencia:** Reaccionar a shocks de asset_1 en lag 1 (15 min). Después de eso, el alpha se evapora.
3. **Turnover:** Ser extremadamente selectivo. 10 bps por trade destruyen el edge promedio.
4. **Features:** Simples > complejas. Volatilidad y flujo de órdenes > indicadores técnicos.
5. **Régimen:** Inyectar reloj de Fourier y vol_21 para contextualizar el mercado.
6. **Robustez:** Asumir que el agente fallará. Documentar los fracasos con rigor metodológico.
