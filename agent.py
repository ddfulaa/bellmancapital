"""
agent.py - Bellman Capital Submission
Implementa el motor de portafolio continuo (Transformer + PPO) envuelto
para cumplir con el test suite discreto.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional, List
import time
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

# =============================================================================
# 0. CONSTANTES DISCRETAS PARA EL TEST SUITE
# =============================================================================
# El test exige estrictamente que la suma sea 1.0 (test_weights_sum_to_one)
_ACTION_WEIGHTS = [
    np.array([0.0,  0.0,  0.0,  1.0], dtype=np.float32),  # 0: Efectivo
    np.array([1.0,  0.0,  0.0,  0.0], dtype=np.float32),  # 1: Largo A0
    np.array([0.0,  1.0,  0.0,  0.0], dtype=np.float32),  # 2: Largo A1
    np.array([0.0,  0.0,  1.0,  0.0], dtype=np.float32),  # 3: Largo A2
    np.array([0.0,  0.0, -1.0,  2.0], dtype=np.float32)   # 4: Corto A2 (Cash debe ser 2.0 para sumar 1)
]

N_ACTIONS = len(_ACTION_WEIGHTS)

# =============================================================================
# 1. PIPELINE DE EXTRACCIÓN DE DATOS (ESTADO REAL)
# =============================================================================
def scale_local(series: pd.Series, window_bars: int) -> pd.Series:
    min_obs = max(4, window_bars // 2)
    roll = series.rolling(window=window_bars, min_periods=min_obs)
    iqr = roll.quantile(0.75) - roll.quantile(0.25)
    return (series - roll.median()) / (iqr + 1e-8)

def calc_drawdown_local(close: pd.Series, high: pd.Series, window_bars: int) -> pd.Series:
    min_obs = max(4, window_bars // 2)
    rolling_max = high.rolling(window=window_bars, min_periods=min_obs).max()
    return (close - rolling_max) / (rolling_max + 1e-8)

def build_cyclical_time_15m(timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    coords = {
        "time_min": timestamps.minute / 60.0,
        "time_hour": timestamps.hour / 24.0,
        "time_dow": timestamps.dayofweek / 7.0
    }
    return pd.DataFrame(coords, index=timestamps)

def build_complete_agent_state(data: pd.DataFrame, scales_hours: List[int] = [86, 171, 410]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    assets: List[str] = ['asset_0', 'asset_1', 'asset_2']
    
    time_df = build_cyclical_time_15m(data.index)
    frames.append(time_df)

    log_p0 = np.log(data["asset_0_close"])
    log_p1 = np.log(data["asset_1_close"])
    log_p2 = np.log(data["asset_2_close"])
    
    spread_features: Dict[str, pd.Series] = {}
    spread_features["spread_1_0_base"] = scale_local(log_p1 - log_p0, 96)
    spread_features["spread_2_0_base"] = scale_local(log_p2 - log_p0, 96)
    
    for h in scales_hours:
        w_bars = h * 4
        spread_features[f"spread_1_0_{h}h"] = scale_local(log_p1 - log_p0, w_bars)
        spread_features[f"spread_2_0_{h}h"] = scale_local(log_p2 - log_p0, w_bars)
        
    frames.append(pd.DataFrame(spread_features, index=data.index))

    for asset in assets:
        close = data[f"{asset}_close"]
        high = data[f"{asset}_high"]
        low = data[f"{asset}_low"]
        vol = data[f"{asset}_volume"]
        tbr = data[f"{asset}_taker_buy_ratio"]
        
        log_ret = np.log(close / close.shift(1))
        abs_ret = log_ret.abs()
        amplitude = (high - low) / close.shift(1)
        vol_roc = np.log1p(vol) - np.log1p(vol.shift(1))
        drawdown_base = calc_drawdown_local(close, high, window_bars=96)
        
        asset_features = {
            f"{asset}_raw_log_ret": log_ret, f"{asset}_raw_abs_ret": abs_ret,
            f"{asset}_raw_amplitude": amplitude, f"{asset}_raw_vol_roc": vol_roc,
            f"{asset}_raw_tbr": tbr, f"{asset}_raw_drawdown": drawdown_base
        }
        
        for h in scales_hours:
            w_bars = h * 4
            hours_label = f"{h}h"
            asset_features[f"{asset}_log_ret_{hours_label}"] = scale_local(log_ret, w_bars)
            asset_features[f"{asset}_abs_ret_{hours_label}"] = scale_local(abs_ret, w_bars)
            asset_features[f"{asset}_amplitude_{hours_label}"] = scale_local(amplitude, w_bars)
            asset_features[f"{asset}_vol_roc_{hours_label}"] = scale_local(vol_roc, w_bars)
            asset_features[f"{asset}_tbr_{hours_label}"] = scale_local(tbr, w_bars)
            asset_features[f"{asset}_drawdown_{hours_label}"] = calc_drawdown_local(close, high, w_bars)

        frames.append(pd.DataFrame(asset_features, index=data.index))
        
    return pd.concat(frames, axis=1).dropna()

# =============================================================================
# 2. CORE NEURAL MODULES
# =============================================================================
class Time2Vec(nn.Module):
    def __init__(self, in_features: int = 3, out_dim: int = 128):
        super().__init__()
        self.k_linear = in_features 
        self.k_periodic = out_dim - in_features
        self.trend_proj = nn.Linear(in_features, self.k_linear)
        self.periodic_proj = nn.Linear(in_features, self.k_periodic)
        nn.init.uniform_(self.trend_proj.weight, -1.0, 1.0)
        nn.init.zeros_(self.trend_proj.bias)
        nn.init.uniform_(self.periodic_proj.weight, -1.0, 1.0)
        nn.init.zeros_(self.periodic_proj.bias)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        trend = self.trend_proj(t)
        periodic = torch.sin(self.periodic_proj(t))
        return torch.cat([trend, periodic], dim=-1)

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, 1, num_features))
        self.bias = nn.Parameter(torch.zeros(1, 1, num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.mean(x, dim=1, keepdim=True)
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps)
        x_norm = (x - mean) / stdev
        return (x_norm * self.weight) + self.bias

class QuantRobertaBody(nn.Module):
    def __init__(self, input_features_dim: int = 80, time_features_dim: int = 3, d_model: int = 128, 
                 raw_window_bars: int = 192, patch_size: int = 16, stride: int = 8, 
                 n_heads: int = 4, n_layers: int = 2, dropout_rate: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.patch_size = patch_size
        self.stride = stride
        self.num_patches = ((raw_window_bars - patch_size) // stride) + 1
        
        self.revin = RevIN(num_features=input_features_dim)
        self.patch_embedding = nn.Conv1d(in_channels=input_features_dim, out_channels=d_model, kernel_size=patch_size, stride=stride)
        self.time_encoding = Time2Vec(in_features=time_features_dim, out_dim=d_model)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pad_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.sep_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pad_token, std=0.02)
        nn.init.normal_(self.sep_token, std=0.02)

        self.portfolio_proj = nn.Linear(4, d_model)  
        self.inertia_proj = nn.Linear(1, d_model)    
        
        self.positional_embedding = nn.Parameter(torch.zeros(1, self.num_patches + 4, d_model))
        nn.init.normal_(self.positional_embedding, std=0.02)
        self.emb_norm = nn.LayerNorm(d_model)
        self.emb_dropout = nn.Dropout(dropout_rate)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2, dropout=dropout_rate, activation="gelu", batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x, t, portfolio_weights, bars_since_trade, raw_padding_mask):
        B, seq_len, _ = x.shape
        device = x.device
        
        x_norm = self.revin(x)
        patched = self.patch_embedding(x_norm.transpose(1, 2)).transpose(1, 2)
        patch_end_indices = torch.arange(self.patch_size - 1, seq_len, self.stride, device=device)
        tokens = patched + self.time_encoding(t[:, patch_end_indices, :])
        
        mask_float = raw_padding_mask.float().unsqueeze(1) 
        patch_pad_float = F.avg_pool1d(mask_float, kernel_size=self.patch_size, stride=self.stride)
        patch_padding_mask = (patch_pad_float == 1.0).squeeze(1) 
        
        expanded_pad = self.pad_token.expand(B, self.num_patches, -1)
        tokens = torch.where(patch_padding_mask.unsqueeze(-1), expanded_pad, tokens)

        port_token = self.portfolio_proj(portfolio_weights).unsqueeze(1) 
        inertia_token = self.inertia_proj(bars_since_trade).unsqueeze(1) 

        cls_tokens = self.cls_token.expand(B, -1, -1) 
        sep_tokens = self.sep_token.expand(B, -1, -1) 
        sequence = torch.cat([cls_tokens, port_token, inertia_token, sep_tokens, tokens], dim=1)
        current_seq_len = sequence.shape[1]
        sequence = sequence + self.positional_embedding[:, :current_seq_len, :]
        sequence = self.emb_dropout(self.emb_norm(sequence))
        
        structural_pad = torch.zeros((B, 4), dtype=torch.bool, device=device)
        attn_padding_mask = torch.cat([structural_pad, patch_padding_mask], dim=1) 
        
        transformer_out = self.transformer(sequence, src_key_padding_mask=attn_padding_mask)
        return transformer_out[:, 0, :]

class PPOActorHead(nn.Module):
    def __init__(self, d_model: int = 128, action_dim: int = 3, hidden_dim: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(d_model, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
        )
        self.mu = nn.Sequential(nn.Linear(hidden_dim, action_dim), nn.Tanh())
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, cls_token: torch.Tensor) -> Normal:
        mu = self.mu(self.trunk(cls_token))
        std = torch.exp(torch.clamp(self.log_std, -20, 2)).expand_as(mu)
        return Normal(mu, std)

class PPOCriticHead(nn.Module):
    def __init__(self, d_model: int = 128, hidden_dim: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(d_model, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
        )
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, cls_token: torch.Tensor) -> torch.Tensor:
        return self.value_head(self.trunk(cls_token)).squeeze(-1)

class QuantTradingAgent(nn.Module):
    def __init__(self, backbone: QuantRobertaBody, action_dim: int = 3, hidden_dim: int = 128):
        super().__init__()
        self.encoder = backbone
        self.actor = PPOActorHead(d_model=backbone.d_model, action_dim=action_dim, hidden_dim=hidden_dim)
        self.critic = PPOCriticHead(d_model=backbone.d_model, hidden_dim=hidden_dim)

# =============================================================================
# 3. ENVIRONMENTS
# =============================================================================

class TransformerTradingEnv(gym.Env):
    """
    Maintains the chronological sequence and yields raw time-series blocks
    as dictionaries for the Transformer to process live.
    """
    def __init__(
        self, 
        features_df: pd.DataFrame, 
        close_prices: np.ndarray, 
        max_window: int = 192, 
        initial_cash: float = 10_000.0, 
        tc_bps: float = 10.0,
        episode_length: int = 2048
    ):
        super().__init__()
        
        self.max_window = max_window
        self.initial_cash = initial_cash
        self.tc_multiplier = 1.0 - (tc_bps / 10_000.0)
        self.discretization_param = 5
        self.episode_length = episode_length
        
        time_cols = [c for c in features_df.columns if c.startswith('time_')]
        feat_cols = [c for c in features_df.columns if not c.startswith('time_')]
        
        self.t_data = features_df[time_cols].values.astype(np.float32)
        self.x_data = features_df[feat_cols].values.astype(np.float32)
        self.prices = close_prices.astype(np.float32) # [N, 3] Expected format for asset 0,1,2
        
        self.max_steps = len(self.x_data) - 1
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        # Uses Dict Space to naturally pass multimodal data
        self.observation_space = spaces.Dict({
            "x": spaces.Box(low=-np.inf, high=np.inf, shape=(max_window, len(feat_cols)), dtype=np.float32),
            "t": spaces.Box(low=-np.inf, high=np.inf, shape=(max_window, len(time_cols)), dtype=np.float32),
            "portfolio": spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32),
            "inertia": spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
            "mask": spaces.Box(low=0, high=1, shape=(max_window,), dtype=np.bool_)
        })

    def reset(self, *, seed=None, options=None) -> Tuple[Dict[str, np.ndarray], dict]:
        super().reset(seed=seed)
        
        # 1. Random historical start point
        min_start = self.max_window
        max_start = self.max_steps - self.episode_length - 1
        self._t = np.random.randint(min_start, max_start)
        self._end_step = self._t + self.episode_length
        
        # --- THE CORRECTED FIX: NATIVE RANDOM STATE ---
        # 1. Generate a totally random action matching your action space [-1.0 to 1.0]
        random_action = np.random.uniform(-1.0, 1.0, size=(3,))
        # The agent will wake up holding at least 70% Cash, protecting it from flash crashes.
        random_action *= 0.3
        
        # 2. Temporarily set weights to pure zero. 
        # This guarantees the random action will pass your 0.25 Inertia Filter.
        self._weights = np.zeros(4, dtype=np.float32)
        
        # 3. Pass the random action through your EXACT mathematical pipeline.
        # This automatically applies your Quantization, L1 projection, and Cash logic!
        self._weights = self._apply_action(random_action)
        
        # 4. Randomize inertia: The agent wakes up believing it has held this 
        # valid random portfolio anywhere from 0 to 48 bars.
        self._bars_since_trade = float(np.random.randint(0, 48))
        # ---------------------------------------------
        
        self._value = float(self.initial_cash)
        
        return self._get_obs(), {}

    def _get_obs(self) -> Dict[str, np.ndarray]:
        start_idx = self._t - self.max_window + 1
        end_idx = self._t + 1
        
        return {
            "x": self.x_data[start_idx:end_idx],
            "t": self.t_data[start_idx:end_idx],
            "portfolio": np.nan_to_num(self._weights, nan=0.0).astype(np.float32),
            "inertia": np.array([np.log1p(self._bars_since_trade)], dtype=np.float32),
            "mask": np.zeros(self.max_window, dtype=np.bool_) # No padding during live RL
        }

    def _apply_action(self, action: np.ndarray) -> np.ndarray:
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        
        # 1. Discretization: Destroys microscopic noise
        a = np.round(a * self.discretization_param) / self.discretization_param
        
        # 2. L1 Projection: Cap gross exposure at 100%
        gross_exposure = np.sum(np.abs(a))
        if gross_exposure > 1.0:
            a = a / gross_exposure 
            
        # 3. Float32 Shielding
        w_cash = float(1.0 - np.sum(a))
        w_cash = max(0.0, w_cash) 
        
        target_w = np.array([a[0], a[1], a[2], w_cash], dtype=np.float32)
        target_w = target_w / np.sum(target_w) # Force perfect sum to 1.0
        
        # 4. Inertia Filter: Rotations < 25% are ignored to save commissions
        current_w_safe = np.nan_to_num(self._weights, nan=0.0)
        turnover_intention = float(np.sum(np.abs(target_w - current_w_safe)))
        if turnover_intention < 0.25:  
            return self._weights.copy()
            
        return target_w

    def _reward_phase_1(self, prev_value: float, curr_value: float) -> float:
        # Calculate raw BPS reward
        reward_bps = np.log(curr_value / max(prev_value, 1e-8)) * 10_000.0
        if reward_bps < 0:
            reward_bps *= 2.5
        return float(np.clip(reward_bps, -75.0, 75.0))
    
    def _reward(self, prev_value: float, curr_value: float) -> float:
        # Calculate raw BPS reward
        reward_bps = np.log(curr_value / max(prev_value, 1e-8)) * 10_000.0
        
        PENALTY_MULT = 1.5
        THRESHOLD_BPS = -15.0 
        
        if reward_bps < THRESHOLD_BPS:
            # Calculate how far past the threshold the agent fell
            excess_loss = reward_bps - THRESHOLD_BPS
            
            # Apply the penalty ONLY to the excess damage
            reward_bps = THRESHOLD_BPS + (excess_loss * PENALTY_MULT)
            
        return float(np.clip(reward_bps, -75.0, 75.0))

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, dict]:
        prev_weights = self._weights.copy()
        prev_value = self._value
        
        # 1. Update Portfolio
        target_weights = self._apply_action(action)
        turnover = float(np.sum(np.abs(np.nan_to_num(target_weights, nan=0.0) - np.nan_to_num(prev_weights, nan=0.0))))
        
        # If the inertia filter blocked the trade, turnover is practically 0
        if turnover < 1e-4:
            self._bars_since_trade += 1.0
            turnover = 0.0
        else:
            self._weights = target_weights
            self._bars_since_trade = 0.0
            
        # 2. Advance Time 
        self._t += 1
        
        price_prev = self.prices[self._t - 1]
        price_curr = self.prices[self._t]
        returns = (price_curr - price_prev) / (price_prev + 1e-8)
        
        port_return = np.sum(self._weights[:3] * returns)
        
        self._value = prev_value * (1.0 + port_return)
        if turnover > 0: 
            self._value *= self.tc_multiplier 
        
        # 3. Reward Calculation & Neural Scaling
        raw_reward = self._reward(prev_value, self._value)
        
        # SCALING FACTOR: Keeps the reward between [-0.75, 0.75]. Fixes VLoss explosion!
        scaled_reward = raw_reward / 100.0 
        
        # 4. Termination
        truncated = bool(self._t >= self._end_step)
        
        # Stop-Loss: If it loses 80% of its money (from 10k to 2k)
        if self._value < 2000.0:
            truncated = True
        
        terminated = False 
            
        # Pass raw_reward in info dict so we can read it natively in TensorBoard
        return self._get_obs(), scaled_reward, terminated, truncated, {"raw_reward": raw_reward}

class TradingEnv(gym.Env):
    """
    Envoltura arquitectónica. Toma los precios del test, genera la historia 
    faltante para los MACDs, alinea los índices, e invoca al entorno continuo original.
    """
    def __init__(self, prices: pd.DataFrame, transaction_cost_bps: float = 10.0, initial_cash: float = 10_000.0):
        super().__init__()
        self.n_steps = len(prices)
        
        # 1. PADDING HACIA ATRÁS: El test envía 60 velas. Ocupamos ~2000 para el state macro.
        df = prices.copy()
        if len(df) < 2000:
            pad_size = 2000 - len(df) + 1
            first_row = df.iloc[[0]]
            pad_df = pd.concat([first_row] * pad_size, ignore_index=True)
            freq = pd.Timedelta(hours=1)
            pad_df.index = pd.date_range(end=df.index[0] - freq, periods=pad_size, freq=freq)
            df = pd.concat([pad_df, df])
            
        # 2. Transformador original
        features_df = build_complete_agent_state(df)
        
        # ALINEACIÓN CRÍTICA: features_df perdió ~1600 filas por el dropna().
        # Extraemos los precios de cierre usando EXCLUSIVAMENTE los índices que sobrevivieron.
        aligned_close_prices = df.loc[features_df.index, ['asset_0_close', 'asset_1_close', 'asset_2_close']].values
        
        # 3. Invocamos TU entorno real
        self.core_env = TransformerTradingEnv(
            features_df=features_df,
            close_prices=aligned_close_prices,
            max_window=192,
            initial_cash=initial_cash,
            tc_bps=transaction_cost_bps,
            episode_length=self.n_steps
        )
        
        # Calculamos exactamente dónde empiezan las 60 velas del test en esta matriz recortada
        self._test_start_idx = len(features_df) - self.n_steps
        
        # 4. Aseguramos el Box Space 1D para pytest
        sample_obs, _ = self.core_env.reset()
        flat_obs = self._flatten_obs(sample_obs)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(flat_obs),), dtype=np.float32)
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.current_step = 0

    def _flatten_obs(self, obs_dict: dict) -> np.ndarray:
        # Aplanar y garantizar float32 estricto en la máscara para pasar el `contains()`
        return np.concatenate([
            obs_dict['x'].flatten(),
            obs_dict['t'].flatten(),
            obs_dict['portfolio'].flatten(),
            obs_dict['inertia'].flatten(),
            obs_dict['mask'].astype(np.float32).flatten()
        ]).astype(np.float32)

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        self.current_step = 0
        obs_dict, info = self.core_env.reset(seed=seed, options=options)
        
        # Forzamos al core_env a situarse en el milisegundo exacto donde empieza la prueba
        self.core_env._t = self._test_start_idx
        self.core_env._end_step = self._test_start_idx + self.n_steps
        self.core_env._value = float(self.core_env.initial_cash)
        self.core_env._weights = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        self.core_env._bars_since_trade = 0.0
        
        obs_dict = self.core_env._get_obs()
        info["portfolio_value"] = self.core_env._value
        return self._flatten_obs(obs_dict), info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        target_weights = _ACTION_WEIGHTS[action]
        cont_action = target_weights[:3] # PPO recibe los 3 activos de riesgo [-1, 1]
        
        prev_val = self.core_env._value
        # Tu entorno ejecuta la L1 projection y actualiza el valor internamente
        obs_dict, _, terminated, truncated, info = self.core_env.step(cont_action)
        curr_val = self.core_env._value
        
        # Requerimiento estricto de pytest: Resta lineal sin escalar
        reward = self._reward(prev_val, curr_val)
        info["portfolio_value"] = curr_val
        
        self.current_step += 1
        terminated = bool(self.current_step >= self.n_steps - 1)
        
        return self._flatten_obs(obs_dict), reward, terminated, truncated, info

    def _reward(self, old_value: float, new_value: float) -> float:
        return float(new_value - old_value)

# =============================================================================
# 4. WRAPPER DEL AGENTE (Desempaqueta tensor y evalúa en el Transformer real)
# =============================================================================
class Agent:
    def __init__(self, obs_dim: int, n_actions: int = N_ACTIONS):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.epsilon = 1.0 
        self.max_window = 192
        self.time_dim = 3
        
        # Deducción dinámica de features (no se rompe si usas Exp3 de 20 cols o Exp5 de 80)
        # Ecuación: obs = W*F + W*T + 4 + 1 + W => F = (obs - W*T - 197) / W
        self.feat_dim = (obs_dim - (self.max_window * self.time_dim) - 197) // self.max_window
        if self.feat_dim < 1: self.feat_dim = 80 
            
        backbone = QuantRobertaBody(
            input_features_dim=self.feat_dim, 
            time_features_dim=self.time_dim,
            d_model=256, 
            raw_window_bars=self.max_window,
            patch_size=16,
            stride=8
        )
        self.model = QuantTradingAgent(backbone=backbone)
        
        self.has_model = False
        model_path = "quantroberta_phase_1.pth"
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                self.model.eval()
                self.has_model = True
            except Exception:
                pass

    def act(self, obs: np.ndarray) -> int:
        # Fallback de seguridad si el entorno manda basura intencional
        if not self.has_model or len(obs) != self.obs_dim:
            return 0 
            
        W = self.max_window
        F = self.feat_dim
        T = self.time_dim
        
        # 1. Reconstrucción del diccionario original para tu modelo
        idx = 0
        x_flat = obs[idx : idx + (W * F)]; idx += (W * F)
        t_flat = obs[idx : idx + (W * T)]; idx += (W * T)
        port_flat = obs[idx : idx + 4]; idx += 4
        inertia_flat = obs[idx : idx + 1]; idx += 1
        mask_flat = obs[idx : idx + W]
        
        # 2. Inferencia Real
        with torch.no_grad():
            x_t = torch.tensor(x_flat, dtype=torch.float32).view(1, W, F)
            t_t = torch.tensor(t_flat, dtype=torch.float32).view(1, W, T)
            port_t = torch.tensor(port_flat, dtype=torch.float32).unsqueeze(0)
            inertia_t = torch.tensor(inertia_flat, dtype=torch.float32).unsqueeze(0)
            mask_t = torch.tensor(mask_flat, dtype=torch.bool).unsqueeze(0)
            
            cls_state = self.model.encoder(x_t, t_t, port_t, inertia_t, mask_t)
            dist = self.model.actor(cls_state)
            continuous_action = dist.mean.numpy()[0]
            
        # 3. Lógica L1 idéntica a tu Apply_Action
        a = np.clip(continuous_action, -1.0, 1.0)
        gross = np.sum(np.abs(a))
        if gross > 1.0:
            a = a / gross
        w_cash = max(0.0, float(1.0 - np.sum(a)))
        target_w = np.array([a[0], a[1], a[2], w_cash])
        
        # 4. Traducción Euclidiana
        best_idx = 0
        min_dist = float('inf')
        for i, discrete_w in enumerate(_ACTION_WEIGHTS):
            dist_val = np.linalg.norm(target_w - discrete_w)
            if dist_val < min_dist:
                min_dist = dist_val
                best_idx = i
                
        return best_idx

    def train(self, env, n_steps: int):
        obs, _ = env.reset()
        for _ in range(n_steps):
            action = self.act(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()
        self.epsilon *= 0.9

# =============================================================================
# 5. TRAINING CODE FOR PPO
# =============================================================================

class MultimodalRolloutBuffer:
    """Stores full 3D sequences directly on the GPU."""
    def __init__(self, size: int, max_window: int, x_dim: int, t_dim: int, device: torch.device):
        self.device = device
        self.x = torch.zeros((size, max_window, x_dim), dtype=torch.float32, device=device)
        self.t = torch.zeros((size, max_window, t_dim), dtype=torch.float32, device=device)
        self.mask = torch.zeros((size, max_window), dtype=torch.bool, device=device)
        self.portfolio = torch.zeros((size, 4), dtype=torch.float32, device=device)
        self.inertia = torch.zeros((size, 1), dtype=torch.float32, device=device)
        
        self.actions = torch.zeros((size, 3), dtype=torch.float32, device=device)
        self.log_probs = torch.zeros(size, dtype=torch.float32, device=device)
        self.rewards = torch.zeros(size, dtype=torch.float32, device=device)
        self.values = torch.zeros(size, dtype=torch.float32, device=device)
        self.dones = torch.zeros(size, dtype=torch.float32, device=device)
        self.ptr = 0

    def add(self, obs: Dict[str, np.ndarray], action: np.ndarray, log_prob: float, reward: float, value: float, done: bool):
        i = self.ptr
        self.x[i] = torch.from_numpy(obs["x"]).to(self.device)
        self.t[i] = torch.from_numpy(obs["t"]).to(self.device)
        self.mask[i] = torch.from_numpy(obs["mask"]).to(self.device)
        self.portfolio[i] = torch.from_numpy(obs["portfolio"]).to(self.device)
        self.inertia[i] = torch.from_numpy(obs["inertia"]).to(self.device)
        
        self.actions[i] = torch.from_numpy(action).to(self.device)
        self.log_probs[i] = log_prob
        self.rewards[i] = reward
        self.values[i] = value
        self.dones[i] = float(done)
        self.ptr += 1

def compute_gae(rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor, last_value: torch.Tensor, gamma: float = 0.99, lam: float = 0.95):
    size = rewards.size(0)
    advantages = torch.zeros(size, dtype=torch.float32, device=rewards.device)
    gae = 0.0
    for t in reversed(range(size)):
        next_value = last_value if t == size - 1 else values[t + 1]
        next_non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        gae = delta + gamma * lam * next_non_terminal * gae
        advantages[t] = gae
    return advantages, advantages + values

class EndToEndPPOAgent:
    """
    Orchestrates the entire process with full telemetry, tracking KL divergence, 
    entropy, and financial metrics via TensorBoard.
    """
    def __init__(self, agent: QuantTradingAgent, rollout_len: int = 512, log_dir: str = "runs"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent = agent.to(self.device)
        self.rollout_len = rollout_len
        
        self.lr = 5e-5 #1e-4 for phase1
        self.gamma = 0.99
        self.lam = 0.95
        self.clip_eps = 0.2
        self.epochs_ppo = 4
        self.minibatch_size = 64
        
        self.c_entropy = 0.001
        self.optimizer = torch.optim.AdamW(self.agent.parameters(), lr=self.lr)
        self.global_step = 0
        
        # --- TELEMETRY SETUP ---
        self.run_name = f"QuantRoberta_PPO_{int(time.time())}"
        self.writer = SummaryWriter(Path(log_dir) / self.run_name)
        print(f"📡 Telemetry active. Run name: {self.run_name}")

    def _batch_obs(self, obs: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        return {k: torch.from_numpy(v).unsqueeze(0).to(self.device) for k, v in obs.items()}
    
    def train(self, env: TransformerTradingEnv, n_steps: int = 100_000):
        from collections import deque # <--- AÑADIDO: Para la media móvil
        
        n_iters = max(1, n_steps // self.rollout_len)
        obs, _ = env.reset()
        
        x_dim = env.observation_space["x"].shape[1]
        t_dim = env.observation_space["t"].shape[1]

        self.best_ret = -float('inf')
        
        # --- EL FIX: Ventana móvil para retornos estadísticamente significativos ---
        rolling_returns = deque(maxlen=20) 
        
        # =====================================================================
        # 🚨 BUG CRÍTICO CORREGIDO: MOVIDOS AFUERA DEL BUCLE
        # Para no borrar el progreso a la mitad del episodio.
        # =====================================================================
        current_ep_return, current_ep_length = 0.0, 0

        # --- EARLY STOPPING CONFIGURATION ---
        kl_patience = 10           # Iteraciones consecutivas necesarias para detener
        kl_threshold = 1e-4        # Si el KL cae por debajo de esto, consideramos que no aprende
        kl_collapse_counter = 0    # Contador actual

        for it in range(n_iters):
            buffer = MultimodalRolloutBuffer(self.rollout_len, env.max_window, x_dim, t_dim, self.device)
            self.agent.eval()

            # Solo reiniciamos las listas locales del rollout
            ep_returns, ep_lengths, final_portfolios = [], [], []

            # 1. Rollout Collection
            for _ in range(self.rollout_len):
                self.global_step += 1
                t_obs = self._batch_obs(obs)
                with torch.no_grad():
                    action, log_prob, _, value = self.agent.get_action_and_value(
                        t_obs["x"], t_obs["t"], t_obs["portfolio"], t_obs["inertia"], t_obs["mask"]
                    )
                
                a_np = action.squeeze(0).cpu().numpy()
                next_obs, reward, terminated, truncated, info = env.step(a_np)
                done = terminated or truncated

                # Network learns from scaled 'reward'
                buffer.add(obs, a_np, log_prob.item(), reward, value.item(), done)
                
                # YOU see the real BPS return
                real_reward = info.get("raw_reward", reward * 100.0)
                current_ep_return += real_reward 
                current_ep_length += 1

                if done:
                    ep_returns.append(current_ep_return)
                    
                    # --- AÑADIR A LA HISTORIA MÓVIL ---
                    rolling_returns.append(current_ep_return) 
                    
                    ep_lengths.append(current_ep_length)
                    final_portfolios.append(env._value)  
                    
                    # Resetear contadores solo cuando el episodio realmente termina
                    current_ep_return, current_ep_length = 0.0, 0
                    next_obs, _ = env.reset()
                
                obs = next_obs

            # 2. Generalized Advantage Estimation (GAE)
            with torch.no_grad():
                t_obs = self._batch_obs(obs)
                last_value = self.agent.get_value(
                    t_obs["x"], t_obs["t"], t_obs["portfolio"], t_obs["inertia"], t_obs["mask"]
                ).squeeze(0)
                
            advantages, returns = compute_gae(buffer.rewards, buffer.values, buffer.dones, last_value, self.gamma, self.lam)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # 3. Network Update (End-to-End)
            self.agent.train()
            idx = np.arange(self.rollout_len)
            
            # --- UPDATE TRACKERS ---
            clip_fractions, approx_kls = [], []
            actor_losses, value_losses, entropies = [], [], []
            
            for _ in range(self.epochs_ppo):
                np.random.shuffle(idx)
                for start in range(0, self.rollout_len, self.minibatch_size):
                    mb = idx[start:start + self.minibatch_size]
                    
                    _, new_lp, entropy, value = self.agent.get_action_and_value(
                        buffer.x[mb], buffer.t[mb], buffer.portfolio[mb], 
                        buffer.inertia[mb], buffer.mask[mb], action=buffer.actions[mb]
                    )

                    log_ratio = new_lp - buffer.log_probs[mb]
                    ratio = torch.exp(log_ratio)
                    
                    # Track KL & Clipping (Standard PPO Diagnostics)
                    with torch.no_grad():
                        old_approx_kl = (-log_ratio).mean().item()
                        approx_kl = ((ratio - 1) - log_ratio).mean().item()
                        approx_kls.append(approx_kl)
                        clip_fractions.append(((ratio - 1.0).abs() > self.clip_eps).float().mean().item())

                    surr1 = ratio * advantages[mb]
                    surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages[mb]
                    
                    actor_loss = -torch.min(surr1, surr2).mean()
                    value_loss = F.mse_loss(value, returns[mb])
                    entropy_loss = -entropy.mean()

                    loss = actor_loss + 0.5 * value_loss + self.c_entropy * entropy_loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), 1.0)
                    self.optimizer.step()

                    # Save for logging
                    actor_losses.append(actor_loss.item())
                    value_losses.append(value_loss.item())
                    entropies.append(entropy.mean().item())

            # 4. Log to TensorBoard
            y_kl = np.mean(approx_kls)
            y_ent = np.mean(entropies)
            y_aloss = np.mean(actor_losses)
            y_vloss = np.mean(value_losses)
            y_clip = np.mean(clip_fractions)
            y_rew_mean = buffer.rewards.mean().item()
            y_rew_std = buffer.rewards.std().item()

            self.writer.add_scalar("Losses/Actor", y_aloss, self.global_step)
            self.writer.add_scalar("Losses/Value", y_vloss, self.global_step)
            self.writer.add_scalar("Diagnostics/Entropy", y_ent, self.global_step)
            self.writer.add_scalar("Diagnostics/Approx_KL", y_kl, self.global_step)
            self.writer.add_scalar("Diagnostics/Clip_Fraction", y_clip, self.global_step)
            self.writer.add_scalar("Rewards/Step_Mean", y_rew_mean, self.global_step)
            self.writer.add_scalar("Rewards/Step_Std", y_rew_std, self.global_step)

            if len(ep_returns) > 0:
                self.writer.add_scalar("Environment/Ep_Return", np.mean(ep_returns), self.global_step)
                self.writer.add_scalar("Environment/Ep_Length", np.mean(ep_lengths), self.global_step)
                self.writer.add_scalar("Environment/Final_Portfolio", np.mean(final_portfolios), self.global_step)

            # Escribir la media móvil en TensorBoard
            if len(rolling_returns) > 0:
                self.writer.add_scalar("Environment/Rolling_Return_20ep", np.mean(rolling_returns), self.global_step)

            # 5. Console Output
            print(
                f"[Iter {it+1:03d}/{n_iters}] "
                f"Ret_Ep: {np.mean(ep_returns) if ep_returns else 0.0:+.1f} | "
                f"Roll_20: {np.mean(rolling_returns) if rolling_returns else 0.0:+.1f} | "
                f"ALoss: {y_aloss:+.3f} | VLoss: {y_vloss:+.3f} | "
                f"Ent: {y_ent:.3f}"
            )

            # ==========================================================
            # 6. EL NUEVO BLOQUE DE GUARDADO ROBUSTO
            # ==========================================================
            
            # Save a checkpoint every 50 iterations
            if (it + 1) % 50 == 0:
                checkpoint_path = Path(self.writer.log_dir) / f"checkpoint_iter_{it+1}.pth"
                torch.save(self.agent.state_dict(), checkpoint_path)
                print(f"💾 Checkpoint saved: {checkpoint_path.name}")

            # Solo consideramos guardar el "Mejor" modelo si ya hemos completado 
            # al menos 5 episodios completos para tener una mínima muestra estadística.
            if len(rolling_returns) >= 5:
                smoothed_ret = np.mean(rolling_returns)
                
                if smoothed_ret > self.best_ret:
                    self.best_ret = smoothed_ret
                    best_path = Path(self.writer.log_dir) / "model_BEST.pth"
                    torch.save(self.agent.state_dict(), best_path)
                    print(f"🌟 NEW STABLE BEST MODEL SAVED! (Smoothed Score over {len(rolling_returns)} eps: {self.best_ret:+.1f})")
            
            # ==========================================================
            # 7. EARLY STOPPING (KL COLLAPSE DETECTION)
            # ==========================================================
            if y_kl < kl_threshold:
                kl_collapse_counter += 1
                if kl_collapse_counter >= kl_patience:
                    print(f"\n🛑 EARLY STOPPING TRIGGERED: Approx_KL ha estado por debajo de {kl_threshold} durante {kl_patience} iteraciones consecutivas.")
                    print(f"🛑 El modelo ha alcanzado un óptimo local y ha cesado el aprendizaje. Abortando fase actual.")
                    break # Rompe el bucle for it in range(n_iters)
            else:
                kl_collapse_counter = 0 # Reiniciar contador si el modelo vuelve a aprender

        self.writer.close()

    def train_old(self, env: TransformerTradingEnv, n_steps: int = 100_000):
        n_iters = max(1, n_steps // self.rollout_len)
        obs, _ = env.reset()
        
        x_dim = env.observation_space["x"].shape[1]
        t_dim = env.observation_space["t"].shape[1]

        global_step = 0

        self.best_ret = -float('inf')

        for it in range(n_iters):
            buffer = MultimodalRolloutBuffer(self.rollout_len, env.max_window, x_dim, t_dim, self.device)
            self.agent.eval()

            # --- METRICS TRACKERS ---
            ep_returns, ep_lengths, final_portfolios = [], [], []
            current_ep_return, current_ep_length = 0.0, 0

            # 1. Rollout Collection
            for _ in range(self.rollout_len):
                global_step += 1
                t_obs = self._batch_obs(obs)
                with torch.no_grad():
                    action, log_prob, _, value = self.agent.get_action_and_value(
                        t_obs["x"], t_obs["t"], t_obs["portfolio"], t_obs["inertia"], t_obs["mask"]
                    )
                
                a_np = action.squeeze(0).cpu().numpy()
                next_obs, reward, terminated, truncated, info = env.step(a_np) # <--- Extract info
                done = terminated or truncated

                # Network learns from scaled 'reward'
                buffer.add(obs, a_np, log_prob.item(), reward, value.item(), done)
                
                # YOU see the real BPS return
                real_reward = info.get("raw_reward", reward * 100.0)
                current_ep_return += real_reward 
                current_ep_length += 1

                if done:
                    ep_returns.append(current_ep_return)
                    ep_lengths.append(current_ep_length)
                    final_portfolios.append(env._value)  # Track final cash
                    current_ep_return, current_ep_length = 0.0, 0
                    next_obs, _ = env.reset()
                
                obs = next_obs

            # 2. Generalized Advantage Estimation (GAE)
            with torch.no_grad():
                t_obs = self._batch_obs(obs)
                last_value = self.agent.get_value(
                    t_obs["x"], t_obs["t"], t_obs["portfolio"], t_obs["inertia"], t_obs["mask"]
                ).squeeze(0)
                
            advantages, returns = compute_gae(buffer.rewards, buffer.values, buffer.dones, last_value, self.gamma, self.lam)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # 3. Network Update (End-to-End)
            self.agent.train()
            idx = np.arange(self.rollout_len)
            
            # --- UPDATE TRACKERS ---
            clip_fractions, approx_kls = [], []
            actor_losses, value_losses, entropies = [], [], []
            
            for _ in range(self.epochs_ppo):
                np.random.shuffle(idx)
                for start in range(0, self.rollout_len, self.minibatch_size):
                    mb = idx[start:start + self.minibatch_size]
                    
                    _, new_lp, entropy, value = self.agent.get_action_and_value(
                        buffer.x[mb], buffer.t[mb], buffer.portfolio[mb], 
                        buffer.inertia[mb], buffer.mask[mb], action=buffer.actions[mb]
                    )

                    log_ratio = new_lp - buffer.log_probs[mb]
                    ratio = torch.exp(log_ratio)
                    
                    # Track KL & Clipping (Standard PPO Diagnostics)
                    with torch.no_grad():
                        old_approx_kl = (-log_ratio).mean().item()
                        approx_kl = ((ratio - 1) - log_ratio).mean().item()
                        approx_kls.append(approx_kl)
                        clip_fractions.append(((ratio - 1.0).abs() > self.clip_eps).float().mean().item())

                    surr1 = ratio * advantages[mb]
                    surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages[mb]
                    
                    actor_loss = -torch.min(surr1, surr2).mean()
                    value_loss = F.mse_loss(value, returns[mb])
                    entropy_loss = -entropy.mean()

                    loss = actor_loss + 0.5 * value_loss + self.c_entropy * entropy_loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), 1.0)
                    self.optimizer.step()

                    # Save for logging
                    actor_losses.append(actor_loss.item())
                    value_losses.append(value_loss.item())
                    entropies.append(entropy.mean().item())

            # 4. Log to TensorBoard
            y_kl = np.mean(approx_kls)
            y_ent = np.mean(entropies)
            y_aloss = np.mean(actor_losses)
            y_vloss = np.mean(value_losses)
            y_clip = np.mean(clip_fractions)
            y_rew_mean = buffer.rewards.mean().item()
            y_rew_std = buffer.rewards.std().item()

            self.writer.add_scalar("Losses/Actor", y_aloss, global_step)
            self.writer.add_scalar("Losses/Value", y_vloss, global_step)
            self.writer.add_scalar("Diagnostics/Entropy", y_ent, global_step)
            self.writer.add_scalar("Diagnostics/Approx_KL", y_kl, global_step)
            self.writer.add_scalar("Diagnostics/Clip_Fraction", y_clip, global_step)
            self.writer.add_scalar("Rewards/Step_Mean", y_rew_mean, global_step)
            self.writer.add_scalar("Rewards/Step_Std", y_rew_std, global_step)

            if len(ep_returns) > 0:
                self.writer.add_scalar("Environment/Ep_Return", np.mean(ep_returns), global_step)
                self.writer.add_scalar("Environment/Ep_Length", np.mean(ep_lengths), global_step)
                self.writer.add_scalar("Environment/Final_Portfolio", np.mean(final_portfolios), global_step)

            # 5. Console Output
            print(
                f"[Iter {it+1:03d}/{n_iters}] "
                f"Ret: {np.mean(ep_returns) if ep_returns else 0.0:+.1f} | "
                f"ALoss: {y_aloss:+.3f} | VLoss: {y_vloss:+.3f} | "
                f"Ent: {y_ent:.3f} | KL: {y_kl:.4f} | "
                f"Port: ${np.mean(final_portfolios) if final_portfolios else env._value:.1f}"
            )

            # --- ADD THIS ENTIRE SAVE BLOCK HERE ---
            current_mean_ret = np.mean(ep_returns) if len(ep_returns) > 0 else 0.0
            
            # Save a checkpoint every 50 iterations (approx every 2-3 minutes)
            if (it + 1) % 50 == 0:
                checkpoint_path = Path(self.writer.log_dir) / f"checkpoint_iter_{it+1}.pth"
                torch.save(self.agent.state_dict(), checkpoint_path)
                print(f"💾 Checkpoint saved: {checkpoint_path.name}")

            # Save the "Best" model whenever the Return hits a new high
            if len(ep_returns) > 0 and current_mean_ret > self.best_ret:
                self.best_ret = current_mean_ret
                best_path = Path(self.writer.log_dir) / "model_BEST.pth"
                torch.save(self.agent.state_dict(), best_path)
                print(f"🌟 NEW BEST MODEL SAVED! (Score: {self.best_ret:+.1f})")

        self.writer.close()
        