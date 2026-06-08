# =============================================================================
# YOUR FILE — agent.py
# Implements TradingEnv (hybrid action) and Agent (PPO with Gating mechanism).
# =============================================================================

import math
from typing import Tuple, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet, Bernoulli
import gymnasium as gym
from gymnasium import spaces
from sklearn.preprocessing import StandardScaler
from pydantic import BaseModel, Field

from src.env import BaseTradingEnv
from src.base import BaseAgent


# ── Configuración Pydantic ───────────────────────────────────────────────────

class AgentConfig(BaseModel):
    """Configuración inmutable del agente (Principio KISS y Tipado Fuerte)."""
    obs_dim: int = Field(default=68, description="Dimensión de la observación")
    action_dim: int = Field(default=8, description="1 (Gate) + 7 (Dirichlet)")
    hidden_dim: int = Field(default=128, description="Dimensión de las capas ocultas")
    lr: float = Field(default=3e-4, description="Tasa de aprendizaje")
    gamma: float = Field(default=0.99, description="Factor de descuento")
    lam: float = Field(default=0.95, description="Parámetro lambda para GAE")
    clip_eps: float = Field(default=0.2, description="Clipping de PPO")
    epochs_ppo: int = Field(default=10, description="Épocas de optimización PPO")
    minibatch_size: int = Field(default=64, description="Tamaño de minibatch PPO")
    rollout_len: int = Field(default=2048, description="Pasos por iteración")
    c_value: float = Field(default=0.5, description="Coeficiente de pérdida de valor")
    c_entropy: float = Field(default=0.01, description="Coeficiente de entropía")
    grad_clip: float = Field(default=0.5, description="Recorte de gradientes")


# ── Encoder MLM (frozen, copia de la arquitectura entrenada) ─────────────────

class MiniTradingRoberta(nn.Module):
    """
    Copia exacta de la arquitectura usada en el pretraining MLM.
    Solo se usa para cargar los pesos y hacer forward en modo eval.
    """
    def __init__(self, input_dim: int = 12, d_model: int = 64, n_heads: int = 4, n_layers: int = 2, max_seq_len: int = 170):
        super().__init__()
        self.d_model = d_model
        self.feature_projection = nn.Linear(input_dim, d_model)
        self.cls_token  = nn.Parameter(torch.randn(1, 1, d_model))
        self.sep_token  = nn.Parameter(torch.randn(1, 1, d_model))
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pad_token  = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=128,
            dropout=0.1, activation="gelu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.mlm_head = nn.Linear(d_model, input_dim)

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None, mlm_mask: Optional[torch.Tensor] = None, is_mlm: bool = False) -> torch.Tensor:
        B, seq_len, _ = x.shape
        x_emb = self.feature_projection(x)
        if padding_mask is not None:
            x_emb = torch.where(padding_mask.unsqueeze(-1), self.pad_token, x_emb)
        if is_mlm and mlm_mask is not None:
            x_emb = torch.where(mlm_mask.unsqueeze(-1), self.mask_token, x_emb)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        sep_tokens = self.sep_token.expand(B, -1, -1)
        sequence = torch.cat((cls_tokens, x_emb, sep_tokens), dim=1)
        sequence = sequence + self.pos_embedding[:, :sequence.size(1), :]
        if padding_mask is not None:
            cls_pad = torch.zeros((B, 1), dtype=torch.bool, device=x.device)
            sep_pad = torch.zeros((B, 1), dtype=torch.bool, device=x.device)
            transformer_pad_mask = torch.cat((cls_pad, padding_mask, sep_pad), dim=1)
        else:
            transformer_pad_mask = None
        out = self.transformer(sequence, src_key_padding_mask=transformer_pad_mask)
        if is_mlm:
            return self.mlm_head(out[:, 1:-1, :])
        return out[:, 0, :]   # CLS embedding (B, d_model)


# ── Feature builder ──────────────────────────────────────────────────────────

def build_raw_stationary(data: pd.DataFrame, scaler: StandardScaler, fit: bool = False) -> pd.DataFrame:
    """
    Aplica la misma transformación usada en el pretraining MLM.
    El scaler debe venir del fit hecho con los datos de training.
    """
    frames = []
    assets = ["asset_0", "asset_1", "asset_2"]
    for asset in assets:
        close = data[f"{asset}_close"]
        high  = data[f"{asset}_high"]
        low   = data[f"{asset}_low"]
        vol   = data[f"{asset}_volume"]
        tbr   = data[f"{asset}_taker_buy_ratio"]
        log_ret   = np.log(close / close.shift(1))
        amplitude = (high - low) / close.shift(1)
        frames.append(pd.DataFrame({
            f"{asset}_log_ret":    log_ret,
            f"{asset}_amplitude":  amplitude,
            f"{asset}_vol":        vol,
            f"{asset}_tbr":        tbr,
        }))
    X = pd.concat(frames, axis=1).dropna()
    if fit:
        scaler.fit(X.values)
    X_scaled = pd.DataFrame(scaler.transform(X.values), index=X.index, columns=X.columns)
    return X_scaled


# ── Funciones Puras (Filosofía Funcional) ────────────────────────────────────

def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_value: torch.Tensor,
    gamma: float,
    lam: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Función pura para calcular GAE (Generalized Advantage Estimation).
    Cumple con el principio de programación funcional (sin mutaciones externas).
    """
    size = rewards.size(0)
    advantages = torch.zeros_like(rewards)
    gae = 0.0
    for t in reversed(range(size)):
        next_value = last_value if t == size - 1 else values[t + 1]
        next_non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        gae = delta + gamma * lam * next_non_terminal * gae
        advantages[t] = gae
    returns = advantages + values
    return advantages, returns


# ── Environment ──────────────────────────────────────────────────────────────

class TradingEnv(BaseTradingEnv):
    """
    Entorno de trading con acción híbrida de 8 dimensiones.
    Dimensión 0: Gate (decisión de operar vs. mantener).
    Dimensiones 1-7: Símplex Dirichlet para el nuevo portafolio.
    """

    OBS_DIM    = 68
    ACTION_DIM = 8
    LOOKBACK   = 168
    EMB_DIM    = 64

    def __init__(
        self,
        prices: pd.DataFrame,
        encoder: MiniTradingRoberta,
        scaler: StandardScaler,
        transaction_cost_bps: float = 10.0,
        initial_cash: float = 10_000.0,
        embed_batch_size: int = 256,
        episode_length: Optional[int] = None,
        random_start: bool = False,
        random_initial_portfolio: bool = False,
    ):
        super().__init__(prices, transaction_cost_bps, initial_cash)

        self._lookback = self.LOOKBACK
        self.episode_length = episode_length
        self.random_start = random_start
        self.random_initial_portfolio = random_initial_portfolio
        self.device = next(encoder.parameters()).device

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.ACTION_DIM,), dtype=np.float32
        )

        self._latents, self._valid_t_min = self._precompute_embeddings(
            prices, encoder, scaler, embed_batch_size
        )
        self._lookback = max(self._lookback, self._valid_t_min)

        self._rng = np.random.default_rng()
        self._episode_end_t = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        gym.Env.reset(self, seed=seed)

        T = len(self.prices)
        if self.random_start:
            ep_len = self.episode_length or (T - self._lookback - 1)
            max_start = T - ep_len - 1
            self._t = int(self._rng.integers(self._lookback, max_start + 1))
            self._episode_end_t = self._t + ep_len
        else:
            self._t = self._lookback
            self._episode_end_t = T - 1

        if self.random_initial_portfolio:
            alpha = np.ones(7, dtype=np.float32)
            p = self._rng.dirichlet(alpha).astype(np.float32)
            w = np.array([
                p[0] - p[3], p[1] - p[4], p[2] - p[5],
            ], dtype=np.float32)
            cash = np.float32(1.0) - w.sum()
            self._weights = np.append(w, cash).astype(np.float32)
        else:
            self._weights = np.array([0., 0., 0., 1.], dtype=np.float32)

        self._value = float(self.initial_cash)
        return self._obs(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = super().step(action)
        if self._episode_end_t is not None and self._t >= self._episode_end_t:
            truncated = True
        return obs, reward, terminated, truncated, info

    def _precompute_embeddings(self, prices_df: pd.DataFrame, encoder: nn.Module, scaler: StandardScaler, batch_size: int) -> Tuple[np.ndarray, int]:
        features = build_raw_stationary(prices_df, scaler=scaler, fit=False)
        feature_vals = features.values.astype(np.float32)

        price_index = prices_df.index
        feat_index  = features.index
        feat_pos = pd.Series(np.arange(len(feat_index)), index=feat_index)

        T_prices = len(price_index)
        latents = np.zeros((T_prices, self.EMB_DIM), dtype=np.float32)
        valid_mask = np.zeros(T_prices, dtype=bool)

        encoder.eval()

        windows, t_targets = [], []
        for t in range(T_prices):
            ts = price_index[t]
            if ts not in feat_pos.index:
                continue
            pos = int(feat_pos.loc[ts])
            if pos < self._lookback:
                continue
            window = feature_vals[pos - self._lookback : pos]
            windows.append(window)
            t_targets.append(t)

        if not windows:
            raise RuntimeError("No hay ventanas válidas. ¿lookback demasiado grande?")

        with torch.no_grad():
            for i in range(0, len(windows), batch_size):
                batch = np.stack(windows[i:i + batch_size], axis=0)
                tb = torch.from_numpy(batch).to(self.device)
                emb = encoder(tb, is_mlm=False).cpu().numpy()
                for j, t_idx in enumerate(t_targets[i:i + batch_size]):
                    latents[t_idx] = emb[j]
                    valid_mask[t_idx] = True

        valid_t_min = int(np.argmax(valid_mask))
        return latents, valid_t_min

    def _obs(self) -> np.ndarray:
        market_emb = self._latents[self._t]
        portfolio  = self._weights.astype(np.float32)
        return np.concatenate([market_emb, portfolio], axis=0).astype(np.float32)

    def _weights_from_action(self, action: np.ndarray) -> np.ndarray:
        """
        action[0]: Gate param (0 = no operar, mantener pesos; 1 = operar)
        action[1:]: Target portfolio desde Dirichlet
        """
        action = np.asarray(action, dtype=np.float32)
        gate = action[0]
        
        # Implementación del mecanismo "Gate" para evitar operaciones inútiles
        if gate < 0.5:
            return self._weights.copy()
            
        p = action[1:]
        w = np.array([
            p[0] - p[3],
            p[1] - p[4],
            p[2] - p[5],
        ], dtype=np.float32)
        cash = np.float32(1.0) - w.sum()
        return np.append(w, cash).astype(np.float32)

    def _reward(self, prev_value: float, curr_value: float) -> float:
        """
        Retorno logarítmico escalado con Ratio de Sortino implícito y recorte de colas pesadas.
        Las comisiones (tc) ya están descontadas orgánicamente en curr_value por el entorno.
        """
        # 1. Retorno logarítmico en basis points (bps)
        reward_bps = np.log(curr_value / max(prev_value, 1e-8)) * 10_000.0
        
        # 2. Aversión a la pérdida
        penalty_factor = 2.5
        if reward_bps < 0:
            reward_bps *= penalty_factor
            
        # 3. Recorte de anomalías (Clipping de colas pesadas)
        return float(np.clip(reward_bps, -75.0, 75.0))


# ── Redes ────────────────────────────────────────────────────────────────────

class ActorCritic(nn.Module):
    """
    Red Neuronal con Action Space Híbrido:
    - Gate (Bernoulli): Decide si operar o mantener.
    - Portfolio (Dirichlet): Decide la nueva distribución óptima.
    """
    def __init__(self, cfg: AgentConfig):
        super().__init__()
        self.actor_trunk = nn.Sequential(
            nn.Linear(cfg.obs_dim, cfg.hidden_dim), nn.Tanh(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.Tanh(),
        )
        self.actor_gate = nn.Linear(cfg.hidden_dim, 1)
        self.actor_dirichlet = nn.Linear(cfg.hidden_dim, 7)

        self.critic_trunk = nn.Sequential(
            nn.Linear(cfg.obs_dim, cfg.hidden_dim), nn.Tanh(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.Tanh(),
        )
        self.critic_head = nn.Linear(cfg.hidden_dim, 1)

    def get_action_dist(self, obs: torch.Tensor) -> Tuple[Bernoulli, Dirichlet]:
        h = self.actor_trunk(obs)
        gate_logit = self.actor_gate(h).squeeze(-1)
        alpha = F.softplus(self.actor_dirichlet(h)) + 1e-3
        return Bernoulli(logits=gate_logit), Dirichlet(alpha)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic_head(self.critic_trunk(obs)).squeeze(-1)

    def sample_action(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gate_dist, dir_dist = self.get_action_dist(obs)
        gate = gate_dist.sample()
        p = dir_dist.sample()
        
        log_prob = gate_dist.log_prob(gate) + dir_dist.log_prob(p)
        value = self.get_value(obs)
        action = torch.cat([gate.unsqueeze(-1), p], dim=-1)
        return action, log_prob, value

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gate_dist, dir_dist = self.get_action_dist(obs)
        gate_action = actions[:, 0]
        p_action = actions[:, 1:]
        
        log_prob = gate_dist.log_prob(gate_action) + dir_dist.log_prob(p_action)
        entropy = gate_dist.entropy() + dir_dist.entropy()
        value = self.get_value(obs)
        return log_prob, entropy, value


# ── Rollout buffer ───────────────────────────────────────────────────────────

class RolloutBuffer:
    def __init__(self, size: int, obs_dim: int, action_dim: int, device: torch.device):
        self.obs       = torch.zeros((size, obs_dim),    device=device)
        self.actions   = torch.zeros((size, action_dim), device=device)
        self.log_probs = torch.zeros(size,               device=device)
        self.rewards   = torch.zeros(size,               device=device)
        self.values    = torch.zeros(size,               device=device)
        self.dones     = torch.zeros(size,               device=device)
        self.ptr = 0

    def add(self, obs: torch.Tensor, action: torch.Tensor, log_prob: torch.Tensor, reward: float, value: torch.Tensor, done: bool) -> None:
        i = self.ptr
        self.obs[i]       = obs
        self.actions[i]   = action
        self.log_probs[i] = log_prob
        self.rewards[i]   = reward
        self.values[i]    = value
        self.dones[i]     = float(done)
        self.ptr += 1


# ── Agente ───────────────────────────────────────────────────────────────────

class Agent(BaseAgent):
    """Agente principal que coordina el entrenamiento PPO con acción híbrida."""
    
    def __init__(self, obs_dim: int = 68, n_actions: int = 8):
        super().__init__(obs_dim, n_actions)
        self.cfg = AgentConfig(obs_dim=obs_dim, action_dim=n_actions)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps"  if torch.backends.mps.is_available() else
            "cpu"
        )
        self.ac = ActorCritic(self.cfg).to(self.device)
        self.optimizer = torch.optim.Adam(self.ac.parameters(), lr=self.cfg.lr)

    def train(self, env: TradingEnv, n_steps: int = 500_000, val_env: Optional[TradingEnv] = None, eval_every: int = 20) -> None:
        n_iters = n_steps // self.cfg.rollout_len
        obs, _ = env.reset()
        obs = torch.from_numpy(obs).to(self.device)

        best_val_return = -float("inf")
        episode_returns = []
        running_return = 0.0

        for it in range(n_iters):
            buffer = RolloutBuffer(self.cfg.rollout_len, env.OBS_DIM, env.ACTION_DIM, self.device)
            self.ac.eval()

            # ---------- ROLLOUT ----------
            for _ in range(self.cfg.rollout_len):
                with torch.no_grad():
                    action, log_prob, value = self.ac.sample_action(obs.unsqueeze(0))
                    action  = action.squeeze(0)
                    log_prob = log_prob.squeeze(0)
                    value    = value.squeeze(0)

                next_obs, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
                done = terminated or truncated

                buffer.add(obs, action, log_prob, float(reward), value, done)
                running_return += reward

                if done:
                    episode_returns.append(running_return)
                    running_return = 0.0
                    next_obs, _ = env.reset()

                obs = torch.from_numpy(next_obs).to(self.device)

            # ---------- GAE ----------
            with torch.no_grad():
                last_value = self.ac.get_value(obs.unsqueeze(0)).squeeze(0)
                
            advantages, returns = compute_gae(
                buffer.rewards, buffer.values, buffer.dones, 
                last_value, self.cfg.gamma, self.cfg.lam
            )
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # ---------- OPTIMIZACIÓN ----------
            self.ac.train()
            idx = np.arange(self.cfg.rollout_len)
            for _ in range(self.cfg.epochs_ppo):
                np.random.shuffle(idx)
                for start in range(0, self.cfg.rollout_len, self.cfg.minibatch_size):
                    mb = idx[start:start + self.cfg.minibatch_size]
                    mb_obs   = buffer.obs[mb]
                    mb_acts  = buffer.actions[mb]
                    mb_old_lp = buffer.log_probs[mb]
                    mb_adv   = advantages[mb]
                    mb_ret   = returns[mb]

                    new_lp, entropy, value = self.ac.evaluate_actions(mb_obs, mb_acts)

                    ratio = torch.exp(new_lp - mb_old_lp)
                    surr1 = ratio * mb_adv
                    surr2 = torch.clamp(ratio, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps) * mb_adv
                    actor_loss   = -torch.min(surr1, surr2).mean()
                    value_loss   = F.mse_loss(value, mb_ret)
                    entropy_loss = -entropy.mean()

                    loss = actor_loss + self.cfg.c_value * value_loss + self.cfg.c_entropy * entropy_loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.ac.parameters(), self.cfg.grad_clip)
                    self.optimizer.step()

            # ---------- LOGGING ----------
            recent = episode_returns[-20:] if episode_returns else [0.0]
            print(
                f"[Iter {it+1}/{n_iters}] "
                f"steps={(it+1)*self.cfg.rollout_len} "
                f"mean_ep_return={np.mean(recent):+.4f} "
                f"actor_loss={actor_loss.item():+.4f} "
                f"value_loss={value_loss.item():.4f} "
                f"entropy={(-entropy_loss.item()):.3f}"
            )

            # ---------- EVALUACIÓN ----------
            if val_env is not None and (it + 1) % eval_every == 0:
                val_return = self.evaluate(val_env)
                print(f"  └─ val_return = {val_return:+.4f}")
                if val_return > best_val_return:
                    best_val_return = val_return
                    torch.save(self.ac.state_dict(), "best_agent.pth")
                    print(f"     ¡Nuevo mejor agente guardado! ({val_return:+.4f})")

        if val_env is not None and Path("best_agent.pth").exists():
            self.ac.load_state_dict(torch.load("best_agent.pth", weights_only=True))
            print(f"\nPesos óptimos restaurados (val_return = {best_val_return:+.4f}).")

    @torch.no_grad()
    def evaluate(self, env: TradingEnv) -> float:
        """Evaluación determinista (Media de la Dirichlet, Umbral del Bernoulli)."""
        self.ac.eval()
        obs, _ = env.reset()
        obs = torch.from_numpy(obs).to(self.device)
        total_return = 0.0
        while True:
            gate_dist, dirichlet_dist = self.ac.get_action_dist(obs.unsqueeze(0))
            gate = (gate_dist.probs > 0.5).float()
            p = dirichlet_dist.mean
            action = torch.cat([gate.unsqueeze(-1), p], dim=-1).squeeze(0)
            
            next_obs, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
            total_return += float(reward)
            if terminated or truncated:
                break
            obs = torch.from_numpy(next_obs).to(self.device)
        return total_return

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        """
        Modo producción: Inferencia puramente determinista.
        """
        self.ac.eval()
        obs_t = torch.from_numpy(obs).to(self.device).unsqueeze(0)
        gate_dist, dirichlet_dist = self.ac.get_action_dist(obs_t)
        
        gate = (gate_dist.probs > 0.5).float()
        p = dirichlet_dist.mean
        action = torch.cat([gate.unsqueeze(-1), p], dim=-1).squeeze(0)
        
        return action.cpu().numpy()