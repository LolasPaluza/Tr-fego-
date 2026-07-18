"""Modelos de configuração (pydantic) e carregamento de YAML.

Toda configuração do projeto passa por aqui: validação de tipos e de
restrições de segurança (ex.: verde mínimo) acontece na construção dos
modelos, não espalhada pelo código.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

ObsMode = Literal["obs_minimal", "obs_rica"]
RewardMode = Literal["r_espera", "r_fila", "r_pressao"]

# Ordem canônica das aproximações em TODAS as estruturas do projeto.
# Avenida (leste-oeste) primeiro, via local (norte-sul) depois.
APPROACHES: tuple[str, ...] = ("E", "W", "N", "S")

# Estágios verdes do programa semafórico (ver envs/phases.py):
# 0 = avenida seguir/direita, 1 = avenida esquerda,
# 2 = local seguir/direita,   3 = local esquerda.
N_STAGES = 4
STAGE_NAMES: tuple[str, ...] = (
    "av_frente_dir",
    "av_esquerda",
    "local_frente_dir",
    "local_esquerda",
)


class NetworkConfig(BaseModel):
    """Especificação paramétrica da rede (vira gerador de grid na Fase 2)."""

    approach_length_m: float = 300.0
    avenue_lanes: int = 3
    avenue_speed_kmh: float = 60.0
    local_lanes: int = 1
    local_speed_kmh: float = 40.0

    @property
    def avenue_speed_ms(self) -> float:
        return self.avenue_speed_kmh / 3.6

    @property
    def local_speed_ms(self) -> float:
        return self.local_speed_kmh / 3.6


class SignalConfig(BaseModel):
    """Tempos de segurança do semáforo. O wrapper do ambiente IMPÕE esses
    valores — o agente nunca consegue violá-los."""

    min_green_s: float = Field(10.0, ge=5.0)
    max_green_s: float = Field(120.0, gt=0)
    yellow_s: float = Field(3.0, ge=3.0)
    all_red_s: float = Field(2.0, ge=2.0)

    @model_validator(mode="after")
    def _check_max_green(self) -> SignalConfig:
        if self.max_green_s <= self.min_green_s:
            raise ValueError("max_green_s deve ser maior que min_green_s")
        return self


class TurnShares(BaseModel):
    straight: float = 0.70
    right: float = 0.15
    left: float = 0.15

    @model_validator(mode="after")
    def _check_sum(self) -> TurnShares:
        total = self.straight + self.right + self.left
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"frações de movimento devem somar 1.0 (soma={total})")
        return self


class ScenarioConfig(BaseModel):
    """Cenário de demanda. Vazões em veíc/h por sentido."""

    name: str
    avenue_flow_vph: float = Field(gt=0)
    local_flow_vph: float = Field(gt=0)
    truck_share: float = Field(0.0, ge=0.0, le=0.5)
    turn_shares: TurnShares = TurnShares()


class StarvationConfig(BaseModel):
    """Componente anti-starvation da recompensa (configurável)."""

    enabled: bool = True
    threshold_s: float = Field(120.0, gt=0)
    penalty: float = Field(1.0, ge=0)


class EnvConfig(BaseModel):
    network: NetworkConfig = NetworkConfig()
    signal: SignalConfig = SignalConfig()
    episode_length_s: float = Field(3600.0, gt=0)
    delta_time_s: float = Field(5.0, ge=1.0)
    obs_mode: ObsMode = "obs_minimal"
    reward_mode: RewardMode = "r_espera"
    starvation: StarvationConfig = StarvationConfig()
    use_libsumo: bool = True

    @model_validator(mode="after")
    def _check_delta(self) -> EnvConfig:
        transition = self.signal.yellow_s + self.signal.all_red_s
        if self.delta_time_s > self.signal.min_green_s:
            raise ValueError("delta_time_s não pode exceder min_green_s")
        if transition <= 0:
            raise ValueError("transição amarelo+all-red deve ser positiva")
        return self


class DQNConfig(BaseModel):
    learning_rate: float = 5e-4
    buffer_size: int = Field(100_000, ge=100_000)
    learning_starts: int = 1_000
    batch_size: int = 64
    gamma: float = 0.99
    train_freq: int = 1
    target_update_interval: int = 1_000
    exploration_fraction: float = 0.3
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.02
    net_arch: list[int] = [256, 256]
    double_dqn: bool = True


class TrainConfig(BaseModel):
    scenario: str = "pico_assimetrico"
    seeds: list[int] = [42, 123, 7, 2024, 777]
    total_timesteps: int = 300_000
    eval_freq: int = 10_000
    n_eval_episodes: int = 3
    checkpoint_freq: int = 25_000
    early_stopping: bool = False
    early_stopping_patience: int = 6
    dqn: DQNConfig = DQNConfig()

    @field_validator("seeds")
    @classmethod
    def _non_empty(cls, v: list[int]) -> list[int]:
        if not v:
            raise ValueError("é preciso ao menos uma seed")
        return v


class EvalConfig(BaseModel):
    n_episodes: int = 20
    # Seeds de tráfego da avaliação: base + índice do episódio.
    # base=10_000 garante disjunção das seeds de treino (42..2024).
    traffic_seed_base: int = 10_000
    scenarios: list[str] = [
        "pico_assimetrico",
        "fora_pico",
        "balanceado",
        "pico_invertido",
    ]


class ProjectConfig(BaseModel):
    env: EnvConfig = EnvConfig()
    scenarios: dict[str, ScenarioConfig]
    train: TrainConfig = TrainConfig()
    eval: EvalConfig = EvalConfig()

    def scenario(self, name: str) -> ScenarioConfig:
        if name not in self.scenarios:
            raise KeyError(f"cenário desconhecido: {name!r} (disponíveis: {list(self.scenarios)})")
        return self.scenarios[name]


def _deep_update(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(
    config_path: Path | str,
    overrides: dict | None = None,
) -> ProjectConfig:
    """Carrega YAML e valida. `overrides` (dict aninhado) tem precedência —
    é assim que o modo smoke reduz parâmetros sem duplicar configuração."""
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if overrides:
        raw = _deep_update(raw, overrides)
    scenarios = {
        name: ScenarioConfig(name=name, **body) for name, body in raw.get("scenarios", {}).items()
    }
    return ProjectConfig(
        env=EnvConfig(**raw.get("env", {})),
        scenarios=scenarios,
        train=TrainConfig(**raw.get("train", {})),
        eval=EvalConfig(**raw.get("eval", {})),
    )
