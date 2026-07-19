"""Configuração da Fase 2: grid NxM com hierarquia viária declarada pelo usuário.

A interface central é o YAML de grid: você lista as ruas horizontais (`rows`,
leste-oeste) e verticais (`cols`, norte-sul), e para cada uma declara nome,
classe (`avenida` = 3 faixas/60 km/h, `local` = 1 faixa/40 km/h) e o fluxo de
entrada (veíc/h POR SENTIDO). As ruas mais movimentadas são simplesmente as
de maior fluxo — o agente não recebe essa informação de graça: precisa
inferi-la das observações locais de cada cruzamento.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from traffic_rl.config import (
    EvalConfig,
    SignalConfig,
    StarvationConfig,
    TrainConfig,
    TurnShares,
    _deep_update,
)

StreetClass = Literal["avenida", "local"]

# Geometria por classe (mesmos valores da Fase 1)
CLASS_LANES: dict[StreetClass, int] = {"avenida": 3, "local": 1}
CLASS_SPEED_MS: dict[StreetClass, float] = {"avenida": 60 / 3.6, "local": 40 / 3.6}


def slugify(name: str) -> str:
    """Nome de rua -> identificador ASCII estável (usado em ids de veículo)."""
    norm = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "", norm.lower()) or "rua"


class StreetSpec(BaseModel):
    name: str
    street_class: StreetClass = Field(alias="class")
    flow_vph: float = Field(gt=0, description="veíc/h entrando por CADA sentido da rua")

    model_config = {"populate_by_name": True}

    @property
    def slug(self) -> str:
        return slugify(self.name)

    @property
    def lanes(self) -> int:
        return CLASS_LANES[self.street_class]

    @property
    def speed_ms(self) -> float:
        return CLASS_SPEED_MS[self.street_class]


class GridSpec(BaseModel):
    rows: list[StreetSpec]  # ruas leste-oeste, de norte para sul
    cols: list[StreetSpec]  # ruas norte-sul, de oeste para leste
    block_length_m: float = Field(300.0, ge=100.0)
    boundary_length_m: float = Field(300.0, ge=100.0)
    turn_shares: TurnShares = TurnShares()
    truck_share_avenida: float = Field(0.10, ge=0.0, le=0.5)
    truck_share_local: float = Field(0.0, ge=0.0, le=0.5)

    @field_validator("rows", "cols")
    @classmethod
    def _at_least_one(cls, v: list[StreetSpec]) -> list[StreetSpec]:
        if not v:
            raise ValueError("o grid precisa de ao menos 1 rua em cada direção")
        return v

    @model_validator(mode="after")
    def _unique_slugs(self) -> GridSpec:
        slugs = [s.slug for s in self.rows + self.cols]
        if len(slugs) != len(set(slugs)):
            raise ValueError(f"nomes de rua colidem após normalização: {slugs}")
        return self

    @property
    def n_intersections(self) -> int:
        return len(self.rows) * len(self.cols)

    def tls_ids(self) -> list[str]:
        return [f"n_{i}_{j}" for i in range(len(self.rows)) for j in range(len(self.cols))]


class GridEnvConfig(BaseModel):
    """Tempos e modos do ambiente de grid (sem NetworkConfig da Fase 1)."""

    signal: SignalConfig = SignalConfig()
    episode_length_s: float = Field(3600.0, gt=0)
    delta_time_s: float = Field(5.0, ge=1.0)
    reward_mode: str = "r_espera"
    starvation: StarvationConfig = StarvationConfig()
    use_libsumo: bool = True

    @model_validator(mode="after")
    def _check_delta(self) -> GridEnvConfig:
        if self.delta_time_s > self.signal.min_green_s:
            raise ValueError("delta_time_s não pode exceder min_green_s")
        return self


class GridProjectConfig(BaseModel):
    grid: GridSpec
    env: GridEnvConfig = GridEnvConfig()
    train: TrainConfig = TrainConfig()
    eval: EvalConfig = EvalConfig()


def load_grid_config(path: Path | str, overrides: dict | None = None) -> GridProjectConfig:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if overrides:
        raw = _deep_update(raw, overrides)
    raw.pop("scenarios", None)  # overlay do smoke da Fase 1 pode trazer chaves alheias
    return GridProjectConfig(**raw)
