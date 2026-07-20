"""Configuração da Fase 4: malha real OSM + demanda provisória.

O YAML aponta para um recorte .osm (extraído fora do sandbox — ver
docs/PENDENTE_LOCAL.md) e define o volume total de viagens. Os demais blocos
(semáforo, treino, avaliação) são os mesmos das fases anteriores.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from traffic_rl.config import EvalConfig, TrainConfig, _deep_update
from traffic_rl.grid_config import GridEnvConfig


class OsmDemandConfig(BaseModel):
    """Demanda provisória (randomTrips) até a OD calibrada entrar (ADR-019)."""

    total_vph: float = Field(2000.0, gt=0, description="viagens/hora no recorte inteiro")


class OsmProjectConfig(BaseModel):
    osm_file: str
    env: GridEnvConfig = GridEnvConfig()
    demand: OsmDemandConfig = OsmDemandConfig()
    train: TrainConfig = TrainConfig()
    eval: EvalConfig = EvalConfig()


def load_osm_config(path: Path | str, overrides: dict | None = None) -> OsmProjectConfig:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if overrides:
        raw = _deep_update(raw, overrides)
    raw.pop("scenarios", None)
    raw.pop("grid", None)
    return OsmProjectConfig(**raw)
