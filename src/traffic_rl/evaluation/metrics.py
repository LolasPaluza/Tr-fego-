"""Métricas por episódio a partir do tripinfo do SUMO.

O tripinfo é a fonte da verdade por veículo (espera, viagem, conclusão),
incluindo veículos NÃO concluídos (--tripinfo-output.write-unfinished) —
essencial para o starvation check: um veículo preso 900 s na via local
aparece aqui mesmo sem ter cruzado.

Métricas por episódio (spec Parte 4):
- espera média, p95 da espera (justiça), espera máxima de um único veículo
  (starvation), fila máxima (do ambiente), throughput (veículos concluídos),
  tempo médio de viagem (só concluídos), e espera média por classe viária
  (avenida vs local — trade-off de justiça).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from traffic_rl.envs.demand import road_class_of_vehicle


@dataclass(frozen=True)
class EpisodeMetrics:
    method: str
    scenario: str
    episode: int
    traffic_seed: int
    espera_media_s: float
    espera_p95_s: float
    espera_max_s: float
    fila_maxima: float
    throughput: int
    tempo_viagem_medio_s: float
    espera_media_avenida_s: float
    espera_media_local_s: float
    n_veiculos: int

    def as_dict(self) -> dict:
        return asdict(self)


def parse_tripinfo(
    tripinfo_path: str | Path,
    method: str,
    scenario: str,
    episode: int,
    traffic_seed: int,
    fila_maxima: float,
    classifier: Callable[[str], str] = road_class_of_vehicle,
) -> EpisodeMetrics:
    root = ET.parse(str(tripinfo_path)).getroot()
    waits: list[float] = []
    travel_finished: list[float] = []
    waits_by_class: dict[str, list[float]] = {"avenida": [], "local": []}
    throughput = 0
    for trip in root.iter("tripinfo"):
        veh_id = trip.get("id", "")
        wait = float(trip.get("waitingTime", 0.0))
        waits.append(wait)
        waits_by_class[classifier(veh_id)].append(wait)
        arrival = float(trip.get("arrival", -1.0))
        if arrival >= 0:
            throughput += 1
            travel_finished.append(float(trip.get("duration", 0.0)))
    if not waits:
        waits = [0.0]
    return EpisodeMetrics(
        method=method,
        scenario=scenario,
        episode=episode,
        traffic_seed=traffic_seed,
        espera_media_s=float(np.mean(waits)),
        espera_p95_s=float(np.percentile(waits, 95)),
        espera_max_s=float(np.max(waits)),
        fila_maxima=fila_maxima,
        throughput=throughput,
        tempo_viagem_medio_s=float(np.mean(travel_finished)) if travel_finished else 0.0,
        espera_media_avenida_s=(
            float(np.mean(waits_by_class["avenida"])) if waits_by_class["avenida"] else 0.0
        ),
        espera_media_local_s=(
            float(np.mean(waits_by_class["local"])) if waits_by_class["local"] else 0.0
        ),
        n_veiculos=len(waits),
    )
