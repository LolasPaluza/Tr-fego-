"""Ambiente Gymnasium de um cruzamento semaforizado controlado via TraCI/libsumo.

Contrato de segurança (imposto AQUI, nunca delegado ao agente):
- verde mínimo de `min_green_s`: pedidos de troca antes disso são ignorados;
- verde máximo de `max_green_s`: ao atingi-lo, a troca é forçada (estágio
  seguinte cíclico) — evita que qualquer política congele um estágio;
- toda troca passa por amarelo (3 s) e vermelho total (2 s) — estados
  derivados da tabela de fases, aplicados por este wrapper.

Ação: Discrete(4) — escolher o PRÓXIMO estágio verde (manter = repetir o
atual). Com amarelo+all-red = 5 s = delta_time, a decisão pós-transição cai
exatamente no início do verde novo.

Observação: o ambiente sempre calcula a `Observation` estruturada completa
(disponível em `info["observation"]` para os controladores clássicos);
o vetor retornado ao Gymnasium é a projeção do modo configurado
(`obs_minimal`/`obs_rica`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from traffic_rl.config import APPROACHES, N_STAGES, EnvConfig, ObsMode, RewardMode
from traffic_rl.envs.network import TLS_ID
from traffic_rl.envs.observation import Observation, ObservationVectorizer
from traffic_rl.envs.phases import PhaseTable, build_phase_table
from traffic_rl.envs.rewards import RewardCalculator
from traffic_rl.sumo_home import ensure_sumo_home, sumo_binary

_HALT_SPEED = 0.1  # m/s — mesmo limiar do SUMO para "veículo parado"


class SumoIntersectionEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        cfg: EnvConfig,
        net_file: str | Path,
        route_file: str | Path,
        traffic_seed_base: int = 0,
        obs_mode: ObsMode | None = None,
        reward_mode: RewardMode | None = None,
        tripinfo_dir: str | Path | None = None,
    ) -> None:
        super().__init__()
        ensure_sumo_home()
        self.cfg = cfg
        self.net_file = str(net_file)
        self.route_file = str(route_file)
        self.traffic_seed_base = traffic_seed_base
        self.tripinfo_dir = Path(tripinfo_dir) if tripinfo_dir else None
        self.phase_table: PhaseTable = build_phase_table(self.net_file)
        self.vectorizer = ObservationVectorizer(cfg, obs_mode)
        self.reward_calc = RewardCalculator(cfg, reward_mode)

        self.action_space = spaces.Discrete(N_STAGES)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(self.vectorizer.size,), dtype=np.float32)

        net = cfg.network
        self._lanes_by_approach: dict[str, list[str]] = {}
        for a in APPROACHES:
            n = net.avenue_lanes if a in ("E", "W") else net.local_lanes
            self._lanes_by_approach[a] = [f"in_{a}_{i}" for i in range(n)]
        self._all_lanes: list[str] = [
            lane for a in APPROACHES for lane in self._lanes_by_approach[a]
        ]
        self._cap_per_lane = net.approach_length_m / 7.5
        self._speed_limits = {
            "E": net.avenue_speed_ms, "W": net.avenue_speed_ms,
            "N": net.local_speed_ms, "S": net.local_speed_ms,
        }
        # (aproximação, movimento) -> (faixas de entrada, aresta de saída)
        self._movement_lanes: dict[tuple[str, str], list[str]] = {}
        self._movement_out_edge: dict[tuple[str, str], str] = {}
        for link in self.phase_table.links:
            key = (link.approach, link.movement)
            self._movement_lanes.setdefault(key, [])
            if link.from_lane not in self._movement_lanes[key]:
                self._movement_lanes[key].append(link.from_lane)
            self._movement_out_edge[key] = link.to_edge
        self._out_lane_count = {
            f"out_{a}": (net.avenue_lanes if a in ("E", "W") else net.local_lanes)
            for a in APPROACHES
        }

        self._conn: Any = None
        self._traci_label: str | None = None
        self._episode_index = 0
        self._stage = 0
        self._time_in_stage = 0.0
        self._episode_max_queue = 0.0
        self._arrived_cum = 0
        self._current_tripinfo: Path | None = None

    # ------------------------------------------------------------------ SUMO

    def _start_sumo(self, seed: int) -> None:
        cmd = [
            sumo_binary(),
            "-n", self.net_file,
            "-r", self.route_file,
            "--seed", str(seed),
            "--waiting-time-memory", "100000",
            "--time-to-teleport", "-1",
            "--no-step-log", "true",
            "--no-warnings", "true",
            "--duration-log.disable", "true",
        ]
        if self.tripinfo_dir is not None:
            self.tripinfo_dir.mkdir(parents=True, exist_ok=True)
            self._current_tripinfo = self.tripinfo_dir / f"tripinfo_seed{seed}.xml"
            cmd += [
                "--tripinfo-output", str(self._current_tripinfo),
                "--tripinfo-output.write-unfinished", "true",
            ]
        if self.cfg.use_libsumo:
            import libsumo

            libsumo.start(cmd)
            self._conn = libsumo
        else:
            import traci

            self._traci_label = f"env_{id(self)}_{self._episode_index}"
            traci.start(cmd, label=self._traci_label)
            self._conn = traci.getConnection(self._traci_label)

    def _close_sumo(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def _apply_state(self, state: str) -> None:
        self._conn.trafficlight.setRedYellowGreenState(TLS_ID, state)

    def _sim(self, seconds: float) -> None:
        target = self._conn.simulation.getTime() + seconds
        # passo a passo (1 s) para acumular chegadas e fila máxima corretamente
        while self._conn.simulation.getTime() < target - 1e-9:
            self._conn.simulationStep()
            self._arrived_cum += self._conn.simulation.getArrivedNumber()

    # ----------------------------------------------------------- observação

    def _compute_observation(self) -> tuple[Observation, float]:
        """Observação estruturada + espera total acumulada (para r_espera)."""
        conn = self._conn
        queues = np.zeros(len(APPROACHES), dtype=np.float32)
        densities: list[float] = []
        speeds = np.zeros(len(APPROACHES), dtype=np.float32)
        first_waits = np.zeros(len(APPROACHES), dtype=np.float32)
        max_waits = np.zeros(len(APPROACHES), dtype=np.float32)
        total_wait = 0.0
        lane_counts: dict[str, int] = {}

        for ai, approach in enumerate(APPROACHES):
            lanes = self._lanes_by_approach[approach]
            speed_sum, speed_n = 0.0, 0
            front_pos = -1.0
            for lane in lanes:
                halted = conn.lane.getLastStepHaltingNumber(lane)
                count = conn.lane.getLastStepVehicleNumber(lane)
                lane_counts[lane] = count
                queues[ai] += halted
                densities.append(min(count / self._cap_per_lane, 1.0))
                speed_sum += conn.lane.getLastStepMeanSpeed(lane)
                speed_n += 1
                for veh in conn.lane.getLastStepVehicleIDs(lane):
                    wait = conn.vehicle.getAccumulatedWaitingTime(veh)
                    total_wait += wait
                    if wait > max_waits[ai]:
                        max_waits[ai] = wait
                    if conn.vehicle.getSpeed(veh) < _HALT_SPEED:
                        pos = conn.vehicle.getLanePosition(veh)
                        if pos > front_pos:
                            front_pos = pos
                            first_waits[ai] = wait
            speeds[ai] = min(
                max(speed_sum / max(speed_n, 1), 0.0) / self._speed_limits[approach], 1.0
            )

        pressures = np.zeros(N_STAGES, dtype=np.float32)
        from traffic_rl.envs.phases import STAGE_MOVEMENTS

        for stage, movements in enumerate(STAGE_MOVEMENTS):
            for key in movements:
                in_lanes = self._movement_lanes.get(key, [])
                upstream = sum(lane_counts.get(lane, 0) for lane in in_lanes)
                out_edge = self._movement_out_edge.get(key)
                if out_edge is None:
                    continue
                down_total = conn.edge.getLastStepVehicleNumber(out_edge)
                downstream = down_total / self._out_lane_count[out_edge]
                pressures[stage] += upstream - downstream

        obs = Observation(
            stage=self._stage,
            time_in_stage_s=self._time_in_stage,
            queues=queues,
            lane_densities=np.array(densities, dtype=np.float32),
            mean_speeds_norm=speeds,
            first_vehicle_waits_s=first_waits,
            max_waits_s=max_waits,
            stage_pressures=pressures,
        )
        return obs, total_wait

    # ------------------------------------------------------------- gym API

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._close_sumo()
        if options and "traffic_seed" in options:
            traffic_seed = int(options["traffic_seed"])
        elif seed is not None:
            traffic_seed = seed
        else:
            traffic_seed = self.traffic_seed_base + self._episode_index
        self._episode_index += 1
        self._start_sumo(traffic_seed)
        self._stage = 0
        self._time_in_stage = 0.0
        self._episode_max_queue = 0.0
        self._arrived_cum = 0
        self.reward_calc.reset()
        self._apply_state(self.phase_table.green_states[self._stage])
        obs, total_wait = self._compute_observation()
        # inicializa a referência de espera para o primeiro delta de r_espera
        self.reward_calc(obs, total_wait)
        info = self._info(obs)
        return self.vectorizer(obs), info

    def step(self, action: int):
        sig = self.cfg.signal
        action = int(action)
        if not 0 <= action < N_STAGES:
            raise ValueError(f"ação inválida: {action}")

        if action != self._stage and self._time_in_stage < sig.min_green_s:
            action = self._stage  # verde mínimo: troca negada
        elif action == self._stage and self._time_in_stage >= sig.max_green_s:
            action = (self._stage + 1) % N_STAGES  # verde máximo: troca forçada

        if action != self._stage:
            self._apply_state(self.phase_table.yellow_states[self._stage])
            self._sim(sig.yellow_s)
            self._apply_state(self.phase_table.all_red_state)
            self._sim(sig.all_red_s)
            self._stage = action
            self._time_in_stage = 0.0
            self._apply_state(self.phase_table.green_states[self._stage])
            remainder = self.cfg.delta_time_s - (sig.yellow_s + sig.all_red_s)
            if remainder > 0:
                self._sim(remainder)
                self._time_in_stage += remainder
        else:
            self._sim(self.cfg.delta_time_s)
            self._time_in_stage += self.cfg.delta_time_s

        obs, total_wait = self._compute_observation()
        self._episode_max_queue = max(self._episode_max_queue, float(np.sum(obs.queues)))
        reward = self.reward_calc(obs, total_wait)
        sim_time = self._conn.simulation.getTime()
        truncated = sim_time >= self.cfg.episode_length_s
        info = self._info(obs, total_wait=total_wait)
        return self.vectorizer(obs), reward, False, truncated, info

    def _info(self, obs: Observation, total_wait: float = 0.0) -> dict:
        return {
            "observation": obs,
            "sim_time": float(self._conn.simulation.getTime()),
            "total_wait_s": total_wait,
            "arrived_cum": self._arrived_cum,
            "episode_max_queue": self._episode_max_queue,
            "tripinfo_path": str(self._current_tripinfo) if self._current_tripinfo else None,
        }

    def close(self) -> None:
        self._close_sumo()
