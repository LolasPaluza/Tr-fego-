"""Ambiente do grid: K semáforos, uma simulação, política compartilhada.

Modelo multi-agente (independent learners com parameter sharing):
- cada cruzamento é um "sub-ambiente" com observação LOCAL (obs_minimal:
  estágio one-hot + tempo no verde + filas das 4 aproximações) e recompensa
  LOCAL (mesmas funções da Fase 1, restritas às suas aproximações);
- `GridVecEnv` expõe os K cruzamentos como um VecEnv do sb3 com num_envs=K:
  o DQN coleta K transições por passo e treina UMA rede usada por todos —
  parameter sharing clássico, e a razão de a observação ter tamanho fixo.

Segurança idêntica à Fase 1, POR cruzamento: verde mín 10 s (troca negada),
verde máx (troca cíclica forçada), amarelo 3 s + all-red 2 s em toda troca.
Como amarelo+all-red = delta_time, os semáforos que trocam atravessam a
transição enquanto os demais seguem verdes — decisões continuam síncronas.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import sumolib
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from traffic_rl.config import APPROACHES, N_STAGES
from traffic_rl.envs.grid_demand import build_grid_routes
from traffic_rl.envs.grid_network import GridTopology, build_grid_network
from traffic_rl.envs.observation import Observation
from traffic_rl.envs.phases import STAGE_MOVEMENTS, PhaseTable, build_phase_table
from traffic_rl.envs.rewards import RewardCalculator
from traffic_rl.grid_config import GridProjectConfig
from traffic_rl.paths import generated_dir
from traffic_rl.sumo_home import ensure_sumo_home, sumo_binary

_HALT_SPEED = 0.1


class GridMinimalVectorizer:
    """Projeção obs_minimal [0,1] com capacidades do próprio cruzamento."""

    size = N_STAGES + 1 + len(APPROACHES)  # 9

    def __init__(self, queue_caps: np.ndarray, max_green_s: float) -> None:
        self._caps = queue_caps.astype(np.float32)
        self._max_green = max_green_s

    def __call__(self, obs: Observation) -> np.ndarray:
        one_hot = np.zeros(N_STAGES, dtype=np.float32)
        one_hot[obs.stage] = 1.0
        t = np.array([min(obs.time_in_stage_s / self._max_green, 1.0)], dtype=np.float32)
        queues = np.clip(obs.queues / self._caps, 0.0, 1.0)
        return np.concatenate([one_hot, t, queues]).astype(np.float32)


class _TLSState:
    """Estado e metadados de um semáforo do grid."""

    def __init__(
        self,
        tls_id: str,
        table: PhaseTable,
        lanes_by_approach: dict[str, list[str]],
        lane_caps: dict[str, float],
        speed_by_approach: dict[str, float],
        out_lane_count: dict[str, int],
        vectorizer: GridMinimalVectorizer,
        reward_calc: RewardCalculator,
    ) -> None:
        self.tls_id = tls_id
        self.table = table
        self.lanes_by_approach = lanes_by_approach
        self.lane_caps = lane_caps
        self.speed_by_approach = speed_by_approach
        self.out_lane_count = out_lane_count
        self.vectorizer = vectorizer
        self.reward_calc = reward_calc
        # (aproximação, movimento) -> faixas de entrada / aresta de saída
        self.movement_lanes: dict[tuple[str, str], list[str]] = {}
        self.movement_out_edge: dict[tuple[str, str], str] = {}
        for link in table.links:
            key = (link.approach, link.movement)
            self.movement_lanes.setdefault(key, [])
            if link.from_lane not in self.movement_lanes[key]:
                self.movement_lanes[key].append(link.from_lane)
            self.movement_out_edge[key] = link.to_edge
        self.stage = 0
        self.time_in_stage = 0.0

    def queue_caps(self) -> np.ndarray:
        return np.array(
            [
                sum(self.lane_caps[lane] for lane in self.lanes_by_approach[a])
                for a in APPROACHES
            ],
            dtype=np.float32,
        )


class GridTrafficEnv:
    """Núcleo da simulação multi-semáforo (não é gym.Env — ver GridVecEnv)."""

    def __init__(
        self,
        cfg: GridProjectConfig,
        use_libsumo: bool | None = None,
        tripinfo_dir: str | Path | None = None,
        label: str = "grid",
    ) -> None:
        ensure_sumo_home()
        self.cfg = cfg
        self.topo = GridTopology(cfg.grid)
        self.use_libsumo = cfg.env.use_libsumo if use_libsumo is None else use_libsumo
        self.tripinfo_dir = Path(tripinfo_dir) if tripinfo_dir else None
        self._label = label
        self.net_file = build_grid_network(cfg.grid, generated_dir())
        net = sumolib.net.readNet(str(self.net_file))

        self.tls: list[_TLSState] = []
        for i in range(self.topo.R):
            for j in range(self.topo.C):
                tls_id = f"n_{i}_{j}"
                table = build_phase_table(str(self.net_file), tls_id)
                incoming = self.topo.incoming_edges(i, j)
                lanes_by_approach: dict[str, list[str]] = {}
                lane_caps: dict[str, float] = {}
                speed_by_approach: dict[str, float] = {}
                out_lane_count: dict[str, int] = {}
                for approach, edge_id in incoming.items():
                    street = self.topo.street_of_edge(edge_id)
                    lanes = [f"{edge_id}_{k}" for k in range(street.lanes)]
                    lanes_by_approach[approach] = lanes
                    length = net.getEdge(edge_id).getLength()
                    for lane in lanes:
                        lane_caps[lane] = max(length / 7.5, 1.0)
                    speed_by_approach[approach] = street.speed_ms
                for link in table.links:
                    out_lane_count[link.to_edge] = self.topo.street_of_edge(link.to_edge).lanes
                state = _TLSState(
                    tls_id,
                    table,
                    lanes_by_approach,
                    lane_caps,
                    speed_by_approach,
                    out_lane_count,
                    vectorizer=None,  # type: ignore[arg-type]
                    reward_calc=RewardCalculator(_reward_cfg(cfg)),
                )
                state.vectorizer = GridMinimalVectorizer(
                    state.queue_caps(), cfg.env.signal.max_green_s
                )
                self.tls.append(state)

        self.n_tls = len(self.tls)
        self._conn: Any = None
        self._episode_max_queue = 0.0
        self._arrived_cum = 0
        self._current_tripinfo: Path | None = None

    # ------------------------------------------------------------------ sumo

    def _start_sumo(self, traffic_seed: int) -> None:
        route_file = build_grid_routes(
            self.cfg.grid, generated_dir(), traffic_seed, self.cfg.env.episode_length_s
        )
        cmd = [
            sumo_binary(),
            "-n", str(self.net_file),
            "-r", str(route_file),
            "--seed", str(traffic_seed),
            "--waiting-time-memory", "100000",
            "--time-to-teleport", "-1",
            "--no-step-log", "true",
            "--no-warnings", "true",
            "--duration-log.disable", "true",
        ]
        if self.tripinfo_dir is not None:
            self.tripinfo_dir.mkdir(parents=True, exist_ok=True)
            self._current_tripinfo = self.tripinfo_dir / f"tripinfo_seed{traffic_seed}.xml"
            cmd += [
                "--tripinfo-output", str(self._current_tripinfo),
                "--tripinfo-output.write-unfinished", "true",
            ]
        if self.use_libsumo:
            import libsumo

            try:
                libsumo.close()
            except Exception:
                pass
            libsumo.start(cmd)
            self._conn = libsumo
        else:
            import traci

            label = f"{self._label}_{traffic_seed}"
            traci.start(cmd, label=label)
            self._conn = traci.getConnection(label)

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def _sim(self, seconds: float) -> None:
        target = self._conn.simulation.getTime() + seconds
        while self._conn.simulation.getTime() < target - 1e-9:
            self._conn.simulationStep()
            self._arrived_cum += self._conn.simulation.getArrivedNumber()

    # ------------------------------------------------------------ observação

    def _observe_tls(self, st: _TLSState) -> tuple[Observation, float]:
        conn = self._conn
        queues = np.zeros(len(APPROACHES), dtype=np.float32)
        densities: list[float] = []
        speeds = np.zeros(len(APPROACHES), dtype=np.float32)
        first_waits = np.zeros(len(APPROACHES), dtype=np.float32)
        max_waits = np.zeros(len(APPROACHES), dtype=np.float32)
        total_wait = 0.0
        lane_counts: dict[str, int] = {}
        for ai, approach in enumerate(APPROACHES):
            speed_sum, speed_n = 0.0, 0
            front_pos = -1.0
            for lane in st.lanes_by_approach[approach]:
                queues[ai] += conn.lane.getLastStepHaltingNumber(lane)
                count = conn.lane.getLastStepVehicleNumber(lane)
                lane_counts[lane] = count
                densities.append(min(count / st.lane_caps[lane], 1.0))
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
                max(speed_sum / max(speed_n, 1), 0.0) / st.speed_by_approach[approach], 1.0
            )
        pressures = np.zeros(N_STAGES, dtype=np.float32)
        for stage, movements in enumerate(STAGE_MOVEMENTS):
            for key in movements:
                in_lanes = st.movement_lanes.get(key, [])
                if not in_lanes:
                    continue
                upstream = sum(lane_counts.get(lane, 0) for lane in in_lanes)
                out_edge = st.movement_out_edge[key]
                down = conn.edge.getLastStepVehicleNumber(out_edge) / st.out_lane_count[out_edge]
                pressures[stage] += upstream - down
        obs = Observation(
            stage=st.stage,
            time_in_stage_s=st.time_in_stage,
            queues=queues,
            lane_densities=np.array(densities, dtype=np.float32),
            mean_speeds_norm=speeds,
            first_vehicle_waits_s=first_waits,
            max_waits_s=max_waits,
            stage_pressures=pressures,
        )
        return obs, total_wait

    def _observe_all(self) -> tuple[list[Observation], list[float]]:
        observations, waits = [], []
        for st in self.tls:
            obs, wait = self._observe_tls(st)
            observations.append(obs)
            waits.append(wait)
        return observations, waits

    # ---------------------------------------------------------------- api

    def reset(self, traffic_seed: int) -> tuple[list[Observation], dict]:
        self.close()
        self._start_sumo(traffic_seed)
        self._episode_max_queue = 0.0
        self._arrived_cum = 0
        for st in self.tls:
            st.stage = 0
            st.time_in_stage = 0.0
            st.reward_calc.reset()
            self._conn.trafficlight.setRedYellowGreenState(
                st.tls_id, st.table.green_states[0]
            )
        observations, waits = self._observe_all()
        for st, obs, wait in zip(self.tls, observations, waits, strict=True):
            st.reward_calc(obs, wait)  # referência inicial do r_espera local
        return observations, self._info()

    def step(self, actions: list[int]) -> tuple[list[Observation], list[float], bool, dict]:
        sig = self.cfg.env.signal
        if len(actions) != self.n_tls:
            raise ValueError(f"esperadas {self.n_tls} ações, recebidas {len(actions)}")
        targets: list[int] = []
        for st, action in zip(self.tls, actions, strict=True):
            action = int(action)
            if not 0 <= action < N_STAGES:
                raise ValueError(f"ação inválida: {action}")
            if action != st.stage and st.time_in_stage < sig.min_green_s:
                action = st.stage
            elif action == st.stage and st.time_in_stage >= sig.max_green_s:
                action = (st.stage + 1) % N_STAGES
            targets.append(action)

        switching = [st.stage != tgt for st, tgt in zip(self.tls, targets, strict=True)]
        if any(switching):
            for st, sw in zip(self.tls, switching, strict=True):
                if sw:
                    self._conn.trafficlight.setRedYellowGreenState(
                        st.tls_id, st.table.yellow_states[st.stage]
                    )
            self._sim(sig.yellow_s)
            for st, sw in zip(self.tls, switching, strict=True):
                if sw:
                    self._conn.trafficlight.setRedYellowGreenState(
                        st.tls_id, st.table.all_red_state
                    )
            self._sim(sig.all_red_s)
            remainder = self.cfg.env.delta_time_s - (sig.yellow_s + sig.all_red_s)
            for st, sw, tgt in zip(self.tls, switching, targets, strict=True):
                if sw:
                    st.stage = tgt
                    st.time_in_stage = 0.0
                    self._conn.trafficlight.setRedYellowGreenState(
                        st.tls_id, st.table.green_states[tgt]
                    )
                else:
                    st.time_in_stage += sig.yellow_s + sig.all_red_s
            if remainder > 0:
                self._sim(remainder)
                for st in self.tls:
                    st.time_in_stage += remainder
        else:
            self._sim(self.cfg.env.delta_time_s)
            for st in self.tls:
                st.time_in_stage += self.cfg.env.delta_time_s

        observations, waits = self._observe_all()
        rewards = [
            st.reward_calc(obs, wait)
            for st, obs, wait in zip(self.tls, observations, waits, strict=True)
        ]
        total_queue = float(sum(np.sum(o.queues) for o in observations))
        self._episode_max_queue = max(self._episode_max_queue, total_queue)
        done = self._conn.simulation.getTime() >= self.cfg.env.episode_length_s
        return observations, rewards, done, self._info()

    def _info(self) -> dict:
        return {
            "sim_time": float(self._conn.simulation.getTime()),
            "arrived_cum": self._arrived_cum,
            "episode_max_queue": self._episode_max_queue,
            "tripinfo_path": str(self._current_tripinfo) if self._current_tripinfo else None,
        }

    def vectorize(self, observations: list[Observation]) -> np.ndarray:
        return np.stack(
            [st.vectorizer(obs) for st, obs in zip(self.tls, observations, strict=True)]
        )


def _reward_cfg(cfg: GridProjectConfig):
    """RewardCalculator da Fase 1 recebe um objeto com reward_mode/starvation."""
    from traffic_rl.config import EnvConfig

    return EnvConfig(
        reward_mode=cfg.env.reward_mode,  # type: ignore[arg-type]
        starvation=cfg.env.starvation,
    )


class GridVecEnv(VecEnv):
    """Os K cruzamentos como VecEnv do sb3 (num_envs = K, política única)."""

    def __init__(self, cfg: GridProjectConfig, traffic_seed_base: int = 0,
                 use_libsumo: bool | None = None, label: str = "grid") -> None:
        self.core = GridTrafficEnv(cfg, use_libsumo=use_libsumo, label=label)
        self.traffic_seed_base = traffic_seed_base
        self._episode_index = 0
        self._actions: np.ndarray | None = None
        observation_space = spaces.Box(
            0.0, 1.0, shape=(GridMinimalVectorizer.size,), dtype=np.float32
        )
        super().__init__(self.core.n_tls, observation_space, spaces.Discrete(N_STAGES))

    # ------------------------------------------------------------- VecEnv API

    def reset(self) -> np.ndarray:
        seed = self.traffic_seed_base + self._episode_index
        self._episode_index += 1
        observations, _ = self.core.reset(seed)
        return self.core.vectorize(observations)

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = actions

    def step_wait(self):
        assert self._actions is not None
        observations, rewards, done, info = self.core.step(list(self._actions))
        obs_vec = self.core.vectorize(observations)
        dones = np.full(self.num_envs, done, dtype=bool)
        infos: list[dict] = [dict(info) for _ in range(self.num_envs)]
        if done:
            for k in range(self.num_envs):
                infos[k]["TimeLimit.truncated"] = True
                infos[k]["terminal_observation"] = obs_vec[k]
            obs_vec = self.reset()
        return obs_vec, np.array(rewards, dtype=np.float32), dones, infos

    def close(self) -> None:
        self.core.close()

    def get_attr(self, attr_name: str, indices=None):
        n = self.num_envs if indices is None else len(self._get_indices(indices))
        if attr_name == "render_mode":
            return [None] * n
        return [getattr(self.core, attr_name)] * n

    def set_attr(self, attr_name: str, value, indices=None) -> None:
        setattr(self.core, attr_name, value)

    def env_method(self, method_name: str, *args, indices=None, **kwargs):
        n = self.num_envs if indices is None else len(self._get_indices(indices))
        return [getattr(self.core, method_name)(*args, **kwargs)] * n

    def env_is_wrapped(self, wrapper_class, indices=None):
        n = self.num_envs if indices is None else len(self._get_indices(indices))
        return [False] * n

    def seed(self, seed: int | None = None):
        if seed is not None:
            self.traffic_seed_base = seed
        return [self.traffic_seed_base] * self.num_envs
