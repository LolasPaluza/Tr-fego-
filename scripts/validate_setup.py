#!/usr/bin/env python3
"""Validação do ambiente — PRIMEIRA coisa a rodar na máquina local.

Uso:  python scripts/validate_setup.py
Saída: uma linha ✅/❌ por verificação; código de saída 0 só se tudo passar.

Verifica: Python, imports das dependências, SUMO instalado e versão,
SUMO_HOME, geração de rede e rotas, e um episódio de 60 s de simulação.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

FALHAS: list[str] = []


def check(nome: str, fn) -> None:
    try:
        detalhe = fn()
        print(f"✅ {nome}" + (f" — {detalhe}" if detalhe else ""))
    except Exception as e:  # noqa: BLE001 — script de diagnóstico
        print(f"❌ {nome} — {type(e).__name__}: {e}")
        FALHAS.append(nome)


def python_version():
    if sys.version_info < (3, 11):  # noqa: UP036 — diagnóstico intencional fora do pip
        raise RuntimeError(f"Python >= 3.11 necessário (atual: {sys.version.split()[0]})")
    return sys.version.split()[0]


def imports():
    import gymnasium  # noqa: F401
    import pandas  # noqa: F401
    import pydantic  # noqa: F401
    import scipy  # noqa: F401
    import stable_baselines3  # noqa: F401
    import sumolib  # noqa: F401
    import torch  # noqa: F401
    import traci  # noqa: F401

    return "gymnasium, sb3, torch, traci, sumolib, pandas, scipy, pydantic"


def libsumo_ok():
    import libsumo  # noqa: F401

    return "libsumo disponível (simulação rápida)"


def sumo_home():
    from traffic_rl.sumo_home import ensure_sumo_home

    return str(ensure_sumo_home())


def sumo_version():
    from traffic_rl.sumo_home import sumo_binary

    out = subprocess.run([sumo_binary(), "--version"], capture_output=True, text=True, timeout=30)
    linha = out.stdout.splitlines()[0] if out.stdout else "?"
    versao = linha.split()[-1]
    major, minor = (int(x) for x in versao.split(".")[:2])
    if (major, minor) < (1, 19):
        raise RuntimeError(f"SUMO >= 1.19 necessário (encontrado {versao})")
    return linha


def config_carrega():
    from traffic_rl.config import load_config
    from traffic_rl.paths import configs_dir

    cfg = load_config(configs_dir() / "default.yaml")
    return f"{len(cfg.scenarios)} cenários"


def rede_e_rotas():
    from traffic_rl.config import load_config
    from traffic_rl.envs.demand import build_routes
    from traffic_rl.envs.network import build_network
    from traffic_rl.envs.phases import build_phase_table
    from traffic_rl.paths import configs_dir

    cfg = load_config(configs_dir() / "default.yaml")
    with tempfile.TemporaryDirectory() as tmp:
        net = build_network(cfg.env.network, Path(tmp))
        build_routes(cfg.scenario("pico_assimetrico"), Path(tmp))
        table = build_phase_table(str(net))
        return f"rede ok, {table.n_links} links semaforizados, 4 estágios"


def episodio_60s():
    from traffic_rl.config import load_config
    from traffic_rl.envs import make_env
    from traffic_rl.paths import configs_dir

    cfg = load_config(
        configs_dir() / "default.yaml", overrides={"env": {"episode_length_s": 60.0}}
    )
    env = make_env(cfg, "pico_assimetrico")
    try:
        env.reset(options={"traffic_seed": 1})
        done, passos = False, 0
        while not done:
            _, _, term, trunc, info = env.step(0)
            done = term or trunc
            passos += 1
        return f"{passos} decisões, {info['arrived_cum']} veículos concluíram"
    finally:
        env.close()


def main() -> int:
    print("=== TrafficRL — validação do ambiente ===\n")
    check("Python >= 3.11", python_version)
    check("Imports das dependências", imports)
    check("libsumo (opcional, recomendado)", libsumo_ok)
    check("SUMO_HOME", sumo_home)
    check("SUMO >= 1.19", sumo_version)
    check("Config default carrega e valida", config_carrega)
    check("Rede e rotas geram (netconvert)", rede_e_rotas)
    check("Episódio de 60 s roda", episodio_60s)
    print()
    if FALHAS:
        print(f"❌ {len(FALHAS)} verificação(ões) falharam: {', '.join(FALHAS)}")
        print("   Consulte a seção de instalação do README.md.")
        return 1
    print("✅ Ambiente pronto. Próximo passo: pipeline smoke "
          "(ver README §Rodando na máquina local).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
