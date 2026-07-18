"""Caminhos do projeto, sempre relativos à raiz (nada hardcoded)."""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    """Raiz do projeto.

    Ordem de resolução:
    1. Variável de ambiente TRAFFIC_RL_ROOT (útil para instalação via pip em outra máquina);
    2. Diretório que contém `pyproject.toml`, subindo a partir deste arquivo (checkout do repo);
    3. Diretório de trabalho atual, como último recurso.
    """
    env = os.environ.get("TRAFFIC_RL_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def configs_dir() -> Path:
    return project_root() / "configs"


def generated_dir() -> Path:
    """Artefatos regeneráveis do SUMO (.net.xml, .rou.xml)."""
    d = project_root() / "data" / "generated"
    d.mkdir(parents=True, exist_ok=True)
    return d


def results_dir() -> Path:
    d = project_root() / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d


def runs_dir() -> Path:
    d = results_dir() / "runs"
    d.mkdir(parents=True, exist_ok=True)
    return d
