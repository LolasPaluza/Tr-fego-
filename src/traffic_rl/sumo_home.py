"""Resolução do SUMO_HOME e dos binários do SUMO.

Prioridade:
1. SUMO_HOME já definido no ambiente (instalação via apt/brew do usuário);
2. Pacote pip `eclipse-sumo` (é o que o projeto instala por padrão) —
   SUMO_HOME é o diretório `sumo/` dentro de site-packages.

A função `ensure_sumo_home()` é chamada em todo ponto de entrada antes de
qualquer uso de traci/sumolib, então nenhum script exige configuração manual.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path


class SumoNotFoundError(RuntimeError):
    pass


def _pip_sumo_home() -> Path | None:
    try:
        import sumo  # pacote eclipse-sumo

        return Path(sumo.__file__).resolve().parent
    except ImportError:
        return None


def ensure_sumo_home() -> Path:
    """Garante SUMO_HOME definido e válido; retorna o caminho."""
    env = os.environ.get("SUMO_HOME")
    if env and (Path(env) / "bin").is_dir():
        return Path(env)
    pip_home = _pip_sumo_home()
    if pip_home and (pip_home / "bin").is_dir():
        os.environ["SUMO_HOME"] = str(pip_home)
        return pip_home
    # Instalação de sistema (apt) sem SUMO_HOME: deduz de `which sumo`.
    which = shutil.which("sumo")
    if which:
        candidate = Path(which).resolve().parent.parent / "share" / "sumo"
        if candidate.is_dir():
            os.environ["SUMO_HOME"] = str(candidate)
            return candidate
    raise SumoNotFoundError(
        "SUMO não encontrado. Instale com `pip install eclipse-sumo` (recomendado) "
        "ou defina SUMO_HOME apontando para a instalação do SUMO (>= 1.19)."
    )


def sumo_binary() -> str:
    """Caminho do binário `sumo` (headless — nunca sumo-gui)."""
    home = ensure_sumo_home()
    candidate = home / "bin" / "sumo"
    if candidate.exists():
        return str(candidate)
    which = shutil.which("sumo")
    if which:
        return which
    raise SumoNotFoundError("binário `sumo` não encontrado nem em SUMO_HOME/bin nem no PATH")


def netconvert_binary() -> str:
    home = ensure_sumo_home()
    candidate = home / "bin" / "netconvert"
    if candidate.exists():
        return str(candidate)
    which = shutil.which("netconvert")
    if which:
        return which
    raise SumoNotFoundError("binário `netconvert` não encontrado")
