"""Contrato central do projeto: a interface Controller.

Baselines clássicos e agente RL implementam EXATAMENTE a mesma interface —
`act(obs) -> ação` sobre a `Observation` estruturada. É isso que permite ao
protocolo de avaliação tratar todos os métodos identicamente e, na Fase 2,
instanciar um controlador por cruzamento.

A ação é o índice do PRÓXIMO estágio verde desejado (manter = repetir o
atual). Tempos de segurança são imposição do ambiente; um controlador não
consegue violá-los mesmo que tente.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from traffic_rl.envs.observation import Observation


class Controller(ABC):
    """Política de controle de um cruzamento."""

    name: str = "controller"

    def reset(self) -> None:
        """Chamado no início de cada episódio (limpa estado interno)."""

    @abstractmethod
    def act(self, obs: Observation) -> int:
        """Escolhe o próximo estágio verde (0..N_STAGES-1)."""
        raise NotImplementedError
