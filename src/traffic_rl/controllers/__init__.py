"""Controladores: interface Controller, baselines clássicos e adaptador DQN."""

from traffic_rl.controllers.base import Controller
from traffic_rl.controllers.registry import BASELINE_NAMES, make_controller

__all__ = ["Controller", "BASELINE_NAMES", "make_controller"]
