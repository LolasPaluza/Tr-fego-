"""Double DQN sobre o DQN do stable-baselines3.

O DQN do sb3 é o vanilla (alvo = max do target network), que superestima
valores Q. Double DQN (van Hasselt et al., 2016) desacopla SELEÇÃO
(rede online) de AVALIAÇÃO (rede alvo) da ação do próximo estado. Só o
cálculo do alvo muda — todo o resto (replay, epsilon, otimização) é herdado.
`train()` abaixo replica o do sb3 2.9 com essa única modificação.
"""

from __future__ import annotations

import numpy as np
import torch as th
from stable_baselines3 import DQN
from torch.nn import functional as F


class DoubleDQN(DQN):
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            with th.no_grad():
                # Double DQN: rede ONLINE seleciona a ação do próximo estado...
                next_actions = self.q_net(replay_data.next_observations).argmax(dim=1, keepdim=True)
                # ...e a rede ALVO avalia o valor dessa ação.
                next_q_values = th.gather(
                    self.q_net_target(replay_data.next_observations), dim=1, index=next_actions
                )
                target_q_values = (
                    replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values
                )

            current_q_values = th.gather(
                self.q_net(replay_data.observations), dim=1, index=replay_data.actions.long()
            )
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", float(np.mean(losses)))
