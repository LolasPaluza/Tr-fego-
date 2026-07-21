# TrafficRL — Relatório de avaliação — grid 3×3 (oficial, 5 seeds × 600k, 20 eps)

- Gerado em: 2026-07-21T16:39:23
- Git hash: `ebda6058ebabf619d29b8b7080cc313564fc29d8`
- Episódios por método × cenário (máx.): 100
- Valores: média [IC 95% via bootstrap]. Testes: Mann-Whitney U bicaudal, correção de Bonferroni por cenário.

## Cenário: grid

| Método | Espera média (s) | Espera p95 (s) | Espera máxima (s) | Fila máxima (veíc.) | Throughput (veíc.) | Tempo de viagem médio (s) |
|---|---|---|---|---|---|---|
| Fixo igual | 173.8 [171.1, 176.8] | 327.8 [317.4, 339.2] | 735.0 [691.2, 781.2] | 260.9 [255.3, 267.4] | 2768.7 [2748.2, 2788.7] | 299.5 [295.8, 303.5] |
| Fixo proporcional | 98.9 [96.5, 101.3] | 297.3 [287.6, 307.4] | 727.8 [667.7, 790.0] | 149.4 [144.7, 154.6] | 2830.6 [2809.5, 2851.3] | 209.7 [206.8, 212.7] |
| Atuado (gap) | 268.8 [253.5, 286.1] | 833.0 [769.8, 905.5] | 1790.5 [1629.0, 1977.4] | 392.9 [365.7, 423.0] | 2580.7 [2550.1, 2609.7] | 378.5 [366.0, 391.3] |
| Max-pressure | 520.3 [478.4, 564.4] | 1857.8 [1713.5, 2008.5] | 3440.4 [3359.7, 3493.3] | 854.0 [795.1, 912.6] | 1988.2 [1916.6, 2056.0] | 481.0 [450.5, 512.6] |
| DQN (nosso) | 234.6 [187.0, 284.1] | 1044.4 [844.4, 1253.3] | 2213.4 [2041.0, 2389.6] | 334.9 [273.2, 399.5] | 2530.0 [2434.2, 2621.2] | 191.9 [185.8, 198.4] |

**Interpretação (grid):** com correção de Bonferroni, o DQN reduziu a espera média em 13% frente ao Atuado (gap) (p corrigido = 0.0018); o Fixo igual superou o DQN (35% de vantagem, p corrigido = 0.0436) — resultado reportado honestamente; não houve diferença significativa frente ao Fixo proporcional (p corrigido = 1.0000); o DQN reduziu a espera média em 55% frente ao Max-pressure (p corrigido = 0.0001). Lembrete metodológico: o mérito real do agente é medido contra os baselines fortes (atuado por gap e max-pressure), não apenas contra o tempo fixo.

## Testes estatísticos — DQN vs baselines (espera média)

| Cenário | Baseline | p (bruto) | p (Bonferroni) | Significativo (α=0,05) | DQN melhor? |
|---|---|---|---|---|---|
| grid | Atuado (gap) | 0.0004 | 0.0018 | sim | sim |
| grid | Fixo igual | 0.0109 | 0.0436 | sim | não |
| grid | Fixo proporcional | 0.7540 | 1.0000 | não | não |
| grid | Max-pressure | 0.0000 | 0.0001 | sim | sim |

## Figuras

![barras_espera_media_s](figures/barras_espera_media_s.png)
![boxplot_p95_espera](figures/boxplot_p95_espera.png)
![justica_avenida_vs_local_grid](figures/justica_avenida_vs_local_grid.png)
![curvas_aprendizado](figures/curvas_aprendizado.png)
