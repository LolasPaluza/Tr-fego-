# TrafficRL — Relatório de avaliação

- Gerado em: 2026-07-20T23:01:03
- Git hash: `edc385a0ab505849b0d979b66b441d50815de00e`
- Episódios por método × cenário (máx.): 100
- Valores: média [IC 95% via bootstrap]. Testes: Mann-Whitney U bicaudal, correção de Bonferroni por cenário.

## Cenário: grid

| Método | Espera média (s) | Espera p95 (s) | Espera máxima (s) | Fila máxima (veíc.) | Throughput (veíc.) | Tempo de viagem médio (s) |
|---|---|---|---|---|---|---|
| Fixo igual | 173.8 [171.1, 176.8] | 327.8 [317.4, 339.2] | 735.0 [691.2, 781.2] | 260.9 [255.3, 267.4] | 2768.7 [2748.2, 2788.7] | 299.5 [295.8, 303.5] |
| Fixo proporcional | 98.9 [96.5, 101.3] | 297.3 [287.6, 307.4] | 727.8 [667.7, 790.0] | 149.4 [144.7, 154.6] | 2830.6 [2809.5, 2851.3] | 209.7 [206.8, 212.7] |
| Atuado (gap) | 268.8 [253.5, 286.1] | 833.0 [769.8, 905.5] | 1790.5 [1629.0, 1977.4] | 392.9 [365.7, 423.0] | 2580.7 [2550.1, 2609.7] | 378.5 [366.0, 391.3] |
| Max-pressure | 509.1 [469.9, 548.1] | 1828.3 [1696.5, 1965.9] | 3424.3 [3364.7, 3475.0] | 827.4 [769.0, 886.8] | 2018.9 [1953.3, 2083.9] | 481.3 [454.2, 510.1] |
| DQN (nosso) | 365.6 [298.3, 434.3] | 1410.9 [1168.5, 1658.4] | 2360.6 [2169.6, 2554.0] | 444.6 [368.4, 523.0] | 2267.7 [2130.3, 2401.6] | 208.5 [196.3, 221.8] |

**Interpretação (grid):** com correção de Bonferroni, não houve diferença significativa frente ao Atuado (gap) (p corrigido = 0.8249); não houve diferença significativa frente ao Fixo igual (p corrigido = 0.8351); não houve diferença significativa frente ao Fixo proporcional (p corrigido = 1.0000); não houve diferença significativa frente ao Max-pressure (p corrigido = 0.5994). Lembrete metodológico: o mérito real do agente é medido contra os baselines fortes (atuado por gap e max-pressure), não apenas contra o tempo fixo.

## Testes estatísticos — DQN vs baselines (espera média)

| Cenário | Baseline | p (bruto) | p (Bonferroni) | Significativo (α=0,05) | DQN melhor? |
|---|---|---|---|---|---|
| grid | Atuado (gap) | 0.2062 | 0.8249 | não | não |
| grid | Fixo igual | 0.2088 | 0.8351 | não | não |
| grid | Fixo proporcional | 0.7540 | 1.0000 | não | não |
| grid | Max-pressure | 0.1499 | 0.5994 | não | sim |

## Figuras

![barras_espera_media_s](figures/barras_espera_media_s.png)
![boxplot_p95_espera](figures/boxplot_p95_espera.png)
![justica_avenida_vs_local_grid](figures/justica_avenida_vs_local_grid.png)
![curvas_aprendizado](figures/curvas_aprendizado.png)
