# TrafficRL — Relatório de avaliação — grid COORDENADO

- Gerado em: 2026-07-26T22:23:11
- Git hash: `a9c8434b3d7a6baa2b77943e2b3ef7afb1b5b337`
- Episódios por método × cenário (máx.): 100
- Valores: média [IC 95% via bootstrap]. Testes: Mann-Whitney U bicaudal, correção de Bonferroni por cenário.

## Cenário: grid

| Método | Espera média (s) | Espera p95 (s) | Espera máxima (s) | Fila máxima (veíc.) | Throughput (veíc.) | Tempo de viagem médio (s) |
|---|---|---|---|---|---|---|
| Fixo igual | 171.4 [168.5, 174.7] | 328.3 [316.7, 341.6] | 724.9 [670.6, 781.9] | 257.8 [252.6, 263.2] | 2769.8 [2750.1, 2789.4] | 295.2 [291.3, 299.4] |
| Fixo proporcional | 99.2 [96.9, 101.5] | 299.2 [290.1, 308.8] | 752.2 [685.8, 820.5] | 148.2 [143.3, 153.4] | 2831.2 [2809.6, 2852.0] | 209.6 [206.8, 212.4] |
| Atuado (gap) | 271.4 [255.6, 288.1] | 835.2 [776.1, 899.8] | 1788.2 [1611.9, 1980.5] | 396.9 [365.9, 432.5] | 2577.7 [2545.2, 2608.3] | 380.9 [367.9, 393.9] |
| Max-pressure | 492.6 [462.7, 526.9] | 1751.7 [1639.6, 1871.0] | 3407.9 [3353.2, 3459.8] | 811.0 [762.9, 861.4] | 2065.1 [2009.5, 2118.4] | 485.9 [463.4, 510.2] |
| DQN (nosso) | 229.8 [174.0, 290.4] | 729.5 [567.4, 905.0] | 1933.6 [1691.0, 2181.1] | 330.0 [256.1, 409.8] | 2524.2 [2399.9, 2640.0] | 245.1 [221.9, 270.2] |

**Interpretação (grid):** com correção de Bonferroni, o DQN reduziu a espera média em 15% frente ao Atuado (gap) (p corrigido = 0.0001); o Fixo igual superou o DQN (34% de vantagem, p corrigido = 0.0180) — resultado reportado honestamente; não houve diferença significativa frente ao Fixo proporcional (p corrigido = 0.6403); o DQN reduziu a espera média em 53% frente ao Max-pressure (p corrigido = 0.0000). Lembrete metodológico: o mérito real do agente é medido contra os baselines fortes (atuado por gap e max-pressure), não apenas contra o tempo fixo.

## Testes estatísticos — DQN vs baselines (espera média)

| Cenário | Baseline | p (bruto) | p (Bonferroni) | Significativo (α=0,05) | DQN melhor? |
|---|---|---|---|---|---|
| grid | Atuado (gap) | 0.0000 | 0.0001 | sim | sim |
| grid | Fixo igual | 0.0045 | 0.0180 | sim | não |
| grid | Fixo proporcional | 0.1601 | 0.6403 | não | não |
| grid | Max-pressure | 0.0000 | 0.0000 | sim | sim |

## Figuras

![barras_espera_media_s](figures/barras_espera_media_s.png)
![boxplot_p95_espera](figures/boxplot_p95_espera.png)
![justica_avenida_vs_local_grid](figures/justica_avenida_vs_local_grid.png)
![curvas_aprendizado](figures/curvas_aprendizado.png)
