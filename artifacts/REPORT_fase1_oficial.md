# TrafficRL — Relatório de avaliação

- Gerado em: 2026-07-20T21:37:18
- Git hash: `e85b8080e7cbd7260c316d6681f0430ecefeb9f2`
- Episódios por método × cenário (máx.): 100
- Valores: média [IC 95% via bootstrap]. Testes: Mann-Whitney U bicaudal, correção de Bonferroni por cenário.

## Cenário: balanceado

| Método | Espera média (s) | Espera p95 (s) | Espera máxima (s) | Fila máxima (veíc.) | Throughput (veíc.) | Tempo de viagem médio (s) |
|---|---|---|---|---|---|---|
| Fixo igual | 152.0 [149.5, 154.5] | 572.2 [546.0, 597.1] | 783.5 [737.4, 827.9] | 89.2 [88.5, 90.1] | 1352.4 [1331.3, 1374.2] | 230.7 [227.4, 233.9] |
| Fixo proporcional | 147.3 [143.6, 150.8] | 586.0 [555.0, 616.0] | 849.5 [796.6, 901.8] | 88.7 [87.7, 90.0] | 1374.6 [1349.7, 1401.1] | 220.0 [215.7, 224.1] |
| Atuado (gap) | 215.0 [210.9, 219.1] | 1045.4 [990.9, 1102.7] | 1518.0 [1411.2, 1634.5] | 136.5 [133.7, 139.3] | 1191.9 [1169.0, 1215.1] | 271.0 [266.5, 276.0] |
| Max-pressure | 672.2 [658.4, 686.1] | 3146.3 [3083.9, 3207.9] | 3554.2 [3545.2, 3562.6] | 256.2 [255.9, 256.6] | 795.5 [780.0, 811.1] | 408.5 [400.2, 416.6] |
| DQN (nosso) | 217.5 [209.4, 225.2] | 1221.8 [1009.2, 1440.5] | 1461.8 [1245.8, 1686.9] | 148.2 [138.5, 157.6] | 1285.9 [1253.2, 1318.2] | 253.2 [229.1, 277.1] |

**Interpretação (balanceado):** com correção de Bonferroni, não houve diferença significativa frente ao Atuado (gap) (p corrigido = 0.7010); o Fixo igual superou o DQN (43% de vantagem, p corrigido = 0.0000) — resultado reportado honestamente; o Fixo proporcional superou o DQN (48% de vantagem, p corrigido = 0.0000) — resultado reportado honestamente; o DQN reduziu a espera média em 68% frente ao Max-pressure (p corrigido = 0.0000). Lembrete metodológico: o mérito real do agente é medido contra os baselines fortes (atuado por gap e max-pressure), não apenas contra o tempo fixo.

## Cenário: fora_pico

| Método | Espera média (s) | Espera p95 (s) | Espera máxima (s) | Fila máxima (veíc.) | Throughput (veíc.) | Tempo de viagem médio (s) |
|---|---|---|---|---|---|---|
| Fixo igual | 52.3 [48.6, 57.0] | 157.2 [135.7, 182.9] | 339.1 [306.7, 374.2] | 41.5 [39.2, 44.2] | 1055.2 [1042.7, 1067.6] | 106.3 [101.8, 111.9] |
| Fixo proporcional | 53.8 [48.2, 61.0] | 230.8 [197.2, 270.9] | 440.7 [395.1, 490.9] | 37.1 [33.7, 41.1] | 1052.9 [1039.8, 1066.2] | 105.6 [99.7, 113.4] |
| Atuado (gap) | 147.8 [136.8, 158.8] | 661.4 [583.0, 739.1] | 1012.6 [924.1, 1106.6] | 99.8 [94.7, 105.0] | 998.8 [979.8, 1017.3] | 195.0 [184.2, 205.4] |
| Max-pressure | 489.3 [443.5, 535.4] | 1908.6 [1750.7, 2067.4] | 3276.1 [3104.4, 3430.8] | 244.8 [237.7, 250.7] | 759.3 [730.3, 788.8] | 401.6 [371.2, 432.4] |
| DQN (nosso) | 163.8 [154.5, 173.9] | 919.6 [794.2, 1058.1] | 1353.6 [1203.8, 1516.0] | 95.3 [90.4, 100.7] | 970.9 [955.4, 985.3] | 187.3 [176.4, 197.9] |

**Interpretação (fora_pico):** com correção de Bonferroni, não houve diferença significativa frente ao Atuado (gap) (p corrigido = 1.0000); o Fixo igual superou o DQN (213% de vantagem, p corrigido = 0.0000) — resultado reportado honestamente; o Fixo proporcional superou o DQN (204% de vantagem, p corrigido = 0.0000) — resultado reportado honestamente; o DQN reduziu a espera média em 67% frente ao Max-pressure (p corrigido = 0.0000). Lembrete metodológico: o mérito real do agente é medido contra os baselines fortes (atuado por gap e max-pressure), não apenas contra o tempo fixo.

## Cenário: pico_assimetrico

| Método | Espera média (s) | Espera p95 (s) | Espera máxima (s) | Fila máxima (veíc.) | Throughput (veíc.) | Tempo de viagem médio (s) |
|---|---|---|---|---|---|---|
| Fixo igual | 246.0 [243.1, 249.0] | 489.8 [472.5, 506.2] | 773.4 [725.6, 824.6] | 205.8 [204.0, 207.7] | 1844.3 [1826.8, 1861.1] | 360.1 [356.3, 364.1] |
| Fixo proporcional | 110.1 [106.5, 113.6] | 603.0 [590.0, 613.4] | 949.3 [898.3, 1000.4] | 127.7 [122.5, 132.3] | 2590.3 [2568.5, 2609.9] | 164.1 [160.3, 167.9] |
| Atuado (gap) | 265.3 [262.7, 267.8] | 825.2 [782.4, 862.6] | 1623.0 [1499.4, 1755.2] | 237.2 [234.6, 240.0] | 1901.3 [1883.1, 1923.4] | 342.1 [339.4, 345.0] |
| Max-pressure | 266.5 [262.6, 270.0] | 3110.0 [3086.6, 3131.1] | 3565.9 [3559.7, 3571.4] | 216.8 [213.1, 220.4] | 1920.0 [1900.9, 1939.3] | 180.5 [177.5, 183.4] |
| DQN (nosso) | 174.3 [170.4, 178.1] | 437.5 [417.8, 458.8] | 1114.5 [949.3, 1291.0] | 214.8 [212.6, 216.8] | 2308.3 [2281.5, 2334.7] | 278.7 [267.5, 289.1] |

**Interpretação (pico_assimetrico):** com correção de Bonferroni, o DQN reduziu a espera média em 34% frente ao Atuado (gap) (p corrigido = 0.0000); o DQN reduziu a espera média em 29% frente ao Fixo igual (p corrigido = 0.0000); o Fixo proporcional superou o DQN (58% de vantagem, p corrigido = 0.0000) — resultado reportado honestamente; o DQN reduziu a espera média em 35% frente ao Max-pressure (p corrigido = 0.0000). Lembrete metodológico: o mérito real do agente é medido contra os baselines fortes (atuado por gap e max-pressure), não apenas contra o tempo fixo.

## Cenário: pico_invertido

| Método | Espera média (s) | Espera p95 (s) | Espera máxima (s) | Fila máxima (veíc.) | Throughput (veíc.) | Tempo de viagem médio (s) |
|---|---|---|---|---|---|---|
| Fixo igual | 145.9 [144.0, 148.0] | 552.7 [532.5, 575.1] | 767.2 [729.7, 806.0] | 84.8 [83.4, 86.5] | 1335.8 [1322.2, 1347.9] | 225.5 [223.1, 228.4] |
| Fixo proporcional | 142.3 [139.4, 145.4] | 547.5 [521.4, 573.9] | 818.0 [770.6, 867.5] | 91.2 [89.2, 93.0] | 1369.3 [1350.2, 1385.7] | 216.8 [213.4, 220.3] |
| Atuado (gap) | 202.9 [198.3, 207.3] | 935.7 [893.0, 975.4] | 1507.0 [1417.3, 1603.8] | 128.4 [126.2, 130.8] | 1188.7 [1172.0, 1206.0] | 260.9 [255.8, 266.4] |
| Max-pressure | 622.4 [605.8, 638.7] | 3223.8 [3135.3, 3294.2] | 3557.8 [3549.7, 3564.9] | 232.1 [229.2, 234.9] | 796.9 [780.2, 813.2] | 382.3 [373.0, 391.2] |
| DQN (nosso) | 210.3 [201.3, 219.3] | 786.4 [659.7, 931.2] | 1050.0 [912.4, 1201.3] | 154.1 [145.8, 162.4] | 1308.9 [1286.2, 1330.8] | 294.7 [276.1, 312.8] |

**Interpretação (pico_invertido):** com correção de Bonferroni, não houve diferença significativa frente ao Atuado (gap) (p corrigido = 1.0000); o Fixo igual superou o DQN (44% de vantagem, p corrigido = 0.0000) — resultado reportado honestamente; o Fixo proporcional superou o DQN (48% de vantagem, p corrigido = 0.0000) — resultado reportado honestamente; o DQN reduziu a espera média em 66% frente ao Max-pressure (p corrigido = 0.0000). Lembrete metodológico: o mérito real do agente é medido contra os baselines fortes (atuado por gap e max-pressure), não apenas contra o tempo fixo.

## Testes estatísticos — DQN vs baselines (espera média)

| Cenário | Baseline | p (bruto) | p (Bonferroni) | Significativo (α=0,05) | DQN melhor? |
|---|---|---|---|---|---|
| balanceado | Atuado (gap) | 0.1752 | 0.7010 | não | não |
| balanceado | Fixo igual | 0.0000 | 0.0000 | sim | não |
| balanceado | Fixo proporcional | 0.0000 | 0.0000 | sim | não |
| balanceado | Max-pressure | 0.0000 | 0.0000 | sim | sim |
| fora_pico | Atuado (gap) | 0.2704 | 1.0000 | não | não |
| fora_pico | Fixo igual | 0.0000 | 0.0000 | sim | não |
| fora_pico | Fixo proporcional | 0.0000 | 0.0000 | sim | não |
| fora_pico | Max-pressure | 0.0000 | 0.0000 | sim | sim |
| pico_assimetrico | Atuado (gap) | 0.0000 | 0.0000 | sim | sim |
| pico_assimetrico | Fixo igual | 0.0000 | 0.0000 | sim | sim |
| pico_assimetrico | Fixo proporcional | 0.0000 | 0.0000 | sim | não |
| pico_assimetrico | Max-pressure | 0.0000 | 0.0000 | sim | sim |
| pico_invertido | Atuado (gap) | 0.3329 | 1.0000 | não | não |
| pico_invertido | Fixo igual | 0.0000 | 0.0000 | sim | não |
| pico_invertido | Fixo proporcional | 0.0000 | 0.0000 | sim | não |
| pico_invertido | Max-pressure | 0.0000 | 0.0000 | sim | sim |

## Figuras

![barras_espera_media_s](figures/barras_espera_media_s.png)
![boxplot_p95_espera](figures/boxplot_p95_espera.png)
![justica_avenida_vs_local_balanceado](figures/justica_avenida_vs_local_balanceado.png)
![justica_avenida_vs_local_fora_pico](figures/justica_avenida_vs_local_fora_pico.png)
![justica_avenida_vs_local_pico_assimetrico](figures/justica_avenida_vs_local_pico_assimetrico.png)
![justica_avenida_vs_local_pico_invertido](figures/justica_avenida_vs_local_pico_invertido.png)
![curvas_aprendizado](figures/curvas_aprendizado.png)
