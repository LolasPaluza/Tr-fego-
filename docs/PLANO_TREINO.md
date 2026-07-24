# Plano de treino — otimizar tempo e maximizar a qualidade da IA

Objetivo: fazer a IA coordenada (Fase 3) melhorar o máximo possível, gastando
o mínimo de tempo de relógio. Abaixo, as alavancas — o que muda o resultado e
o que muda o tempo — e a receita concreta para a nuvem e para a máquina local.

## As alavancas (o que realmente importa)

| Alavanca | Efeito na QUALIDADE | Efeito no TEMPO |
|---|---|---|
| Mais passos de treino por seed | ↑↑ (a maior) | ↑↑ custo linear |
| Mais seeds | ↑ confiança estatística | ↑ custo linear |
| Treinar seeds em PARALELO | = (não muda) | ↓↓ ~2–3× mais rápido |
| Retomar de checkpoint | = | evita perder trabalho |
| Ajustar hiperparâmetros | ↑ incerto, arriscado | varia |

**Conclusão:** para MELHORAR mais, a alavanca honesta é **mais passos**. Para
ir mais RÁPIDO sem perder qualidade, é **paralelismo** + **retomada robusta**.
Mexer em hiperparâmetro é aposta — mantemos os que já funcionam para a
comparação coordenado × não-coordenado ficar limpa (só a coordenação muda).

## Estratégia (nuvem — o que rodamos aqui, apesar dos reinícios)

1. **Paralelismo 2×:** os 4 núcleos rodam 2 seeds ao mesmo tempo (cada treino
   usa ~1,5 núcleo: SUMO + rede pequena). Corta o tempo de relógio ~2×.
2. **Foco em 3 seeds primeiro** (42, 123, 7): já dá uma comparação credível.
   As outras 2 (2024, 777) entram depois, para rigor total.
3. **Mesmo orçamento do não-coordenado (600k por seed):** a comparação
   "coordenação ajuda?" só é justa se a única diferença for a coordenação.
4. **Retomável + protegido:** cada seed retoma do checkpoint; o guardião
   salva no GitHub a cada 10 min. Sobrevive aos reinícios do sandbox.

Comando: `bash scripts/run_coord_parallel.sh`

## Receita "qualidade máxima" (máquina local — CPU de verdade, sem reinícios)

Na sua máquina, sem os limites do sandbox, vale investir mais:

1. **3×3 no talo:** suba `total_timesteps` para **1.000.000** em
   `configs/grid_coord.yaml` e rode as 5 seeds. Mais passos = melhor pico.
   ```bash
   bash scripts/run_coord_parallel.sh          # usa os núcleos que tiver
   traffic-rl grid-compare --grid configs/grid_coord.yaml
   ```
2. **Seu 5×5 (25 cruzamentos):** precisa de orçamento proporcional —
   suba `total_timesteps` para **~1.500.000** (são 25 cruzamentos dividindo a
   experiência). Rode com o `grid-run`:
   ```bash
   traffic-rl grid-run --grid configs/meu_grid_5x5.yaml
   ```
3. **Monitore:** `tensorboard --logdir results/runs` — veja a curva subir.

## Estimativas honestas (8 núcleos, sem GPU)

- 3×3 coordenado, 600k × 5 seeds, paralelo: ~3–5 h.
- 3×3 coordenado, 1M × 5 seeds (qualidade máxima): ~6–8 h (deixe de noite).
- 5×5, 1,5M × 5 seeds: ~1–1,5 dia (é grande — rode no fim de semana).

## Onde a qualidade pode subir mais (pesquisa futura)

- **Observação mais rica:** dar ao agente densidade/velocidade além da fila.
- **Coordenação de mais longe:** hoje cada um vê só o vizinho imediato; ver
  2 quarteirões à frente ajuda a onda verde em malhas grandes.
- **"Maestro" hierárquico:** um coordenador de estratégia por corredor.
- **Treino centralizado, execução distribuída:** um crítico global no treino.
