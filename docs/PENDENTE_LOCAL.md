# Pendências para a máquina local

O sandbox cloud não tem GUI nem tempo de computação para runs completos.
Este arquivo lista o que só pode ser feito/verificado localmente.

## Runs completos (comandos no README §Rodando na máquina local)

1. `python scripts/validate_setup.py` — deve terminar 100% ✅
2. `traffic-rl train --smoke && traffic-rl compare --smoke` — sem erro
3. `bash scripts/run_full_training.sh` (~2 h paralelo / ~5 h sequencial)
4. `bash scripts/run_full_evaluation.sh` (~2 h) → `results/REPORT.md`
5. `bash scripts/run_ablations.sh` (~24–30 h, retomável)
6. Critério de aceitação nº 4: conferir no REPORT.md que o DQN bate
   `fixo_igual` no `pico_assimetrico` com p (Bonferroni) < 0,05; reportar
   honestamente o resultado contra `atuado_gap` e `max_pressure`.

## Fase 2 — grid (comandos no README §Fase 2)

1. Edite `configs/grid.yaml` com as SUAS ruas (nomes, classes, fluxos) —
   ou crie um YAML próprio e passe com `--grid`.
2. `traffic-rl grid-train --smoke && traffic-rl grid-compare --smoke` (~2 min)
3. `bash scripts/run_grid_full.sh` (~45–60 min/seed de treino + ~1 h de
   comparação) → `results/grid/REPORT.md`
4. Visual: `netedit data/generated/grid.net.xml` — conferir a malha, faixas
   por classe de rua e conexões de conversão em cada cruzamento.

## Verificações visuais (exigem sumo-gui/netedit)

Instale a GUI localmente (`pip install eclipse-sumo` já a inclui como
`sumo-gui`; no apt: `sudo apt install sumo-gui`).

1. **Geometria da rede** — abrir a rede gerada:
   `netedit data/generated/intersection.net.xml`
   - avenida L-O com 3 faixas/sentido e faixa da esquerda SÓ com seta de
     conversão; via local N-S com 1 faixa/sentido;
   - conexões: sem retornos (U-turn); esquerda da via local entrando na
     faixa mais à esquerda da avenida.
2. **Programa semafórico em ação** — rodar um episódio com GUI:
   `sumo-gui -n data/generated/intersection.net.xml -r data/generated/pico_assimetrico.rou.xml`
   (dar Play; o programa default do netconvert aparece — para ver os 4
   estágios do projeto é preciso rodar via ambiente, ex.: um episódio do
   `traffic-rl evaluate` com `use_libsumo: false` e sumo-gui trocado no
   código, OU conferir os estados RYG impressos por
   `python -c "from traffic_rl.envs.phases import build_phase_table;
   print(build_phase_table('data/generated/intersection.net.xml'))"`).
   - checar: amarelo de 3 s e vermelho total de 2 s em TODA transição;
     nunca dois estágios conflitantes verdes ao mesmo tempo.
3. **Comportamento de fila** — no cenário `pico_assimetrico`, observar se a
   fila da avenida cresce de forma realista (sem teleports — estão
   desligados) e se caminhões (10%) aparecem maiores/mais lentos.

## Qualidade de vida (opcional)

- `pip install torch --index-url https://download.pytorch.org/whl/cpu` antes
  de `pip install -e ".[dev]"` para economizar ~4 GB de disco (ADR-012).
- `tensorboard --logdir results/runs` durante o treino para acompanhar
  `rollout/ep_rew_mean` e `eval/mean_reward`.
