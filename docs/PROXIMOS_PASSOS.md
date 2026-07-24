# Próximos passos — planejamento vivo do TrafficRL

Atualizado conforme avançamos. Legenda: ✅ feito · 🔄 em andamento · ⏳ próximo · 🔮 futuro.

## Onde estamos (o que já está pronto)

- ✅ **Fase 1** — cruzamento único: DQN vence os 4 baselines (p<0,0001), o mais justo.
- ✅ **Generalista** — treino multi-cenário consertou a especialização (−74% no fora-pico).
- ✅ **Fase 2** — grid 3×3: DQN bate os clássicos fortes; empata com o tempo fixo.
- ✅ **Fase 3 (implementada)** — coordenação entre cruzamentos, LIGADA por padrão.
- ✅ **Fase 4 (MVP)** — pipeline OSM pronto (falta baixar recorte real de SP).
- ✅ **Ferramentas** — designer de grid, comando `grid-run`, apresentação, relatório final.

## Horizonte 1 — agora (nuvem, automático)

- 🔄 **Treinar 3 seeds coordenadas** (42✅, 123, 7) em paralelo, retomável.
- ⏳ **Comparação coordenado × não-coordenado × clássicos** → gráficos + explicação.
  - **Ponto de decisão:** a coordenação faz a IA vencer o tempo fixo na malha?
    - Se **sim** → é o fecho do arco; atualizar apresentação/relatório.
    - Se **não/parcial** → diagnóstico honesto e ir ao Horizonte 2.
- ⏳ Treinar as 2 seeds restantes (2024, 777) para rigor estatístico completo.
- ⏳ Regerar o mapa de decisão da rede treinada (mostrar a coordenação "aparecendo").

## Horizonte 2 — melhorar a IA (nuvem + local)

- 🔮 **Qualidade máxima no 3×3:** subir treino de 600k → 1M passos (receita em
  PLANO_TREINO.md). Máquina local rende mais.
- 🔮 **Observação mais rica no grid:** dar densidade/velocidade além da fila.
- 🔮 **Coordenação de mais longe:** ver 2 quarteirões à frente, não só o vizinho
  imediato — ajuda a onda verde em malhas grandes.
- 🔮 **Seu 5×5 completo:** treino proporcional (~1,5M) na máquina local.

## Horizonte 3 — São Paulo real (máquina local)

- 🔮 **Baixar recorte OSM** de um bairro (ex.: Paulista/Pinheiros) — 1 comando.
- 🔮 **Rodar o pipeline Fase 4** na malha real e ver quantos cruzamentos a IA controla.
- 🔮 **Demanda real:** matriz Origem-Destino do Metrô + contagens da CET no lugar
  do tráfego provisório.

## Horizonte 4 — pesquisa e portfólio

- 🔮 **"Maestro" hierárquico:** um coordenador de estratégia por corredor (a sua
  ideia do cérebro, sem a explosão de combinações).
- 🔮 **Treino centralizado / execução distribuída:** crítico global no treino.
- 🔮 **Material final:** TCC/artigo com todas as fases; apresentação atualizada.
- 🔮 **Limitações e sim-to-real:** documentar honestamente o caminho até um piloto.

## Regras que não mudam (a disciplina do projeto)

1. Nunca maquiar texto: se a IA é fraca em algo, **treinar**, não reescrever.
2. Comparação limpa: mudar uma coisa por vez (ex.: só a coordenação).
3. Sempre contra os baselines fortes (atuado, max-pressure), não só o burro.
4. Reportar o resultado honesto — vença ou perca.
5. Tudo reprodutível: config, git hash, seeds, checkpoints versionados.
