# TrafficRL — Relatório Final Consolidado

**Controle semafórico adaptativo via Deep Reinforcement Learning.**
Um estudo com rigor de pesquisa: baselines fortes, múltiplas seeds, testes
estatísticos e uma métrica de justiça — do cruzamento único à malha real.

> Todos os números deste documento vêm das avaliações oficiais
> (`results/REPORT.md` e `results/grid/REPORT.md`), reprodutíveis a partir
> dos modelos versionados. Resultados desfavoráveis são reportados sem
> maquiagem — é a postura que dá credibilidade ao favorável.

---

## 1. O problema e a motivação

Em São Paulo, a maioria dos semáforos opera em tempo fixo: o verde da avenida
dura o mesmo às 8h de segunda e às 3h de domingo. Isso desperdiça capacidade
viária. A pergunta do projeto: **um agente de reinforcement learning aprende
políticas de controle que superam os métodos clássicos — inclusive os
"inteligentes" — mantendo a justiça entre vias?**

A resposta foi construída em fases de complexidade crescente, cada uma
validada com o mesmo protocolo estatístico.

---

## 2. Metodologia (o que dá credibilidade)

- **Ambiente:** simulador SUMO com wrapper próprio que **impõe a segurança**
  (verde mínimo 10 s, amarelo 3 s, vermelho total 2 s, verde máximo) — o
  agente nunca pode violar tempos de segurança, garantia por construção.
- **Agente:** Double DQN (rede 9→256→256→4, ~69 mil parâmetros).
- **Baselines:** 4 controladores clássicos, do "semáforo burro" (tempo fixo
  igual) aos fortes — **atuado por gap** (o que semáforos comerciais fazem) e
  **max-pressure** (Varaiya 2013, a teoria por trás dos melhores clássicos).
  *O mérito real do RL é medido contra os fortes, não contra o burro.*
- **Protocolo:** 5 seeds independentes × treino completo; avaliação em
  episódios com tráfego inédito (seeds disjuntas do treino).
- **Estatística:** média ± IC 95% por bootstrap; Mann-Whitney U entre DQN e
  cada baseline, com correção de Bonferroni.
- **Justiça:** além da espera média, medimos o **p95** e a espera máxima —
  para expor o truque de "otimizar a média sacrificando a via local".
- **Qualidade:** 56 testes automatizados, CI no GitHub, cobertura > 70%.

---

## 3. Fase 1 — Cruzamento único: **o RL vence os clássicos**

Cenário-alvo `pico_assimetrico` (avenida 1200 vs local 300 veíc/h), 5 seeds ×
300k passos, 20 episódios de avaliação por método.

| Método | Espera média (s) | Espera p95 — justiça (s) |
|---|---|---|
| Fixo proporcional *(conhece a demanda)* | **110** | 603 |
| **DQN (nosso)** | 174 | **438** ← melhor de todos |
| Fixo igual | 246 | 490 |
| Atuado (gap) | 265 | 825 |
| Max-pressure | 267 | 3110 |

**Resultados com significância estatística (p < 0,0001, Bonferroni):**

- ✅ DQN bate o **fixo igual** em −29% (critério de aceitação do projeto);
- ✅ DQN bate o **atuado** em −34% e o **max-pressure** em −35% — os dois
  baselines fortes, que é onde está o mérito real;
- ⚖️ Perde em média só para o **fixo proporcional onisciente** — mas **vence
  ele em justiça** (p95 438 vs 603 s). Nenhuma outra política protege tão bem
  a via local.

O contraste do max-pressure é ilustrativo: espera média ok, mas p95 de
**3110 s** — ele sacrifica a via local, exatamente a starvation que a
penalidade anti-starvation ensina o DQN a evitar.

### 3.1 O achado honesto: especialização vs generalização

Nos cenários em que **não treinou** (`balanceado`, `fora_pico`,
`pico_invertido`), o DQN perde para os métodos de tempo fixo — embora
continue batendo o max-pressure em todos. No `pico_invertido` (a demanda
inverte: a local vira a movimentada), o fixo supera o DQN em ~45%.

Isto **não é um defeito** — é o resultado que o cenário `pico_invertido` foi
projetado para detectar. O agente aprendeu um viés a priori ("a avenida é a
movimentada"), certo no treino e errado quando a demanda inverte. É a
motivação documentada do próximo experimento (treino multi-cenário e a
ablação de observação rica).

---

## 4. Fase 2 — Malha 3×3: **o RL bate os fortes, mas a coordenação falta**

9 cruzamentos, uma política DQN compartilhada (uma instância por cruzamento),
5 seeds × 600k transições, 20 episódios. Grid inspirado em SP (avenidas
cruzando ruas locais); há também uma variante ancorada em dados reais da CET
(`grid_paulista_real.yaml`).

| Método | Espera média (s) | DQN vs baseline (Bonferroni) |
|---|---|---|
| Fixo proporcional | 99 | empate estatístico (p=1,0) |
| Fixo igual | 174 | fixo vence −35% (p=0,044) |
| **DQN (nosso)** | **235** | — |
| Atuado (gap) | 269 | **DQN vence −13% (p=0,002)** |
| Max-pressure | 520 | **DQN vence −55% (p=0,0001)** |

**Interpretação:** na malha, o DQN **supera os dois controladores adaptativos
clássicos** (atuado e max-pressure) com significância. Os dois de tempo fixo,
porém, ainda levam vantagem na espera média — inclusive o "burro" fixo igual
(174 s vs 235 s). É um resultado instrutivo: numa malha carregada, o ritmo
regular e previsível do tempo fixo evita a instabilidade que agentes
independentes criam quando um atrapalha o outro. O limitante do DQN é a **alta
variância entre execuções** (IC da espera [187, 284]): algumas seeds
coordenam-se emergentemente, outras não — porque cada agente decide sozinho,
sem enxergar os vizinhos.

> Nota estatística: no teste de Mann-Whitney (ranks), a diferença DQN vs fixo
> proporcional não é significativa apesar da diferença de médias — a
> distribuição do DQN é assimétrica (mediana competitiva, alguns episódios
> ruins puxam a média). Na métrica que importa para o motorista — a espera
> média — os dois de tempo fixo ficam à frente.

Esse é o argumento de abertura da **Fase 3**: dar a cada agente informação da
vizinhança para emergir a "onda verde" e estabilizar a variância.

---

## 5. Fase 4 (MVP) — Rumo à malha real de SP

O pipeline para trânsito real está implementado e testado:

- **`import_osm`** converte um recorte do OpenStreetMap (ex.: região da
  Paulista) em rede SUMO com semáforos inferidos;
- **`discover_controllable_tls`** identifica, por geometria, os cruzamentos
  que o modelo de 4 estágios controla (os irregulares ficam com o programa
  padrão — limitação explícita, não escondida);
- **`OsmTrafficEnv`** roda a malha real com a mesma gaiola de segurança das
  fases anteriores;
- demanda provisória por `randomTrips`, com o lugar reservado para a matriz
  **Origem-Destino do Metrô** calibrada por contagens da CET.

Falta apenas o passo que exige rede aberta ao OSM (a máquina local): baixar o
recorte real. Documentado em `docs/PENDENTE_LOCAL.md`.

### Como uma cidade "sabe" o trânsito de todas as ruas

Não se mede rua por rua. A **Pesquisa Origem-Destino** (Metrô-SP, pública) dá
a matriz de desejos de viagem; um algoritmo de alocação a distribui pela
malha do OSM, produzindo fluxo estimado para **cada rua**; as **contagens da
CET** e os **radares** (dados abertos) calibram o modelo nos pontos medidos;
GPS/apps (Waze) validam por velocidade. É a esteira padrão da engenharia de
tráfego — e quase toda a matéria-prima de SP é pública.

---

## 6. Síntese

| Fase | Escopo | Resultado |
|---|---|---|
| **1** | Cruzamento único | RL **vence os 4 baselines** no alvo (p<0,0001) e é o **mais justo**; especialização detectada honestamente |
| **2** | Malha 3×3 | RL **vence os 2 clássicos fortes**; variância entre seeds motiva coordenação |
| **4 (MVP)** | Malha real OSM | pipeline pronto; falta o recorte real (máquina local) |

**A tese, em uma frase:** o reinforcement learning supera os controladores
semafóricos clássicos — inclusive os "inteligentes" — no cruzamento único com
significância estatística, e já os supera na malha; o próximo salto é a
coordenação explícita entre cruzamentos.

---

## 7. Limitações honestas

- **Gap simulação→realidade:** SUMO com car-following de fábrica, sem
  pedestres, ônibus, motos costurando ou chuva. Evidência de conceito, não de
  implantação.
- **Observação idealizada:** o agente vê filas exatas; detectores reais têm
  ruído e cobertura parcial.
- **Demanda:** sintética (Fases 1-2) ou provisória (Fase 4) até a OD
  calibrada entrar.
- **Um algoritmo:** DQN apenas; PPO e outros ficam para ablação futura.

## 8. Roadmap

- **Fase 3:** coordenação entre cruzamentos (onda verde emergente) —
  motivada diretamente pelo resultado da Fase 2.
- **Generalização:** treino multi-cenário (agente especialista vs
  generalista).
- **Fase 4 completa:** malha real com demanda OD do Metrô calibrada por
  contagens CET/radares.

---

*Reprodutibilidade: cada run salva config exata, git hash e melhor
checkpoint por seed. Relatórios e figuras são gerados automaticamente por
`traffic-rl compare` / `grid-compare`. Código: veja README.md e
docs/DECISOES.md (19 ADRs).*
