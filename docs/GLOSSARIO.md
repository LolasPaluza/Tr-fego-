# Glossário do TrafficRL — os conceitos, explicados com o nosso projeto

Todo termo aqui aparece com um exemplo real do projeto. A ideia central que
amarra tudo: **a IA joga um videogame de semáforo milhões de vezes e aprende
com o placar.**

---

## Nível 1 — O jogo

### Episódio
Uma "partida" completa. No nosso caso: **1 hora de trânsito simulado**
(3600 segundos). Começa com as ruas vazias, os carros vão chegando, e no fim
medimos o estrago (quanto todo mundo esperou).

> No projeto: cada episódio = 3600 s de simulação. A avaliação usa 20
> episódios por método para a média não depender de sorte.

### Passo (ou *timestep*)
Uma decisão. A cada **5 segundos** de simulação, o semáforo decide quem ganha
o verde. Cada uma dessas decisões é um passo.

> 1 episódio de 3600 s ÷ 5 s = **720 passos**. Quando dizemos "treinar 600 mil
> passos", são ~830 episódios de experiência.

### Observação
O que a IA **vê** antes de decidir. Não é uma câmera — são números.

> No projeto (9 números): qual fase está verde (4), há quanto tempo (1), e o
> tamanho da fila em cada uma das 4 direções (4). Com coordenação, viram 13
> (mais 4: o quão cheio está cada vizinho).

### Ação
O que a IA **faz**. Aqui: escolher qual das 4 fases recebe o verde
(avenida em frente, avenida à esquerda, local em frente, local à esquerda).

### Recompensa
O **placar**. Depois de cada ação, a IA ganha ou perde pontos:
- espera total **diminuiu** → pontos positivos ✅
- espera **aumentou** → pontos negativos ❌
- alguém esperando > 2 min → **multa** extra (anti-starvation)

> É só isso que a IA quer maximizar. Ela nunca "entende" trânsito — ela
> persegue pontos, e a boa engenharia da recompensa faz com que perseguir
> pontos = melhorar o trânsito.

---

## Nível 2 — O aprendizado

### Política
A "personalidade" da IA: a regra que mapeia **observação → ação**. É o que a
rede neural representa. Treinar = melhorar a política.

### Valor Q (o "Q" de DQN)
Para cada ação possível, a nota que a IA dá: *"se eu escolher isso agora,
quão bom será o futuro?"*. A IA executa a ação de maior nota.

> Crucial: a nota inclui o **futuro**, não só o efeito imediato. Segurar o
> verde da avenida pode parecer bom agora e cobrar caro dois ciclos depois.

### DQN (Deep Q-Network)
"Rede neural profunda que estima os valores Q". É o nosso algoritmo.
**Double DQN** é uma correção conhecida que evita a IA ficar otimista demais
com as próprias notas (usa duas redes: uma escolhe, outra avalia).

### Rede neural / camadas / parâmetros
A "máquina de dar notas". A nossa:
`13 entradas → 256 neurônios → 256 neurônios → 4 saídas`, com **69.380
parâmetros** (os números ajustáveis). É minúscula para padrões de IA — cabe
num e-mail e roda em qualquer computador.

### Treino
Repetir milhões de vezes: observar → agir → receber pontos → **ajustar um
tiquinho os 69 mil parâmetros** na direção que teria dado mais pontos.

### Exploração vs aproveitamento (*epsilon*)
No começo, a IA age **aleatoriamente** para descobrir o mundo (exploração).
Com o tempo, passa a usar o que aprendeu (aproveitamento). O `epsilon` é a
chance de agir ao acaso: começa em 100% e cai para 2%.

> É por isso que a curva de aprendizado **zigue-zagueia**: às vezes a IA
> testa algo ruim de propósito. Faz parte.

### Replay buffer (memória de experiências)
Uma "caderneta" com as últimas 100 mil situações vividas. A IA revisa
situações antigas ao acaso em vez de aprender só com a última — isso
estabiliza o aprendizado.

### Curva de aprendizado
O gráfico da nota ao longo do treino. **Subindo = aprendendo.**

> No nosso: a rede coordenada saiu de −4608 e chegou a −3228 (nota negativa
> porque é o negativo da espera; mais alto = melhor).

---

## Nível 3 — O rigor (o que separa ciência de achismo)

### Seed (semente)
Um número que fixa toda a aleatoriedade. Mesma seed = **exatamente** o mesmo
experimento, sempre. Serve para duas coisas:

1. **Reprodutibilidade**: qualquer pessoa roda e obtém o mesmo resultado.
2. **Honestidade**: treinar com 5 seeds diferentes (42, 123, 7, 2024, 777) e
   reportar a média — porque **uma seed sortuda pode parecer um avanço**.

> Se alguém te mostra um resultado de RL com uma seed só, desconfie.

### Baseline
O adversário. Sem ele, "reduziu 30%" não significa nada — reduziu comparado a
quê? Nossos 4:
- `fixo_igual` — o semáforo burro (tempos iguais)
- `fixo_proporcional` — plano bem calibrado (é o mais parecido com a CET)
- `atuado_gap` — o "inteligente" comercial (estende o verde com sensor)
- `max_pressure` — o melhor da teoria clássica (Varaiya, 2013)

> Regra do projeto: mérito real é bater os **fortes** (atuado, max-pressure),
> não o burro.

### p-valor e significância
A probabilidade de a diferença observada ser **acaso**. p < 0,05 = "menos de
5% de chance de ser sorte" → consideramos real.

> No projeto: DQN vs max-pressure na Fase 1 deu **p < 0,0001** — praticamente
> impossível ser coincidência.

### Correção de Bonferroni
Se você faz muitos testes, algum dá "significativo" por acaso. Bonferroni
torna o critério mais exigente proporcionalmente ao número de comparações.
É ser rigoroso consigo mesmo.

### Média vs p95 (a métrica de justiça)
- **Média**: o tempo de espera típico.
- **p95**: o tempo dos 5% que mais sofreram.

> Por que importa: um método pode ter média ótima **abandonando** a via local.
> O max-pressure tinha média parecida com a nossa IA mas p95 de **3110 s** —
> gente presa quase 1 hora. Nossa IA: **438 s**, a melhor de todas.

### Intervalo de confiança (IC 95%) e bootstrap
A margem de erro da média. "102 s [95, 110]" = a média verdadeira está
provavelmente nessa faixa. *Bootstrap* é a técnica que calcula isso
reembaralhando os dados, sem supor que eles seguem uma curva normal.

### Especialização vs generalização (*overfitting*)
- **Especialista**: decorou o cenário do treino, quebra fora dele.
- **Generalista**: aprendeu a regra, funciona em situações novas.

> Foi o nosso maior experimento: o especialista (treinado só no pico) falhava
> quando a demanda invertia. Treinamos com os 4 cenários misturados
> (*domain randomization*) e a espera caiu **74%** no fora-pico.

---

## Nível 4 — A engenharia

### SUMO
O simulador de trânsito (Eclipse SUMO). É onde o "videogame" acontece: ele
move os carros com física realista de aceleração, frenagem, filas.

### Checkpoint
Um "save" do treino. Se o computador desliga no meio, retoma dali em vez de
recomeçar. Salvamos a cada 25 mil passos — foi o que nos salvou nos dezenas
de reinícios do servidor.

### Gaiola de segurança (o wrapper)
As regras que a IA **não pode** violar, impostas por código em volta dela:
verde mínimo de 10 s, amarelo de 3 s, vermelho geral de 2 s, verde máximo.

> Isso não é aprendido — é garantido. É a arquitetura que se usaria de
> verdade: a IA sugere, o sistema clássico garante a segurança.

### Coordenação (Fase 3)
Cada semáforo passa a ver o quão cheios estão os **4 vizinhos**. Permite
antecipar a onda de carros e criar a "onda verde" — sem um chefe central.

> Por que não um cérebro central? Porque escolher a ação de 25 cruzamentos
> ao mesmo tempo dá 4²⁵ ≈ **1 quatrilhão** de combinações. Não cabe em rede
> neural nenhuma, e vira ponto único de falha.

### Ablação
Tirar/trocar **uma peça** e medir o efeito, para saber o que realmente
importa. Ex.: mesma IA com e sem coordenação → a diferença é o valor da
coordenação.

> Regra de ouro: mude **uma coisa por vez**. Se mudar cinco, não saberá qual
> causou o resultado.

---

## O ciclo completo, em uma frase

**A IA observa** as filas (observação), **decide** quem ganha o verde (ação),
**recebe pontos** conforme a espera cai (recompensa), **ajusta** seus 69 mil
parâmetros, e repete isso **600 mil vezes** (passos) com **5 aleatoriedades
diferentes** (seeds) — depois comparamos com **4 adversários clássicos**
(baselines) em **20 partidas** (episódios) e checamos se a diferença é real
(**p-valor**), sempre olhando também quem sofreu mais (**p95**).
