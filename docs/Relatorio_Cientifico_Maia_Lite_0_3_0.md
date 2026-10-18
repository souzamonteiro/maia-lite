# Maia Lite 0.3.0 — Fundamentação técnico-científica da mudança de direção

**Documento de trabalho para fundamentação de artigo científico**  
**Data:** 18 de outubro de 2026  
**Projeto:** Maia Platform / Maia Lite  
**Status:** proposta de pesquisa; hipóteses e parâmetros sujeitos a validação

## Resumo executivo

O desenvolvimento da Maia Lite, um modelo autoregressivo de linguagem com aproximadamente 355 milhões de parâmetros baseado na arquitetura GPT-2, produziu resultados exploratórios que motivam uma revisão de sua estratégia de treinamento. Em avaliações internas, versões sucessivas exibiram limitações de correção factual, inferência elementar, seguimento de instruções e estabilidade conversacional. Um achado qualitativo particularmente relevante foi a diferença entre a capacidade atribuída à Maia Lite 0.1.0 de responder ao comando aberto “Fale sobre a saudade” e a dificuldade observada na Maia Lite 0.2.0 com essa formulação, embora respondesse à pergunta definicional “O que é a saudade?”. Essa observação sugere possível regressão de competência instrucional, mas ainda exige reprodução controlada para estabelecer magnitude e causa.

Comparações exploratórias com Gemma 3 270M, SmolLM2 360M e Qwen 2.5 0.5B evidenciaram desempenhos heterogêneos: fluência verbal não assegurou correção conceitual, e alguns modelos exibiram deterioração em interações de múltiplos turnos. Essas comparações não isolam arquitetura, dados, volume de treinamento, tokenizer ou pós-treinamento e, portanto, não demonstram que corpora maiores sejam ineficazes nem que GPT-2 seja superior a arquiteturas recentes. Elas mostram, entretanto, que o simples aumento do volume de dados não constitui, por si só, uma justificativa experimental suficiente para a próxima etapa do projeto.

Propõe-se a Maia Lite 0.3.0 como um programa de pesquisa orientado por competências: pré-treinamento a partir de pesos novos, arquitetura GPT-2 como referência simples e reprodutível, tokenizer trilíngue desenvolvido especificamente para português, inglês e espanhol, corpus curado de linguagem natural, diálogo, explicação, criação textual e lógica formal expressa em linguagem natural. O ajuste fino supervisionado (SFT) deixa de ser etapa obrigatória e passa a ser uma intervenção experimental opcional, condicionada à avaliação do modelo base. A seleção de checkpoints será guiada por testes de aquisição, generalização, transferência entre idiomas e preservação de competências, e não exclusivamente pela perda de previsão de tokens.

**Contribuição científica pretendida:** investigar em que medida a especialização da distribuição de treinamento, em lugar do simples aumento indiscriminado de dados, melhora competências linguísticas e inferenciais de um modelo autoregressivo compacto, sob orçamento computacional controlado.

---

## 1. Contexto e problema de pesquisa

A Maia Lite foi concebida como uma iniciativa de desenvolvimento e estudo de modelos de linguagem pequenos, com execução acessível e suporte a português, inglês e espanhol. O desenvolvimento anterior envolveu pré-treinamento autoregressivo em corpus multilíngue, com forte presença de material enciclopédico, e etapas posteriores de adaptação supervisionada.

O objetivo de um modelo conversacional, entretanto, não se reduz à reprodução de regularidades textuais. Para cumprir instruções em linguagem natural, o modelo deve identificar a intenção comunicativa, sustentar uma elaboração coerente, distinguir o que decorre ou não das premissas e manter estabilidade ao longo de interações.

O problema de pesquisa pode ser formulado assim:

> **Sob um orçamento computacional limitado, uma distribuição de pré-treinamento orientada por competências linguísticas e lógicas permite a um modelo causal compacto adquirir e preservar melhor essas competências do que uma distribuição predominantemente generalista?**

Essa formulação não pressupõe que um modelo de 355M parâmetros desenvolva raciocínio geral, tampouco equipara geração de justificativas verbais a inferência formalmente correta.

## 2. Evidências exploratórias acumuladas

### 2.1. Maia Lite 0.1.0 versus 0.2.0: formulações de uma mesma intenção

Uma observação reportada no desenvolvimento foi:

- **Maia Lite 0.1.0:** conseguia responder à instrução aberta “Fale sobre a saudade”.
- **Maia Lite 0.2.0:** respondia a “O que é a saudade?”, mas apresentava dificuldade com “Fale sobre a saudade”.

Esses enunciados envolvem competências parcialmente diferentes. O primeiro demanda reconhecer uma solicitação aberta e produzir discurso autônomo; o segundo admite uma resposta definicional relativamente padronizada. A diferença é consistente com a hipótese de sensibilidade ao formato instrucional ou de regressão de competência, mas **não prova esquecimento catastrófico**. Para essa conclusão seria necessário controlar checkpoints, prompts, templates, decodificação e repetições de teste, bem como identificar a etapa na qual o comportamento se alterou.

O par “saudade” deve ser preservado como **teste sentinela de regressão**, acompanhado de temas e paráfrases adicionais para evitar sobreajuste ao exemplo.

### 2.2. Comportamento da Maia Lite em avaliações internas

Avaliações exploratórias de versões 0.2.0 e Stage 4 registraram respostas vazias ou encerramento precoce, erros em tarefas elementares, respostas factuais inadequadas e dificuldades em seguir instruções. O histórico de desenvolvimento também indicou que diminuições da loss não se converteram automaticamente em melhorias correspondentes nas competências conversacionais medidas.

A interpretação deve ser restrita: tais observações mostram desalinhamento entre uma métrica de treinamento e o objetivo de aplicação, **não** demonstram que a otimização da previsão do próximo token seja incapaz de produzir competências complexas.

### 2.3. Comparações exploratórias com modelos de pequeno porte

| Modelo | Escala aproximada | Observações nas avaliações internas | Limitação interpretativa |
|---|---:|---|---|
| Gemma 3 270M | 270M | Fluência conversacional em alguns exemplos, coexistindo com erros conceituais e matemáticos | Não foi isolado o efeito de arquitetura ou dados |
| Maia Lite 0.2.0 | 355M | Respostas vazias e desempenho instável em algumas tarefas | Configuração de treinamento e inferência específica |
| SmolLM2 360M Instruct | 360M | Algumas respostas adequadas em turno único; em uma execução multiturmo, recusa inicial propagou-se às respostas seguintes | Resultado dependente da sequência, template e decodificação |
| Qwen 2.5 0.5B | ~500M | Melhor desempenho em parte das tarefas matemáticas simples examinadas | Maior número de parâmetros e treinamento não controlado |

Esses resultados são **exploratórios**, provenientes de um conjunto limitado de perguntas e configurações. Não constituem benchmark público padronizado nem comparação causal entre arquiteturas. A ausência de vantagem uniforme nos testes **não autoriza** concluir que modelos modernos não sejam melhores ou que mais tokens não tragam benefícios; autoriza, sim, exigir evidência experimental antes de investir em escalonamento volumétrico.

### 2.4. Síntese do diagnóstico

Há três explicações plausíveis e não mutuamente exclusivas:

1. **Cobertura insuficiente de competências:** os dados anteriores podem ter favorecido conhecimento declarativo e padrões enciclopédicos, sem exposição adequada a instruções abertas, diálogo e inferência.
2. **Sensibilidade à distribuição de pós-treinamento:** a adaptação supervisionada pode ter privilegiado formatos estreitos de pergunta–resposta, com regressão em outros formatos; isso precisa ser testado.
3. **Efeitos de implementação e inferência:** templates de chat, tokens de parada, amostragem, truncamento e exportação podem alterar substancialmente o comportamento observado.

A investigação de 0.3.0 deve distinguir essas explicações em vez de atribuir todos os problemas a uma causa única.

## 3. Justificativa para a mudança de estratégia

### 3.1. De volume indiscriminado para competência mensurável

A literatura de escalonamento mostra relações entre parâmetros, dados, computação e perda preditiva, mas essas relações não implicam que qualquer ampliação de corpus produza ganho equivalente em uma tarefa específica. Para um modelo compacto e um orçamento restrito, é pertinente investigar a **eficiência de aquisição de competências por token processado e por unidade de computação**.

A nova proposta prioriza dados com funções pedagógicas e semânticas definidas, sem abandonar a diversidade necessária à generalização.

### 3.2. Manutenção do GPT-2 como plataforma experimental

A arquitetura GPT-2 causal é proposta por simplicidade, maturidade de implementação, facilidade de instrumentação e continuidade com a experiência adquirida. **Não se afirma superioridade geral de GPT-2** em qualidade ou velocidade. Qualquer comparação de eficiência com arquiteturas recentes deverá controlar hardware, precisão numérica, backend de inferência, comprimento de contexto, batch e número de parâmetros.

O modelo-base proposto terá aproximadamente 355M parâmetros, sujeito a ajuste de contagem após a escolha do vocabulário. A troca do tokenizer altera as dimensões das matrizes de embeddings e saída; por isso, a configuração final deve ser documentada e os pesos inicializados do zero.

### 3.3. Tokenizer trilíngue novo

Propõe-se treinar um tokenizer byte-level BPE com dados balanceados e representativos de português, inglês e espanhol. Vocabulários candidatos de 32k, 40k e 50k serão comparados em conjuntos não vistos quanto a:

- tokens por palavra e por caractere, por idioma e gênero textual;
- tratamento de acentos, flexões, pontuação e nomes próprios;
- custo computacional do vocabulário e comprimento efetivo de contexto;
- robustez de codificação/decodificação e tratamento de tokens especiais.

O valor de **40k** é apenas candidato inicial, não escolha definitiva. Serão definidos explicitamente os marcadores de início/fim de documento e de turnos de diálogo, evitando colisões com texto comum.

### 3.4. Treinamento do zero e SFT não obrigatório

O pré-treinamento do zero evita herdar vieses e efeitos acumulados de uma cadeia de adaptações anterior e permite observar o surgimento de competências ao longo do treinamento. Contudo, **treinar do zero não impede esquecimento**, e o pré-treinamento especializado também pode comprometer competências quando sua distribuição é estreita.

O SFT será aplicado apenas após caracterização rigorosa do modelo base. A comparação entre **sem SFT**, **SFT curto e diversificado** e, quando justificável, **SFT mais intenso** permitirá estimar benefícios e regressões. A hipótese de esquecimento será examinada empiricamente, não assumida como fato universal.

## 4. Competências-alvo da Maia Lite Chat 0.3.0

### 4.1. Linguagem e conversação

- Compreensão de instruções abertas, definicionais, comparativas, explicativas e criativas.
- Produção coerente de parágrafos e narrativas, com progressão temática.
- Manutenção de referência e contexto em diálogos de múltiplos turnos.
- Resposta apropriada à incerteza, sem inventar fatos ou alegar capacidades inexistentes.
- Transferência dessas capacidades entre português, inglês e espanhol.

### 4.2. Lógica e inferência

Lógica será uma competência central, **sem confundi-la com programação ou matemática avançada**. O corpus poderá conter:

- lógica proposicional: negação, conjunção, disjunção, implicação e equivalência;
- lógica de predicados: quantificadores, relações, inclusão e exclusão;
- dedução: cadeias de inferência, silogismos e contradições;
- condições necessárias e suficientes;
- validade argumentativa, falácias e contraexemplos;
- inferência indutiva e analogias, explicitando sua natureza não dedutiva;
- raciocínio causal, distinguindo causalidade de correlação e de implicação lógica.

Exemplos com palavras inventadas, entidades sintéticas e permutações de símbolos são essenciais para reduzir a possibilidade de acerto por memorização de fatos conhecidos. As conclusões formais deverão ser produzidas ou verificadas por ferramentas simbólicas, quando aplicável.

### 4.3. Distinção entre lógica formal e explicações textuais

Uma explicação convincente não garante uma inferência correta. O benchmark deverá avaliar **a conclusão independentemente da justificativa** e medir separadamente se a justificativa é fiel às premissas. Para tarefas indeterminadas, a resposta correta pode ser “não é possível concluir”.

## 5. Desenho preliminar do corpus

### 5.1. Distribuição por competência

| Domínio | Participação inicial em tokens amostrados | Finalidade |
|---|---:|---|
| Linguagem natural, prosa e conhecimento | 30% | Competência linguística geral e contextualização |
| Diálogos e conversação | 20% | Interação, turnos e referência contextual |
| **Lógica e raciocínio estruturado** | **25%** | Inferência, validade, limites das premissas |
| Textos explicativos e educacionais | 15% | Exposição, explicação e organização conceitual |
| Narrativas e produção criativa | 10% | Elaboração livre e flexibilidade discursiva |
| **Total** | **100%** | |

Essas proporções são **pontos de partida experimentais**, não resultados de otimização. A participação de 25% em lógica é deliberadamente forte para testar sua contribuição, mas deve ser confrontada com misturas alternativas (por exemplo, 10%, 15% e 25%) e com um controle sem enriquecimento lógico.

### 5.2. Distribuição por idioma

Proposta inicial: **PT 40%, EN 35%, ES 25%**, aferida por tokens amostrados. A proporção será revisada conforme qualidade e disponibilidade dos dados. A divisão de treinamento, validação e teste deve ser feita antes da tradução ou geração de variantes, agrupando exemplos semanticamente equivalentes para evitar vazamento entre conjuntos.

### 5.3. Qualidade, procedência e licenciamento

Cada fonte deverá ter identificação, idioma, licença/termos de uso, origem, data de coleta, método de limpeza e histórico de transformação. Serão aplicados filtros de qualidade, deduplicação exata e aproximada, controle de contaminação e verificação de representatividade. Dados sintéticos deverão preservar rastreabilidade do gerador e do verificador.

A construção do corpus deverá evitar uma predominância de formatos artificiais rígidos, pois o objetivo é transferir inferência para linguagem natural variada.

### 5.4. Currículo de complexidade: hipótese a testar

Pode-se avaliar um currículo progressivo — premissas simples, inferências encadeadas e problemas com informação insuficiente — contra amostragem aleatória de dificuldades. **Não se pressupõe que currículo progressivo seja superior**; essa é outra variável experimental.

## 6. Hipóteses científicas

**H1 — Especialização:** sob o mesmo orçamento de tokens e computação, um corpus orientado por competências melhora o desempenho em instruções abertas e diálogo em relação a um corpus generalista de referência.

**H2 — Inferência:** enriquecimento com exemplos lógicos verificados melhora a precisão em problemas inéditos de validade e dedução.

**H3 — Generalização estrutural:** parte do ganho em lógica persiste quando se alteram entidades, vocabulário, ordem das premissas e profundidade das inferências.

**H4 — Transferência trilíngue:** competências lógicas aprendidas em uma distribuição multilíngue se transferem parcialmente entre português, inglês e espanhol.

**H5 — Preservação:** é possível elevar competências lógicas sem regressão substantiva nas competências de elaboração livre e conversação, mediante seleção apropriada da mistura de dados e dos checkpoints.

**H6 — Pós-treinamento:** SFT pode melhorar seguimento de instruções, mas seus ganhos e eventuais regressões dependem da composição, intensidade e configuração; sua necessidade deve ser determinada por comparação controlada.

## 7. Protocolo experimental proposto

### 7.1. Controles mínimos

Treinar pilotos com arquitetura, tokenizer, inicialização estatística, comprimento de contexto, tokens processados, otimizador, cronograma de taxa de aprendizagem, batch efetivo e orçamento computacional equivalentes. Utilizar mais de uma semente quando viável e relatar variabilidade.

**Comparação primária:** corpus generalista versus corpus especializado, com mesmo número de tokens processados.

**Ablação de lógica:** 0%, 10%, 15% e 25% de tokens lógicos, compensando os demais domínios de maneira pré-especificada.

**Ablação de pós-treinamento:** modelo base versus SFT controlado, usando o mesmo checkpoint inicial.

**Atenção:** mudar tokenizer e corpus simultaneamente impede atribuir todo o ganho apenas ao corpus. Para isolar a contribuição do tokenizer, será necessário um experimento específico mantendo os demais fatores comparáveis.

### 7.2. Métricas de treinamento

- Loss e perplexidade em conjuntos retidos por idioma e por domínio.
- Quantidade de tokens efetivamente processados e estimativa de tokens únicos.
- Tempo de treinamento, memória, throughput e custo computacional.
- Sensibilidade a checkpoints, sementes e comprimento de contexto.

### 7.3. Métricas de competência

| Competência | Avaliação proposta |
|---|---|
| Definição | Relevância, correção e completude |
| Instrução aberta | Atendimento à intenção, coerência e elaboração |
| Diálogo | Consistência referencial e estabilidade em múltiplos turnos |
| Lógica formal | Acurácia de conclusão, validade e reconhecimento de indeterminação |
| Generalização | Desempenho em templates, entidades e profundidades não vistos |
| Transferência trilíngue | Desempenho por idioma e diferença entre idiomas |
| Preservação | Variação relativa às competências já adquiridas em checkpoints anteriores |
| Robustez de inferência | Taxa de respostas vazias, loops, encerramento precoce e sensibilidade ao template |

A avaliação de respostas abertas combinará rubricas explícitas, avaliação humana cega em amostras e, se utilizada, avaliação automatizada calibrada contra julgamentos humanos. Problemas lógicos terão gabarito independente do modelo, preferencialmente verificado simbolicamente.

### 7.4. Teste sentinela de formulação

Um conjunto de tópicos — saudade, memória, amizade, liberdade, conhecimento, entre outros — será apresentado por diferentes atos comunicativos:

1. “O que é X?” — definição.
2. “Fale sobre X.” — elaboração livre.
3. “Explique X com um exemplo.” — explicação aplicada.
4. “Compare X e Y.” — contraste.
5. “Escreva um parágrafo sobre X.” — produção dirigida.
6. “Continue sua explicação anterior.” — continuidade conversacional.

O teste será aplicado em PT, EN e ES, com versões semanticamente equivalentes e variação de ordem. Não deverá ser usado como fonte de exemplos de treinamento. O par original da saudade será preservado como item histórico, mas conclusões dependerão de um conjunto mais amplo.

### 7.5. Controles de inferência

Registrar para cada execução: hash do checkpoint, tokenizer, template, tokens especiais, temperatura, top-p, seed quando suportada, comprimento máximo, penalidades de repetição, hardware e versão do runtime. Comparar também a inferência no framework original com a versão exportada para Ollama/GGUF para excluir discrepâncias de conversão.

## 8. Critérios de decisão e riscos

A adoção definitiva da mistura especializada dependerá de melhoria reproduzível nas competências-alvo **sem degradação inaceitável** em linguagem geral, medida em conjunto de regressão pré-registrado. Os limiares quantitativos devem ser definidos antes da análise dos pilotos.

Riscos principais:

- **Sobreajuste a templates lógicos:** mitigar com geração composicional, variação lexical e testes fora da distribuição.
- **Contaminação de avaliação:** separar famílias de problemas, não apenas linhas individuais.
- **Perda de fluência por excesso de lógica sintética:** preservar diversidade de prosa e acompanhar testes abertos em cada checkpoint.
- **Desbalanceamento linguístico:** medir cobertura real e métricas por idioma.
- **Conclusões causais indevidas:** realizar ablações de corpus, tokenizer e SFT separadamente.
- **Limite de capacidade de 355M parâmetros:** reportar limites observados, inclusive se não houver transferência ou ganho.
- **Custos de treinamento do zero:** executar pilotos e curvas de aprendizagem antes da escala completa.

## 9. Relação com uma futura família de modelos

A Maia Lite 0.3.0 **Chat** será o primeiro objeto de investigação, com competências centrais em linguagem, conversação e lógica. Versões futuras **Math** e **Python** poderão usar a mesma infraestrutura, mas com objetivos, corpora e benchmarks especializados. Não se assume que uma única base ou sequência de SFTs seja a melhor solução para todas as competências.

Uma arquitetura de aplicação com roteamento por competência poderá combinar modelos especializados, sem exigir que cada modelo execute todas as tarefas. Essa possibilidade é uma perspectiva de engenharia, não um resultado do presente estudo.

## 10. Contribuições esperadas para publicação

1. Um procedimento reproduzível para construção e auditoria de corpus trilíngue orientado por competências.
2. Um tokenizer trilíngue caracterizado por eficiência de segmentação e custo computacional.
3. Um protocolo de avaliação que separa fluência, seguimento de instruções, inferência lógica e preservação de competências.
4. Ablações controladas do enriquecimento lógico e do pós-treinamento em um Transformer compacto.
5. Evidências sobre os limites e as possibilidades de generalização lógica e transferência entre PT, EN e ES em escala de aproximadamente 355M parâmetros.
6. Um relato transparente de resultados negativos, caso o enriquecimento não produza ganhos mensuráveis.

## 11. Formulação sugerida para o artigo

> Motivados por avaliações exploratórias de modelos de linguagem compactos e por indícios de regressão de comportamento instrucional entre versões sucessivas da Maia Lite, investigamos uma estratégia de pré-treinamento orientada por competências. Em vez de priorizar exclusivamente a expansão volumétrica do corpus, propomos construir uma distribuição trilíngue que combine linguagem natural, diálogo e problemas de inferência lógica formalmente verificáveis. Utilizamos uma arquitetura causal GPT-2 como plataforma experimental pela simplicidade e reprodutibilidade, sem pressupor superioridade arquitetural. O estudo avaliará aquisição, generalização e preservação de competências sob orçamentos de treinamento controlados, incluindo ablações da composição dos dados e do ajuste fino supervisionado.

Este texto descreve **motivação e desenho de pesquisa**. Para a versão final do artigo, deverá ser reescrito no passado e complementado com resultados, intervalos de confiança e análise estatística somente após a execução dos experimentos.

## 12. Próximas etapas recomendadas

**Etapa 0 — Reprodutibilidade histórica:** recuperar configurações, checkpoints, prompts e saídas de 0.1.0/0.2.0/Stage 4; reproduzir a diferença de formulação e identificar possíveis fatores de inferência.

**Etapa 1 — Especificação:** congelar arquitetura de referência, métricas, política de dados, regras de separação dos conjuntos e benchmark pré-registrado.

**Etapa 2 — Tokenizer:** treinar e comparar candidatos, selecionar com base em eficiência multilíngue e custo.

**Etapa 3 — Corpus piloto:** preparar controles generalista e especializado, com proveniência e deduplicação documentadas.

**Etapa 4 — Pilotos e ablações:** executar treinamentos equivalentes, avaliar checkpoints e decidir se há evidência para escala maior.

**Etapa 5 — Treinamento 0.3.0:** somente após confirmação experimental, ampliar o orçamento e selecionar checkpoint por competências.

**Etapa 6 — SFT opcional:** aplicar intervenção pequena e auditável apenas se houver ganho mensurável sem regressão relevante.

---

## Referências de enquadramento teórico

As referências abaixo são pontos de partida conceituais; a revisão bibliográfica definitiva deverá verificar edições, resultados específicos, literatura posterior e adequação das citações ao texto final.

1. Radford, A. et al. (2019). *Language Models are Unsupervised Multitask Learners*. OpenAI. (Arquitetura e pré-treinamento GPT-2.)
2. Kaplan, J. et al. (2020). *Scaling Laws for Neural Language Models*. arXiv:2001.08361. (Relações empíricas entre escala e perda.)
3. Hoffmann, J. et al. (2022). *Training Compute-Optimal Large Language Models*. arXiv:2203.15556. (Alocação de parâmetros e dados sob orçamento computacional.)
4. Wei, J. et al. (2022). *Emergent Abilities of Large Language Models*. arXiv:2206.07682. (Debate sobre desempenho e escala; interpretação dependente de métricas.)
5. Schaeffer, R., Miranda, B., & Koyejo, S. (2023). *Are Emergent Abilities of Large Language Models a Mirage?* NeurIPS. (Cautelas sobre métricas e aparentes transições abruptas.)
6. Kirkpatrick, J. et al. (2017). *Overcoming catastrophic forgetting in neural networks*. PNAS. (Fundamentação geral de esquecimento catastrófico, sem atribuir causalidade ao caso Maia Lite.)
7. McCoy, R. T., Pavlick, E., & Linzen, T. (2019). *Right for the Wrong Reasons: Diagnosing Syntactic Heuristics in Natural Language Inference*. ACL. (Importância de testes que distingam heurísticas de generalização.)
8. Sennrich, R., Haddow, B., & Birch, A. (2016). *Neural Machine Translation of Rare Words with Subword Units*. ACL. (Fundamentação de segmentação em subpalavras.)

## Nota sobre rastreabilidade da evidência

Este relatório foi elaborado a partir das observações e resultados discutidos durante o desenvolvimento do projeto, **sem reprocessamento integral dos logs originais nesta etapa**. Antes da submissão científica, os resultados numéricos, condições de teste, artefatos de avaliação e alegações históricas deverão ser vinculados a arquivos versionados, hashes, scripts reproduzíveis e tabelas auditáveis. As comparações aqui relatadas são evidências preliminares de motivação, não resultados conclusivos do estudo proposto.
