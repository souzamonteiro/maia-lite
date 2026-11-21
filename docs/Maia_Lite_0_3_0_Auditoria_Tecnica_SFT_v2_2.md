# Maia Lite 0.3.0 — Auditoria técnica do SFT v2.2

**Escopo:** inspeção estática do notebook executado `Maia_Lite_0_3_0_SFT_v2_2_LoRA_L4_A100(1).ipynb`, do construtor `Maia_Lite_0_3_0_SFT_v2_2_Aya_Multilingual_Builder.ipynb` e do benchmark `Maia_Lite_0_3_0_Base_vs_LoRA_v2_2_Multilingual_Benchmark.ipynb`. Não foram reexecutados o treinamento ou os checkpoints remotos. Resultados empíricos citados são os apresentados pelo usuário.

## Veredito executivo

O SFT v2.2 apresenta um pipeline causal coerente: os rótulos são deslocados em um token, o prefixo da pergunta é mascarado, os pesos do modelo-base são congelados e o LoRA é aplicado a `qkv` e `proj`. A redução da loss de validação e o melhor comportamento de parada EOS sugerem aprendizado real de formato instrucional. Entretanto, o pipeline ainda apresenta riscos de **formatação ambígua da fronteira prompt/resposta**, **descarte evitável de registros**, **retomada que não preserva integralmente o estado**, e **ausência de avaliação generativa durante o treino**. Não há evidência para atribuir a esses pontos, isoladamente, as alucinações e repetições observadas.

## Achados priorizados

### P0 — Retomada não totalmente determinística (células 16 e 18)

- O arquivo `adapter_latest.pt` salva adaptador, optimizer, scheduler, RNG e `update`, mas não guarda explicitamente o estado do grad scaler (não há scaler neste treino), a posição de microbatch em atualização parcial nem hashes do próprio checkpoint-base/tokenizer.
- A lógica de `KeyboardInterrupt` salva `update` mesmo quando uma atualização já foi concluída mas ainda não foi persistida em disco; se a interrupção ocorrer no meio de uma acumulação, os gradientes parciais são descartados corretamente, porém o estado de RNG pode ter avançado. Uma retomada nesse caso pode divergir de uma execução contínua.
- O `except KeyboardInterrupt` chama `save_adapter(update)` sem antes restaurar o estado de RNG correspondente ao início da atualização interrompida. Isso quebra a garantia de reprodução bit a bit para dropout.
- A ordem dos exemplos é reconstruída por `default_rng(SEED+epoch)`, e o índice `update` é usado para pular atualizações, o que é bom; mas não corrige o consumo de RNG em uma atualização parcial.

**Correção v2.3:** salvar checkpoints somente em limites confirmados de atualização; guardar estado RNG no início de cada atualização e restaurá-lo se houver interrupção antes de `optimizer.step`; salvar identificador/hash de base e tokenizer. Testar execução contínua vs. retomada em passos equivalentes.

### P1 — Delimitação implícita prompt/resposta (célula 12)

```python
text = prompt + answer
enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
```

- Não existe separador estrutural explícito no treinador; depende de o campo `prompt` já terminar com uma marca de resposta, espaço ou quebra de linha.
- O `boundary_merge` rejeita qualquer token BPE que atravesse o limite entre as duas strings. Essa rejeição é uma proteção correta contra supervisionar uma fronteira de token ambígua, mas custa exemplos.
- O prompt de inferência precisa reproduzir **exatamente** o prefixo de treino. Inserir arbitrariamente `### Response:` só na inferência pioraria o alinhamento.

**Correção v2.3:** definir um formato versionado e uniforme, por exemplo `Instrução: ...\nResposta:\n` (ou equivalente multilíngue), com tokens existentes no vocabulário, e tokenizar prompt e resposta de forma separada, verificando a equivalência do prefixo; preferir uma fronteira que não produza merges problemáticos. Testar com exemplos reais e offsets. Esta é uma **mudança de distribuição**: exige novo adaptador, não deve ser aplicada retroativamente ao v2.2.

### P1 — Descarte de exemplos e viés de seleção (célula 12)

- `if len(ids) > MAX_LENGTH: return None, 'too_long'` descarta integralmente registros longos, em vez de truncar **apenas o prompt** de maneira controlada ou admitir sequências mais longas.
- Na execução relatada, 1.138 exemplos de treino e 52 de validação foram descartados; 746 por comprimento e 392 por merge no treino. Isso reduz a cobertura e pode mudar a proporção por fonte/idioma.
- A perda de validação é calculada somente nos registros aceitos; deve-se publicar também a cobertura por idioma e por fonte.

**Correção v2.3:** medir histogramas de comprimento e exclusão por idioma/fonte, avaliar `MAX_LENGTH=768` ou `1024` na L4/A100 conforme memória, e usar uma política explícita de truncamento que preserve a resposta inteira. Não truncar silenciosamente a resposta.

### P1 — Validação mede previsão de tokens, não comportamento (células 15 e 20)

- `evaluate()` acumula soma de cross-entropy apenas nos tokens supervisionados e divide pelo total de tokens; isso está correto para uma **loss microponderada por token**.
- O valor global pode favorecer idiomas com respostas mais longas; os valores por idioma não são diretamente comparáveis como medida de competência.
- A célula de geração usa temperatura 0,7 sem top-k/top-p e decodifica também o prompt. A avaliação de geração não é integrada à escolha do melhor checkpoint.
- O benchmark posterior encontrou respostas que terminam mais cedo, mas também contradições e repetições. A revisão humana cega permanece essencial.

**Correção v2.3:** congelar um conjunto de perguntas independentes (inclusive perguntas sem resposta fornecida), avaliar com greedy e uma amostragem padronizada a cada checkpoint relevante, registrar EOS, repetições, factualidade e incerteza, e preservar loss por idioma. Não escolher checkpoint exclusivamente por uma métrica textual automática.

### P2 — Treino LoRA e precisão numérica (células 9, 10 e 18)

- O base é carregado em BF16, congelado; `A` e `B` são FP32, o update é convertido para o dtype da ativação; a implementação é internamente coerente.
- A normalização do gradiente pelo total de tokens supervisionados no grupo de acumulação está correta, inclusive no último grupo parcial.
- `model.train()` em `evaluate()` é restaurado incondicionalmente; adequado durante treino, mas uma função de avaliação reutilizável deveria preservar o estado anterior (`was_training`).
- Os alvos LoRA limitam-se a atenção (`qkv`, `proj`); isso é uma decisão de capacidade, não um bug. Uma comparação futura com MLP (`fc`, `out`) pode ser útil, porém altera número de parâmetros e custo.

**Correção v2.3:** adicionar testes unitários para equivalência do forward antes/depois da inserção do LoRA (com B inicial zero), checagem de gradientes só nos adaptadores, e round-trip de save/load.

### P2 — Integridade dos dados, versões e licenças (células 4 e 12)

- O notebook exige `audit_pass.json`, `issues_count==0`, revisão humana/licenças aprovadas e hashes SHA-256 dos JSONL; boa barreira de integridade.
- O teste de vazamento por `paperId` e `groupId` detecta interseção de IDs, mas não detecta paráfrases, duplicação semântica ou contaminação do benchmark.
- A aprovação de licença no arquivo não equivale, por si só, a auditoria jurídica externa de todos os componentes; conservar o manifesto de proveniência.

**Correção v2.3:** guardar proveniência por registro e hashes de todos os artefatos; revisar possíveis near-duplicates e manter conjunto de benchmark separado da seleção de hiperparâmetros.

## Inspeção do mascaramento causal — sem erro encontrado

A implementação faz:

```python
ids = ids + [tokenizer.eos_token_id]
labels = [-100] * (len(ids) - 1)
for i in range(len(labels)):
    if i + 1 >= answer_idx:
        labels[i] = ids[i + 1]
return {'x': ids[:-1], 'y': labels, ...}
```

Como o modelo recebe `ids[:-1]` e prediz `ids[1:]`, o primeiro token da resposta é previsto a partir do prefixo da pergunta e recebe supervisão; o EOS final também recebe supervisão. As posições anteriores são mascaradas. A loss usa `ignore_index=-100`. Isso é causalmente correto. **Não recomendo alterar esse deslocamento.**

## Testes mínimos de aceitação antes do v2.3

1. **Fronteira:** 100 registros amostrados por idioma; reconstruir `prompt`, `completion`, primeiro token supervisionado e EOS; zero merge involuntário.
2. **Cobertura:** tabela por idioma/fonte com originais, aceitos, descartados e motivo; meta explícita de perdas.
3. **Mascaramento:** assert de que todos os tokens da pergunta têm label `-100` e todos os tokens da resposta + EOS têm label válido.
4. **LoRA:** comparar logits antes/depois de inserir adaptadores zerados; gradientes de base nulos; save/load reproduz logits.
5. **Retomada:** comparar pesos, optimizer, scheduler, LR e RNG após N updates contínuos vs. execução N/2 + reinício + N/2.
6. **Avaliação:** benchmark multilíngue independente, revisão humana cega e perguntas de incerteza; registrar EOS e repetição.
7. **Reprodutibilidade:** hash de base, tokenizer, dados, código, configuração, seeds e versão de PyTorch.

## Decisão recomendada

**Não repetir o SFT v2.2 agora.** Preservar `adapter_best.pt` do passo 1800 como baseline imutável. Antes de investir em GPU para v2.3, implementar os testes acima e revisar o formato de entrada. Depois, executar um piloto curto com comparações controladas (mesmo corpus, mesma validação e mesmo orçamento de tokens) para isolar o efeito da nova formatação. Se a qualidade não melhorar, investigar composição/qualidade do corpus e capacidade do modelo, em vez de aumentar épocas indiscriminadamente.

## Limitações desta auditoria

Inspeção estática do código local; não há acesso direto aos JSONL e checkpoints armazenados no Drive do Colab nesta sessão. Os números do treinamento e do benchmark são os fornecidos no histórico da conversa. Não foram executados testes de gradiente, equivalência numérica, vazamento semântico nem validação de licenças externas.
