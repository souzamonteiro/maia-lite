# Maia Lite 0.3.0 — Relatório de conformidade e início do pré-treinamento 0,5B

**Data:** 2026-11-04  
**Decisão de engenharia:** novo modelo de aproximadamente 500 milhões de parâmetros, inicializado aleatoriamente, pré-treinado do zero com Maia Corpus v8. **Não** utilizar pesos do Maia Lite 355M anterior.  
**Situação:** auditoria de proveniência concluída; revisão de direitos e conclusão do corpus ainda pendentes. Este relatório **não** concede autorização jurídica.

## 1. Evidência técnica confirmada

- Inventário de recuperação SHA-256: `ad19d1d98e2cf5b91e650f343e6430bbde34b549b5fbc352b47682a91d7ab562`.
- 21.021 chunks no inventário de recuperação; 20.371 concluídos e 650 de `wikipedia_pt` pendentes de processamento na última situação reportada. Esses números não constituem uma validação atual de integridade de todos os shards nem uma contagem de tokens finais.
- Auditoria de proveniência dos 650 chunks: **650/650**, **313.162 registros**, `COMPLETE_PENDING_HUMAN_REVIEW`.
- Os 313.162 registros têm `rights_status=pending_review` e `eligible_for_training=false`. A ocorrência `not_marked_training_eligible` é um bloqueio de política, não um erro de leitura.
- Snapshot Hugging Face consultado em 2026-11-04: `wikimedia/wikipedia`, commit `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`, configuração `20231101.pt`, seis arquivos Parquet; `resolved_sha` igual ao solicitado.
- Metadados de licença no snapshot: **`cc-by-sa-3.0` e `gfdl`**. A referência anterior a CC BY-SA 4.0 não deve substituir silenciosamente esses metadados.

## 2. Escopo e interpretação jurídica

O repositório Wikimedia Wikipedia identifica CC BY-SA 3.0 e GFDL e descreve utilização em modelagem de linguagem. Licenças livres podem permitir reutilização comercial sob condições de atribuição e compartilhamento. Isso **não determina por si só** o tratamento jurídico dos pesos treinados, a elegibilidade de todo artigo individual, nem os direitos de todas as outras fontes do corpus. É necessário revisar as exceções, a política de atribuição, direitos de terceiros, dados pessoais e a distribuição comercial planejada.

**Não aprovado:** transformar automaticamente `pending_review` em `approved`, modificar `eligible_for_training=false`, gerar `license_approval.json` fictício ou contornar `LICENSE AUDIT: BLOCKED`. Uma revisão feita durante o treinamento não substitui a autorização necessária antes de usar dados bloqueados.

## 3. Fontes e rastreabilidade

- Dataset card: https://huggingface.co/datasets/wikimedia/wikipedia
- Snapshot: https://huggingface.co/datasets/wikimedia/wikipedia/tree/b04c8d1ceb2f5cd4588862100d08de323dccfbaa
- Termos Wikimedia: https://foundation.wikimedia.org/wiki/Terms_of_Use
- Auditoria local: `production_resume_v2/wikimedia_rights_review/provenance_audit/provenance_summary.json` e `chunks/`.
- Documentos anteriores: `MAIA_CORPUS_V8_COMPLIANCE_REPORT.md`, `WIKIPEDIA_PT_NOTICE_DRAFT.md`, `COMPLIANCE_REVIEW_STATUS.json`.

## 4. Arquitetura e treinamento: decisões ainda a fixar

- **Modelo:** decoder-only autoregressivo, pesos aleatórios, alvo ~0,5B. A configuração exata deve ser medida pelo contador de parâmetros do código antes de qualquer treinamento; não chamar de 500M uma arquitetura sem contagem.
- **Tokenizer:** treinar e congelar a partir de amostra balanceada e autorizada; registrar hash, normalização, tokens especiais e vocabulário. Não reutilizar tokenizer do 355M sem decisão explícita.
- **Contexto:** avaliar 2.048 tokens como ponto inicial, com teste de memória e throughput na GPU efetivamente disponível.
- **Mistura:** usar os dados elegíveis do corpus v8, com amostragem PT/EN/ES documentada, deduplicação entre fontes e split held-out por documento; não assumir que todos os 21.021 chunks são autorizados.
- **Orçamento:** plano anterior propunha 21,289 bilhões de tokens para ~355M; para 0,5B é apenas referência histórica, não orçamento automaticamente válido. Medir tokens únicos e repetições antes de definir steps.
- **Otimização:** AdamW, BF16 se suportado, grad accumulation, gradient checkpointing e salvamento atômico de checkpoints contendo modelo, optimizer, scheduler, scaler (se aplicável), RNG, passo e posição do dataset.
- **Métricas:** loss e perplexidade held-out por idioma, tokens/s, GPU RAM, taxa de repetição, detecção de vazamento, testes de memorização e benchmarks congelados.

## 5. Gates de execução

**Gate A — Preparação (pode iniciar agora):** congelar especificação 0,5B, tokenizer, auditoria de integridade, splits e benchmarks; realizar smoke test com dados sintéticos. Nenhum shard bloqueado é lido para atualização de pesos.

**Gate B — Corpus elegível:** cada fonte usada no treinamento deve possuir aprovação humana rastreável para uso pretendido; auditar também fontes FineWeb e demais fontes dos 20.371 chunks. Reexecutar auditoria e exigir `LICENSE AUDIT: PASS`; completar a produção dos 650 chunks sem alterar o original; conferir SHA, deduplicação e contagens.

**Gate C — Pré-treinamento real:** iniciar pesos aleatórios, executar canary curto com subconjunto elegível, avaliar perda, validação e retomada de checkpoint; somente depois lançar treinamento integral.

**Gate D — Distribuição:** revisão separada da liberação pública/comercial dos pesos, documentação de proveniência, avisos e avaliação de riscos de memorização.

## 6. Ações imediatas

1. Rodar o notebook de preparação incluído no pacote para verificar caminhos, estado de aprovação e GPU, sem modificar o corpus.
2. Fornecer o resumo atualizado do inventário completo, tokenizer e GPU-alvo para dimensionar o modelo e treinamento.
3. Concluir a revisão humana e registrar decisões autênticas por fonte; jamais inferir aprovação da licença do card isoladamente.
4. Processar os 650 chunks quando autorizados e criar dataset final versionado com split de validação congelado.
5. Iniciar o canary de treinamento do zero e, após aprovação dos testes, o pré-treinamento integral.
