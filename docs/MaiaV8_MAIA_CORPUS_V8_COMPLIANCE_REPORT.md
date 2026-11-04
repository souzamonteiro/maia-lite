# Maia Corpus v8 — Relatório de conformidade e proveniência

**Versão:** 1.0 — 04/11/2026  
**Status:** `TECHNICAL_PROVENANCE_COMPLETE / RIGHTS_REVIEW_PENDING`  
**Escopo:** 650 chunks pendentes de `wikipedia_pt`, mais avaliação preliminar de publicação dos pesos do Maia Lite 0.3.0.  
**Natureza:** relatório técnico-documental; **não é parecer jurídico nem autorização para treinamento/publicação**.

## 1. Resumo executivo

A auditoria incremental de proveniência foi concluída para os 650 chunks restantes de `wikipedia_pt`, examinando **313.162 registros**. Todos apresentam os mesmos identificadores declarados de dataset, revisão, configuração e split. Entretanto, **todos** permanecem com `rights_status="pending_review"` e `eligible_for_training=false`. A conclusão técnica não equivale a uma autorização de direitos. A produção deve permanecer bloqueada até uma decisão humana fundamentada.

Foi encontrada uma divergência documental: o cartão público de `wikimedia/wikipedia` no Hugging Face lista **cc-by-sa-3.0** e **gfdl**, enquanto o registro interno de revisão usa **CC-BY-SA-4.0**. Os Termos de Uso da Wikimedia atualmente descrevem CC BY-SA 4.0 e GFDL para textos contribuídos, com exceções. **A licença aplicável ao snapshot específico não foi determinada de modo conclusivo.**

## 2. Evidências internas fornecidas

| Métrica / campo | Resultado |
|---|---|
| Chunks restantes auditados | 650 / 650 |
| Registros auditados | 313.162 |
| `dataset_id` | `wikimedia/wikipedia` (313.162) |
| `dataset_revision` | `b04c8d1ceb2f5cd4588862100d08de323dccfbaa` (313.162) |
| `config` | `20231101.pt` (313.162) |
| `split` | `train` (313.162) |
| `rights_status` | `pending_review` (313.162) |
| `eligible_for_training` | `false` (313.162) |
| Issues reportados | `not_marked_training_eligible`: 313.162 |
| Inventory SHA-256 | `ad19d1d98e2cf5b91e650f343e6430bbde34b549b5fbc352b47682a91d7ab562` |
| Estado da auditoria | `COMPLETE_PENDING_HUMAN_REVIEW` |

**Limite da evidência:** uniformidade de metadados **não** demonstra integridade do upstream, direitos de cada texto, ausência de duplicações entre chunks ou cobertura de atribuição de todos os artigos. A soma de URLs únicas por chunk não é unicidade global.

**Local declarado dos artefatos no Google Drive:**
`/content/drive/MyDrive/Colab_LLMs/Maia_Lite_0_3_0_Chat/corpus_v8/runs/acquisition_v8_5_001/pretraining_production_v1_full/recovery_audit_v4/production_resume_v2/wikimedia_rights_review/provenance_audit/`

Arquivos principais: `provenance_summary.json` e diretório `chunks/` com manifestos por chunk. Estes arquivos remotos **não foram inspecionados diretamente neste relatório**; os resultados acima foram fornecidos pelo operador.

## 3. Verificação externa preliminar

1. Dataset público: https://huggingface.co/datasets/wikimedia/wikipedia — cartão anuncia `cc-by-sa-3.0` e `gfdl`, com configuração `20231101.pt` disponível em versões públicas do repositório.
2. Exemplo de árvore pública da configuração: https://huggingface.co/datasets/wikimedia/wikipedia/tree/614d2de0683a26dc60e9ca3fcb4a14a5da94bf55/20231101.pt — arquivos Parquet do split `train`.
3. A revisão **exata** `b04c8d1ceb2f5cd4588862100d08de323dccfbaa` foi declarada no corpus, mas **não foi verificada diretamente** contra a API/árvore dessa revisão nesta análise. Confirmar hash, configuração e cartão de licença da revisão, não apenas da branch atual.
4. Termos da Wikimedia: https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use/pt — condições para reutilização, atribuição, importações de terceiros e exceções.
5. Creative Commons: https://creativecommons.org/using-cc-licensed-works-for-ai-training-2/ — análise de treinamento de IA, obrigações potenciais de atribuição e ShareAlike, com variação por jurisdição e fatos.

**Atenção:** a presença de licença de dataset não substitui a avaliação de conteúdos importados de terceiros nem prova a licença de cada artigo na data do snapshot.

## 4. Matriz de conformidade

| Requisito | Evidência atual | Estado | Ação |
|---|---|---|---|
| Proveniência declarada dos 650 chunks | Metadados uniformes | Concluído tecnicamente | Arquivar sumário e manifestos |
| Identidade da revisão upstream | Hash declarado | Não verificado independentemente | Consultar API Hugging Face na revisão exata |
| Licença aplicável ao snapshot | HF: 3.0/GFDL; registro: 4.0 | **Divergência crítica** | Conferir README/licença da revisão e termos históricos |
| Elegibilidade dos registros | 313.162 `false` | Bloqueado | Manter bloqueio até decisão registrada |
| Exceções/importações de terceiros | Não comprovadas | Pendente | Definir triagem e exclusões quando necessário |
| Atribuição | URLs por artigo observadas; manifestos gerados | Parcial | Validar cobertura, links e histórico dos contribuidores |
| Treinamento e uso comercial | Finalidade definida | Pendente de avaliação | Documentar base legal/licença aplicável |
| Distribuição pública de pesos | Finalidade definida | Pendente de avaliação | Analisar ShareAlike, memorizações e termos de publicação |
| Outros datasets já processados | Sem chunks restantes; registro sem aprovação | **Não aprovados para release** | Revisar FineWeb e Wikipedia EN/ES separadamente |
| Aprovação humana | Ausente | Bloqueado | Assinar decisão vinculada ao inventário e evidências |

## 5. Questões jurídicas a decidir

- Se e quando treinamento, pesos e saídas implicam atos sujeitos a direitos autorais em cada jurisdição de distribuição.
- Se as obrigações de CC BY-SA se aplicam a algum artefato distribuído e como cumprir atribuição/ShareAlike. A Creative Commons descreve uma abordagem conservadora de conformidade, mas não uma regra universal de que pesos sejam sempre adaptações.
- Tratamento de trechos importados, citações, conteúdos com licenças alternativas e eventuais direitos não autorais.
- Compatibilidade da licença escolhida para os pesos com o objetivo de permitir uso comercial, modificações e redistribuição.

## 6. Plano de encerramento, sem refazer auditorias técnicas

1. **Verificar upstream:** consultar revisão exata, README e metadados da configuração `20231101.pt`, guardar JSON/URL/data/hash da evidência.
2. **Resolver divergência de licença:** documentar quais termos se aplicam ao snapshot de 2023, incluindo exceções e eventuais condições de 3.0/4.0/GFDL.
3. **Validar atribuição:** consolidar manifestos de artigo e verificar a cobertura e o formato de URLs; publicar `NOTICE` revisado conforme avaliação.
4. **Revisar exceções:** triagem de textos importados e material de terceiros, com exclusão/quarentena quando necessário.
5. **Avaliar os outros cinco datasets:** `PASS` por ausência de chunks pendentes não é aprovação para lançamento do modelo.
6. **Revisão humana independente:** emitir decisão assinada com escopo, justificativa, evidências, inventário SHA-256 e data. Só então atualizar `license_registry.json` e gerar `license_approval.json` conforme o contrato real do notebook.
7. **Reexecutar apenas o gate de licenças**, sem tocar em SQLite v5, shards ou estado original.

## 7. Critérios de aceite

**Para conclusão da auditoria técnica:** 650/650, sumário arquivado e manifestos disponíveis — **atingido segundo a saída fornecida**.

**Para liberar processamento:** licença e exceções avaliadas; decisão humana explícita; registro e aprovação consistentes com o inventário; auditoria do notebook retorna PASS — **não atingido**.

**Para publicar pesos sem restrições de finalidade:** revisão de direitos de todas as fontes do treinamento, documentação de atribuição, termos de licença dos pesos, riscos de memorizações e eventual revisão jurídica — **não atingido**.

## 8. Decisão provisória

**NO-GO para autorização automática de produção e para publicação pública de pesos com alegação de liberdade irrestrita.**

**GO para documentação, verificação upstream, revisão de exceções e preparação de atribuições.**

Não editar artificialmente `rights_status`, `eligible_for_training`, o resultado `LICENSE AUDIT`, nem criar um arquivo de aprovação fictício.

## 9. Referências

- Hugging Face, Wikimedia Wikipedia dataset: https://huggingface.co/datasets/wikimedia/wikipedia
- Hugging Face, árvore histórica `20231101.pt`: https://huggingface.co/datasets/wikimedia/wikipedia/tree/614d2de0683a26dc60e9ca3fcb4a14a5da94bf55/20231101.pt
- Wikimedia Foundation, Terms of Use: https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use/pt
- Creative Commons, Using CC-licensed Works for AI Training: https://creativecommons.org/using-cc-licensed-works-for-ai-training-2/
- Creative Commons, Understanding CC Licenses and AI Training: https://creativecommons.org/2025/05/15/understanding-cc-licenses-and-ai-training-a-legal-primer/

*Relatório elaborado com base nas saídas de auditoria fornecidas pelo operador e consulta documental pública. A verificação upstream exata e a aprovação jurídica continuam pendentes.*
