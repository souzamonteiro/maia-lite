# Auditoria direcionada — resultados preliminares

- Corpus: 29995 treino, 1521 validação.
- Amostra para revisão: 420 exemplos.
- **Revisão factual humana: PENDENTE**. Os sinais automáticos não provam erros.

## Integridade
- exact_prompt_overlap_train_val: 0
- exact_pair_overlap_train_val: 0
- duplicate_ids: 0
- empty_prompt: 0
- empty_completion: 0
- group_id_overlap_train_val: 0

## Sinais automáticos (contagem, não erros confirmados)
- empty: 0
- very_short: 4438
- long_answer: 2056
- repeated_trigrams: 826
- starts_as_question: 185
- prompt_echo: 3
- numeric_not_in_prompt: 13750
- unqualified_absolute: 927

## Arquivos
- human_review_corpus.csv — formulário de avaliação
- risk_signals.csv — explicações dos sinais
- benchmark_cases_prioritized.csv — casos do benchmark (se fornecido)
- audit_summary.json — resumo reproduzível

## Decisão
Não promover nem treinar v2.4 com base apenas nesta triagem. Revisar manualmente os exemplos e documentar evidências.