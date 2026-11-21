# Maia Lite — Avaliação de chat, língua e lógica

- Assinatura: `c4dcdbd48dfc`
- Tarefas: 42
- Respostas no formato nativo: 168
- Revisão humana: **pendente**

## Como ler os resultados

Comparar `native_summary.csv` para capacidade conversacional no formato de cada adaptador.
Usar `format_stress_summary.csv` somente para robustez ao delimitador.
Verificar respostas em `generations_scored.csv` e preencher `human_review_blind.csv`.
Não promover uma versão com base apenas em EOS, loss ou acerto exato.

## Limitações

- Sem avaliação de matemática, programação ou conhecimento especializado.
- Acertos exatos não medem compreensão profunda; paráfrases corretas podem ser penalizadas.
- Repetição e EOS não medem qualidade factual ou utilidade.
- Teste de contexto é transcrição textual, não memória persistente.
- O formato nativo difere entre v2.2 e v2.3; comparar ambos como sistemas completos.
- Não inferir melhora estatística sem múltiplas amostras e intervalos de confiança.
- Amostra rápida é somente teste funcional; não é benchmark definitivo.