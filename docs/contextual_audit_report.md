# Maia Lite — Auditoria contextual v2

- Amostra analisada: **420** exemplos (amostra enriquecida por risco; não representativa do corpus completo).
- Suspeitas de contexto ausente: **5**.
- Sinais de repetição: **143**.
- Ligações lexicais exploratórias com benchmark: **133**.
- Prioridade alta: **20**; média: **221**; baixa: **179**.

## Interpretação

A triagem não confirma erros factuais, não estima a prevalência de problemas no corpus completo e não estabelece causalidade entre exemplos e gerações do benchmark. Revisar a fila e registrar evidência antes de corrigir ou excluir dados.

## Decisão

**Não treinar v2.4 automaticamente.** Realizar revisão humana dos itens prioritários, documentar fontes, e só então gerar uma proposta de patch versionado do corpus.
