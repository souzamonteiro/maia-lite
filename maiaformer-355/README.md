# MaiaFormer-355 — protótipo arquitetural v0.1

**Objetivo:** testar uma arquitetura decoder-only com operações equivalentes às do Llama antes de investir em treinamento de bilhões de tokens. **Não é ainda um modelo treinado ou um conversor GGUF validado.**

## Especificação provisória

- 29 camadas, `hidden_size=896`, `intermediate_size=3072` (SwiGLU).
- 14 cabeças Q, 2 cabeças KV, dimensão por cabeça 64 (GQA).
- RoPE, RMSNorm, embeddings compartilhados, contexto 2048.
- Vocabulário provisório 65.536; escolha definitiva do tokenizer pendente.
- Atenção global em todas as camadas. Atenção híbrida local/global adiada para preservar compatibilidade de runtime.

## Testes locais

```bash
python maiaformer.py --tiny
python maiaformer.py --write-config config.json
```

O primeiro comando executa forward/backward num modelo reduzido; o segundo calcula a contagem analítica de parâmetros e grava uma configuração de referência no formato Hugging Face Llama. O protótipo usa PyTorch, sem exigir `transformers` para esses testes.

## Próximos gates (ainda não executados)

1. **Paridade PyTorch ↔ Hugging Face:** instalar versão fixada de `transformers`, instanciar `LlamaForCausalLM`, mapear os pesos do protótipo e comparar logits para IDs idênticos, especialmente RoPE e RMSNorm. A implementação atual é apenas uma aproximação operacional da arquitetura Llama; a paridade precisa ser comprovada.
2. **Conversão GGUF:** salvar modelo e tokenizer compatíveis em `save_pretrained`, executar `convert_hf_to_gguf.py` de uma revisão fixada do llama.cpp e comparar logits com a referência PyTorch. Testar no Ollama.
3. **Treinamento piloto:** alguns milhões de tokens; checar estabilidade, loss, throughput e exportação de pesos atualizados.
4. **Pré-treino 7,1B tokens:** somente após auditoria do corpus existente, licenças e decisão do tokenizer. Se o tokenizer mudar, retokenizar o texto bruto; os IDs antigos não são intercambiáveis.
5. **Continuação até 30B acumulados:** novo corpus deduplicado, com replay controlado e checkpoints de avaliação.

## Cuidados

- A configuração `bos_token_id=1`, `eos_token_id=2` é **placeholder**: substituir pelos IDs reais do tokenizer selecionado antes de exportar.
- A contagem de parâmetros depende de vocabulário e tying. A meta é **aproximadamente** 355M, não um valor exato obrigatório.
- A implementação de atenção atual expande KV via `repeat_interleave`, adequada à validação, não necessariamente à melhor eficiência de treino.
- Checkpoints de treino devem guardar pesos, otimizador, scheduler, RNG, contador de tokens, cursor/amostragem e hashes dos dados.
