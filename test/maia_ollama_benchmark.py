#!/usr/bin/env python3
"""Compare Maia Lite, Qwen 2.5 0.5B and Gemma 3 270M via Ollama's chat API.

Uses only Python's standard library. Each request has a fresh, isolated context.
"""
import argparse
import datetime as dt
import json
import os
import pathlib
import sys
import time
import urllib.error
import urllib.request

DEFAULT_PROMPTS = [
    "Escreva um único parágrafo, em português, com aproximadamente 100 palavras, explicando o que é a solidão. Não utilize listas.",
    "Explique, em português e de forma simples, a diferença entre solidão e solitude.",
    "Se Ana tem 12 livros, empresta 5 e compra mais 3, quantos livros ela possui? Mostre a conta.",
    "Escreva uma função Python chamada fibonacci(n) que retorne o n-ésimo número de Fibonacci. Inclua um exemplo.",
    "Explique em português o que é overfitting em aprendizado de máquina e apresente uma forma de reduzi-lo.",
]


def api_request(base_url, endpoint, payload=None, timeout=180):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + endpoint,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if payload is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
    parser.add_argument("--models", nargs="+", default=["maia-lite", "qwen2.5:0.5b", "gemma3:270m"])
    parser.add_argument("--maia", help="Override the Maia model name in --models")
    parser.add_argument("--prompt", action="append", help="Custom prompt; repeat for several prompts")
    parser.add_argument("--prompts-file", help="UTF-8 file with one prompt per non-empty line")
    parser.add_argument("--interactive", action="store_true", help="Read prompts from terminal until blank input")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repeat-penalty", type=float, default=1.1)
    parser.add_argument("--num-predict", type=int, default=256)
    parser.add_argument("--num-ctx", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--system", default="", help="Optional identical system message for every model (default: none)")
    parser.add_argument("--output", default="maia_ollama_benchmark.jsonl")
    args = parser.parse_args()

    if args.maia:
        args.models[0] = args.maia
    prompts = list(args.prompt or [])
    if args.prompts_file:
        with open(args.prompts_file, encoding="utf-8") as file:
            prompts.extend(line.strip() for line in file if line.strip())
    if args.interactive:
        print("Digite um prompt (linha vazia encerra):", flush=True)
        while True:
            try:
                prompt = input("prompt> ").strip()
            except EOFError:
                break
            if not prompt:
                break
            prompts.append(prompt)
    if not prompts:
        prompts = DEFAULT_PROMPTS

    try:
        tags = api_request(args.url, "/api/tags", timeout=15)
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        sys.exit(f"ERRO: Ollama indisponível em {args.url}: {exc}")

    available = {item.get("name", "") for item in tags.get("models", [])}
    print("Modelos instalados:", ", ".join(sorted(available)) or "nenhum", flush=True)
    missing = [model for model in args.models if model not in available]
    if missing:
        sys.exit("ERRO: modelos ausentes: " + ", ".join(missing) + "\nUse: ollama list; ollama pull qwen2.5:0.5b; ollama pull gemma3:270m")

    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    options = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repeat_penalty": args.repeat_penalty,
        "num_predict": args.num_predict,
        "num_ctx": args.num_ctx,
        "seed": args.seed,
    }
    print(f"\n{len(prompts)} prompts x {len(args.models)} modelos. Saída: {output_path}\n", flush=True)
    with output_path.open("a", encoding="utf-8") as output:
        for i, prompt in enumerate(prompts, 1):
            print("=" * 80, f"\nPROMPT {i}/{len(prompts)}: {prompt}\n", flush=True)
            for model in args.models:
                messages = []
                if args.system:
                    messages.append({"role": "system", "content": args.system})
                messages.append({"role": "user", "content": prompt})
                payload = {"model": model, "messages": messages, "stream": False, "options": options, "keep_alive": "5m"}
                started = time.perf_counter()
                record = {
                    "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "model": model,
                    "prompt_number": i,
                    "prompt": prompt,
                    "system": args.system,
                    "options": options,
                }
                try:
                    result = api_request(args.url, "/api/chat", payload, timeout=args.timeout)
                    elapsed = time.perf_counter() - started
                    answer = result.get("message", {}).get("content", "")
                    eval_count = result.get("eval_count")
                    eval_duration = result.get("eval_duration")
                    tokens_per_second = (eval_count * 1e9 / eval_duration) if eval_count and eval_duration else None
                    record.update({
                        "answer": answer,
                        "elapsed_seconds": round(elapsed, 3),
                        "eval_count": eval_count,
                        "eval_duration_ns": eval_duration,
                        "tokens_per_second": round(tokens_per_second, 2) if tokens_per_second else None,
                        "done_reason": result.get("done_reason"),
                    })
                    print(f"--- {model} | {elapsed:.1f}s | {tokens_per_second:.1f} tok/s ---" if tokens_per_second else f"--- {model} | {elapsed:.1f}s ---", flush=True)
                    print(answer or "[RESPOSTA VAZIA]", "\n", flush=True)
                except (urllib.error.URLError, TimeoutError, ValueError) as exc:
                    record["error"] = str(exc)
                    print(f"--- {model} | ERRO: {exc} ---\n", flush=True)
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()
    print(f"Resultados salvos em {output_path}", flush=True)


if __name__ == "__main__":
    main()
