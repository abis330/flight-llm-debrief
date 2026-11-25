"""``flightdebrief`` command line."""
from __future__ import annotations

import json

import typer

app = typer.Typer(add_completion=False, help="LLM flight debriefs: dataset, baselines, evaluation")


@app.command("build-dataset")
def build_dataset(lake: str = "data/lake", out: str = "data/sft"):
    from .dataset import build

    typer.echo(json.dumps(build(lake, out)))


@app.command()
def baselines(data: str = "data/sft", out: str = "results"):
    """Evaluate oracle / majority / message-rules / GBM baselines on the test split."""
    from .baselines import GBMBaseline, majority, message_rules, oracle
    from .evaluate import evaluate, load_jsonl

    test, train = load_jsonl(f"{data}/test.jsonl"), load_jsonl(f"{data}/train.jsonl")
    gbm = GBMBaseline().fit(train)
    for name, g in [("oracle_target", oracle), ("majority", majority), ("message_rules", message_rules),
                    ("gbm_deviation_features", gbm)]:
        r = evaluate(test, g, name, out)
        typer.echo(f"{name}: {json.dumps({k: r[k] for k in ('full', 'silent')})}")


@app.command("eval-model")
def eval_model(backend: str = "hf", model: str = "Qwen/Qwen2.5-1.5B-Instruct", adapter: str = None,
               url: str = "http://localhost:8000/v1", data: str = "data/sft", out: str = "results",
               name: str = "finetuned", limit: int = 0):
    """Evaluate a model (hf | openai | anthropic) on the test split."""
    from .evaluate import evaluate, load_jsonl
    from .infer import AnthropicGenerator, HFGenerator, OpenAICompatGenerator

    test = load_jsonl(f"{data}/test.jsonl")
    test = test[:limit] if limit else test
    g = {"hf": lambda: HFGenerator(model, adapter), "openai": lambda: OpenAICompatGenerator(url, model),
         "anthropic": lambda: AnthropicGenerator(model if model.startswith("claude") else None,
                                                 load_jsonl(f"{data}/train.jsonl")[:2])}[backend]()
    r = evaluate(test, g, name, out)
    typer.echo(json.dumps(r, indent=2))


@app.command()
def debrief(flight_id: str, lake: str = "data/lake", backend: str = "anthropic", model: str = None,
            adapter: str = None, baseline: str = "data/sft/fleet_baseline.json"):
    """Generate a debrief for one flight."""
    from .infer import AnthropicGenerator, HFGenerator, OpenAICompatGenerator, debrief_flight

    g = {"hf": lambda: HFGenerator(model, adapter), "openai": lambda: OpenAICompatGenerator(model=model),
         "anthropic": lambda: AnthropicGenerator(model)}[backend]()
    typer.echo(json.dumps(debrief_flight(lake, flight_id, g, baseline), indent=2))


if __name__ == "__main__":
    app()
