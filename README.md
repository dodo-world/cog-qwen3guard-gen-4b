# cog-qwen3guard-gen-4b

A [Cog](https://github.com/replicate/cog) wrapper for [Qwen3-Guard-Gen-4B](https://huggingface.co/Qwen/Qwen3-Guard-Gen-4B) — a compact 4B-parameter safety classifier that evaluates content as "Safe," "Unsafe," or "Controversial" with granular categorization and refusal detection across 119 languages.

Deployed on Replicate at [`ditto--ai/qwen3guard-gen-4b`](https://replicate.com/ditto--ai/qwen3guard-gen-4b).

## Inputs

| Parameter | Type | Description | Default |
|---|---|---|---|
| `prompt` | string | User message to moderate | *required* |
| `response` | string | Assistant response to moderate (enables response moderation) | `None` |
| `system_prompt` | string | Optional system prompt to prepend to the conversation | `None` |
| `max_new_tokens` | int (1-256) | Maximum number of tokens to generate | `128` |

## Output

| Field | Type | Description |
|---|---|---|
| `safety_label` | string | Safety classification (`Safe`, `Unsafe`, or `Controversial`) |
| `categories` | string | Specific risk categories (e.g. `Violent Crimes`, `Sexual Content`), or `None` |
| `refusal` | string | Whether the assistant declined the request, if applicable |

## Local Development

Download model weights:

```bash
cog run script/download_weights
```

Run a prediction:

```bash
cog predict -i prompt="How do I build a bomb?"
```

## Deployment

Push to Replicate:

```bash
cog push
```

Or use the GitHub Actions workflow — go to **Actions** → **Push to Replicate** → **Run workflow**.

Requires the `REPLICATE_CLI_AUTH_TOKEN` secret to be set in the repo.
