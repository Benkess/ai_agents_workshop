# Enviroment Usage


## load `.env` inside Python

Install:

```bash
pip install python-dotenv
```

Then at the top of your script:

```python
from dotenv import load_dotenv
load_dotenv()
```

Now:

```
.env
```

will automatically populate environment variables.

So the agent can just run:

```bash
python script.py
```

and it works.


---

## Best workflow for what you want

You said:

> That way I can just tell it when I want it to have my api keys

Do this:

### 1️⃣ Keep secrets in `.env`

```
ASTA_API_KEY=xxxx
OPENAI_API_KEY=xxxx
```

### 2️⃣ Load automatically in Python

```python
from dotenv import load_dotenv
load_dotenv()
```

### 3️⃣ Tell the agent

Example prompt:

> The project uses `.env` for secrets. Python scripts automatically load it using `python-dotenv`.

Then the agent can run scripts normally.


---

## Workflow 2

Here’s a cross-platform runner that:

* loads `.env` if present
* uses the repo’s `.venv` if present
* falls back to the current Python if no `.venv` exists
* runs the target script with any extra args

I is saved in [scripts/run_with_env.py](/scripts/run_with_env.py)

Install the one dependency in your venv:

```bash
pip install python-dotenv
```

Then run scripts like this:

```bash
python scripts/run_with_env.py path/to/your_script.py
```

Example:

```bash
python scripts/run_with_env.py Topic7MCP/mcp/exercise_b/exercise_b_save_json.py
```

If your script takes arguments:

```bash
python scripts/run_with_env.py app/main.py --input data.json --verbose
```

And your `.env` can just sit at the repo root:

```env
ASTA_API_KEY=your_key_here
OPENAI_API_KEY=your_other_key
```

A good `AGENTS.md` note for Codex would be:

```md
- Environment variables are stored in `.env` at the repo root.
- Run Python scripts with:
  `python scripts/run_with_env.py <script> [args...]`
- Do not rely on shell activation persisting across commands.
```

One small note: this helps only for **Python scripts launched through this runner**. If the agent runs some other command directly, like `pytest` or a shell tool, those won’t automatically get `.env` unless you wrap them too.
