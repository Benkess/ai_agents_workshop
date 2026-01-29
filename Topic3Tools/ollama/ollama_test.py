import requests
import json
import time
import argparse

def call_ollama(prompt, model="llama3.2:1b", timeout=120, retries=3):
    url = "http://localhost:11434/api/generate"
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(
                url,
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=timeout,
            )
            last_exc = None
            break
        except requests.RequestException as e:
            last_exc = e
            if attempt >= retries:
                raise RuntimeError(f"Request failed after {retries} attempts: {e}") from e
            wait = min(5 * attempt, 30)
            time.sleep(wait)
    if last_exc is not None:
        # Shouldn't reach here because we re-raised above, but keep safe
        raise RuntimeError(f"Request failed: {last_exc}") from last_exc

    try:
        data = resp.json()
    except ValueError:
        raise RuntimeError(f"Non-JSON response (status {resp.status_code}): {resp.text!r}")

    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {json.dumps(data, indent=2)}")

    # Accept common possible shapes returned by Ollama / proxies
    if isinstance(data, dict):
        for k in ("response", "text", "output"):
            if k in data:
                return data[k]
        if "results" in data and isinstance(data["results"], list) and data["results"]:
            first = data["results"][0]
            if isinstance(first, dict):
                for k in ("content", "text", "response"):
                    if k in first:
                        return first[k]
            return first
        if "responses" in data and data["responses"]:
            return data["responses"][0]
    return json.dumps(data, indent=2)

def _main():
    p = argparse.ArgumentParser(description="Call local Ollama HTTP API")
    p.add_argument("prompt", nargs="+", help="Prompt to send to the model")
    p.add_argument("--model", default="llama3.2:1b")
    p.add_argument("--timeout", type=int, default=120, help="Read timeout in seconds")
    p.add_argument("--retries", type=int, default=3, help="Number of attempts")
    args = p.parse_args()
    prompt = " ".join(args.prompt)
    try:
        result = call_ollama(prompt, model=args.model, timeout=args.timeout, retries=args.retries)
        print(result)
    except RuntimeError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    _main()