import requests, os

headers = {
    "Content-Type": "application/json",
    "x-api-key": os.environ["ASTA_API_KEY"]
}
payload = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/list",
    "params": {}
}
resp = requests.post(
    "https://asta-tools.allen.ai/mcp/v1",
    headers=headers,
    json=payload
)
tools = resp.json()["result"]["tools"]
# Your printing logic here