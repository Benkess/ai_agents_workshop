import argparse
import requests
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Send a POST request to /task endpoint"
    )
    
    parser.add_argument(
        "--url",
        required=True,
        help="Base ngrok URL (e.g., https://abc123.ngrok-free.app)"
    )

    parser.add_argument(
        "--question",
        default="What year was the first Super Bowl?",
        help="Question to send"
    )

    parser.add_argument(
        "--sender",
        default="test",
        help="Sender name"
    )

    args = parser.parse_args()

    # Ensure no trailing slash issues
    base_url = args.url.rstrip("/")
    endpoint = f"{base_url}/task"

    payload = {
        "question": args.question,
        "sender": args.sender
    }

    try:
        response = requests.post(endpoint, json=payload)
        response.raise_for_status()

        print("Status:", response.status_code)
        print("Response:", response.json())

    except requests.exceptions.RequestException as e:
        print("Request failed:", e)
        sys.exit(1)


if __name__ == "__main__":
    main()