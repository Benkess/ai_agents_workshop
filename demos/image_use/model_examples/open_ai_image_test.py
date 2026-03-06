import argparse

import base64
import mimetypes
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

def image_file_to_base64_and_mime(image_path: str) -> tuple[str, str]:
    path = Path(image_path)
    mime_type, _ = mimetypes.guess_type(str(path))
    if mime_type is None:
        # Fallback if extension is unknown
        mime_type = "application/octet-stream"

    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    return b64, mime_type

def main():
    parser = argparse.ArgumentParser(description="Send an image to a vision-capable OpenAI model.")
    parser.add_argument("image_path", help="Path to the image file to analyze")
    parser.add_argument("--prompt", nargs="?", default="Describe what’s on this screen and point out any errors.", help="The prompt to accompany the image (default: 'Describe what’s on this screen and point out any errors.')")
    args = parser.parse_args()

    llm = ChatOpenAI(model="gpt-4o-mini")  # pick a vision-capable OpenAI model

    img_b64, img_mime = image_file_to_base64_and_mime(args.image_path)
    user_prompt = args.prompt

    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(
            content_blocks=[
                {"type": "text", "text": user_prompt},
                {"type": "image", "base64": img_b64, "mime_type": img_mime},
            ]
        ),
    ]

    response = llm.invoke(messages)
    print(response.content)


if __name__ == "__main__":
    main()