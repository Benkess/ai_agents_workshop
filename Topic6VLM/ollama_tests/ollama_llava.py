import ollama
response = ollama.chat(
    model='llava',
    messages=[{
        'role': 'user',
        'content': 'Describe this image in English.',
        'images': ['Topic6VLM/ollama_tests/photo.jpg']
    }]
)
print(response['message']['content'])