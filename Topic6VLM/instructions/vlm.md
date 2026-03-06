# Topic 6: Vision-Language Models (VLM)

![LLaVA Show](vlm.jpg)

### Learning Goals

- Know the parts of a basic VLM pipeline:
  - How contrastive pre-training on image-caption pairs creates a pair of encoders where the language and images vectors for an entity are aligned
  - How a projection layer aligns the dimensions of the visual encodings with those of the token embeddings of a language model
  - How the LLM mixes image and language tokens
- Be able to build an agent that can answer questions about supplied images
- Be able to build an agent that handles video input by extracting keyframes and suppling them to a VLM
- Have a high-level understanding of how image generation works

### Tasks

You will be writing your own code for the exercises using Ollama and the llava open vision-language model. It can run on a MacBook or PC with a GPU or Colab.  To start up Ollama, use the shell command

```
ollama pull llava
```

Ollama handles selecting the appropriate GPU and quantization level for all platforms. 

This Python snippet shows how to run the model:

```
import ollama
response = ollama.chat(
    model='llava',
    messages=[{
        'role': 'user',
        'content': 'Describe this image in English.',
        'images': ['./photo.jpg']
    }]
)
print(response['message']['content'])
```

The `images` key in the message is a list of file paths and/or base64 encoded strings of the file contents. The following snippet turns the contents of a file into a base64 encoded string:

```
import base64
with open("photo.jpg", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode("utf-8")
```

You have several choices for creating the user interfaces for your programs:

- Use [`tkinter`]((https://docs.python.org/3/library/tkinter.html) https://docs.python.org/3/library/tkinter.html) for local Python command-line programs.
- Use [`ipywidgets`](https://pypi.org/project/ipywidgets/) for Jupyter notebooks running locally or on Colab.
- Use the [`gradio`](https://pypi.org/project/gradio/) library for a polished interface that runs in a separate web page and that works with both local Python command-line programs and notebooks running locally or in Colab.

#### Exercise 1: Vision-Language LangGraph Chat Agent

Write a chat agent that can carry on a multi-turn conversation about an image you upload.  Use all that you have learned about good LangGraph style to structure the agent and manage the context.  If the program runs slowly, check if it helps to reduce the resolution of the uploaded image.

#### Exercise 2: Video-Surveillance Agent

LLaVA is not powerful enough to handle video directly.  However, you can still use it for video by extracting frames from the video every few seconds and sending them to LLaVA.

1. Create a 2-minute video clip of an empty space that a person enters and exits at some point.

2. Split in the video into frames that are 2 seconds apart using OpenCV (cv2).  Install it with 

   `pip install opencv-python`

   You do not need to install the full `opencv-contrib-python`.  Here is a code snipped that spits a file and saves it as images:

   ```
   import cv2
   
   cap = cv2.VideoCapture("video.mp4")
   fps = cap.get(cv2.CAP_PROP_FPS)
   interval = int(fps * 2)  # frame interval for ~2 seconds
   
   frames = []
   frame_num = 0
   while cap.isOpened():
       ret, frame = cap.read()
       if not ret:
           break
       if frame_num % interval == 0:
           frames.append(frame)
       frame_num += 1
   
   cap.release()
   
   for i, frame in enumerate(frames):
       cv2.imwrite(f"frame_{i:04d}.jpg", frame)
   ```

3. Write a program that runs through the all the frames with a prompt that asks LLaVA if there is a person in the scene.  Write the times at which a person enters and exits the scene.

4. *Optional*: Connect your program to a webcam and run it in real time.  `cv2.VideoCapture(0)` typically captures a frame from the webcam in your laptop.  Running locally on my MacBook Pro, it takes about 8.5 seconds for LLaVA to process a frame, so you will either need to increase the time between samples and/or run on a platform with more powerful GPUs such as Colab or a GPU-equipped PC.

5. *Optional*: Run the program in a place where there are cats or dogs.  Prompt LLaVA to let you know if there is a person, cat, or dog in the scene and how many of each. Is LLaVA able to count? If it sees people, it should also say INTRUDER ALERT (a good text to speech library is `edge-tts`)but not if it sees a cat or dog.  

### Resources

- [Vision-Language Model Guide](vlm_guide.html)
- [Image Generation Model Guide](image_generation_guide.html)
- [Gradio Quickstart](gradio_quickstart.py)
- [The Illustrated Stable Diffusion, by Jay Alammar](https://jalammar.github.io/illustrated-stable-diffusion/)


### Portfolio

Create a subdirectory in your GitHub portfolio named Topic6VLM and save your programs, each modified version named to indicate its task number and purpose.  Create appropriately named text files saving the outputs from your terminal sessions running the programs.  Create README.md with a table of contents of the directory.