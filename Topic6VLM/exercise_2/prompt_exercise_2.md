# Task: "Exercise 2: Video-Surveillance Agent"

The task description is in refences and copied bellow.

## References:
### Task specific context:
- [/Topic6VLM/README.md](/Topic6VLM/README.md): The unfinished readme for Topic6VLM. No need to edit this but it has info on setup that has been repeated for all units.
- [/Topic6VLM/exercise_1/configs/llava_agent.json](/Topic6VLM/exercise_1/configs/llava_agent.json): The config used to prompt llava in exercise 1.
- [/Topic6VLM/instructions/vlm.md](/Topic6VLM/instructions/vlm.md): The class module instructions copied into an md file. You are working on "Exercise 2: Video-Surveillance Agent". This is the most important.

### Exampl AI Agents from other projects:
- [/Topic6VLM/exercise_1/README.md](/Topic6VLM/exercise_1/README.md): Exercise 1 solution.
- [/demos/image_use/agent_state_obs_api/README.md](/demos/image_use/agent_state_obs_api/README.md): A demo program that can already send images to ollama hosted models using the openai api. This demo is copied into `Topic6VLM/exercise_2` as starter code.
- [/demos/image_use/agent_state_obs_api/agent_state_obs_api_server/config/qwen_server.json](/demos/image_use/agent_state_obs_api/agent_state_obs_api_server/config/qwen_server.json): The config used to setup the demo to interface with qwen hosted locally on ollama. 

## Description

### Copied from instrucitons

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

### What the coding agent should do

I have included the assignment instruction in my references. We are making a script that lets the user upload a video, then runs through sampled frames of the video asking the model if there is a human, and finally outputs the timestamps where the human enters or exits.

I have already coppied example code into `Topic6VLM/exercise_2/agent_state_obs_api`. You should not need to edit the agent code in `Topic6VLM/exercise_2/agent_state_obs_api/agent_state_obs_api_agent`. Replace the networking layers so that the methods can be called directly without passing over network. Create the non-netwrok wrapper in `Topic6VLM/exercise_2/agent_state_obs_api/agent_state_obs_api` add a llava config similar to `Topic6VLM/exercise_1/configs/llava_agent.json`. Make the llava config default and just allow the user the config path to be passed through to initialize the object.

Finally in `Topic6VLM/exercise_2/video_serveillence_agent` including a file with the video parsing util, a file that runs the core loop through frames logic, and a launch where the user can specify a config path for the agent and a video path. Thus the video is parsed, we loop through the frames and record the time stamps where the `has_human` state changes, and the print to the cli the list of time stamps and if the human enters or exits in that frame.

---
