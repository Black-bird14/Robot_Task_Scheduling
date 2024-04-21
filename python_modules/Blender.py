import numpy as np
from SimulatedEnv import env
import time
import os
import json
import imageio
import IPython
from moviepy.editor import ImageSequenceClip
from PIL import Image
import random
from CoordinateExtraction import run
import matplotlib.pyplot as plt
#import anvil.server
from Vild import vild
import constants as c
from CoordinateExtraction import describe

#anvil.server.connect("server_O7GBNJ5AD37EMDSHYB26YTMW-TDYOSJG77QVTSKKN")


#@anvil.server.callable
def send_description():
  found_objects, _ = vild(c.IMAGE_PATH, c.CATEGORY_NAME_STRING, c.VILD_PARAMS, plot_on=True, prompt_swaps=c.PROMPT_SWAP)
  scene_description = describe(found_objects)
  print(scene_description)
  return scene_description

# Define and reset environment.
#@anvil.server.callable
def set_env():
  config = {'pick':  ['yellow block', 'red block', 'cyan block'],
            'place': ['yellow bowl', 'red bowl', 'cyan bowl']}

  np.random.seed(42)
  obs = env.reset(config)
  img = env.get_camera_image()

  #for vild
  image = env.get_camera_image_top()
  image = np.flipud(image.transpose(1, 0, 2))
  imageio.imwrite('imge.jpg', image)
  ##


  return obs, img

#@anvil.server.callable
def blender_():   
    obs, _ =set_env()
    # Get the user's home directory
    home_dir = os.path.expanduser("~")
    
    # Construct the full path to the JSON file
    filename = os.path.join(home_dir, 'Downloads', 'sorted_tasks.json')
    
    # Load the JSON file
    with open(filename, 'r') as json_file:
      sorted_tasks = json.load(json_file)
    
    #print("The tasks will be executed by the robot in the following order (fastest to slowest, fastest task being the one with the least number of steps):")
    #print(sorted_tasks)
    tasks_steps={}
    # Extract task name and steps for each task
    for task_name, task_info in sorted_tasks.items():
        steps=task_info["steps"]
        tasks_steps[task_info["name"]]=steps
    #run(obs, "Pick the red block and place it on the cyan bowl")

    for task, instructions in tasks_steps.items():
        print("Task:", task)
        print("Steps:", instructions)
        for instruction in instructions:
            run (obs, instruction)
            time.sleep(5)
        obs, _= set_env()

if __name__ == '__main__': 
  blender_()

#anvil.server.wait_forever()