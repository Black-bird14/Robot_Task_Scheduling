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
from Vild import vild
import constants as c
from CoordinateExtraction import describe
from feasibility import *
#import anvil.server
#anvil.server.connect("server_O7GBNJ5AD37EMDSHYB26YTMW-TDYOSJG77QVTSKKN")

task_feasibility={}
#@anvil.server.callable
def send_description():
  """Sends environment description to anvil front end"""
  found_objects, _ = vild(c.IMAGE_PATH, c.CATEGORY_NAME_STRING, c.VILD_PARAMS, plot_on=True, prompt_swaps=c.PROMPT_SWAP)
  scene_description = describe(found_objects)
  return scene_description


#@anvil.server.callable
def set_env():
  """Define and reset environment"""
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
  """Main runner function, combines all the necessary functionalities"""
  global task_feasibility
  obs, _ =set_env()
  # Get the user's home directory
  home_dir = os.path.expanduser("~")
  
  # Construct the full path to the JSON file
  filename = os.path.join(home_dir, 'Downloads', 'sorted_tasks.json')
  
  # Load the JSON file
  with open(filename, 'r') as json_file:
    sorted_tasks = json.load(json_file)
  
  tasks_steps={}
  # Extract task name and steps for each task
  for task_name, task_info in sorted_tasks.items():
    steps=task_info["steps"]
    tasks_steps[task_info["name"]]=steps

  for task, instructions in tasks_steps.items():
    print("Task:", task)
    print("Steps:", instructions)
    for instruction in instructions:
      pickxyz, placexyz=run (obs, instruction)
      pickxy.append([pickxyz[0],pickxyz[1]])
      placexy.append([placexyz[0], placexyz[1]])
      time.sleep(20)
    task_feasibility[task]=check_feasibility(task, instructions, scene_description, pickxy, placexy)
    obs, _= set_env()

if __name__ == '__main__': 
  blender_()

#anvil.server.wait_forever()