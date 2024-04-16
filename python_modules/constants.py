from easydict import EasyDict
import numpy as np


#Global constants: pick and place objects, colors, workspace bounds
PICK_TARGETS = {
  "cyan block": None,
  "red block": None,
  "green block": None,
  "yellow block": None,
  "purple block": None,
}

COLOURS = {
    "blue":   (78/255,  121/255, 167/255, 255/255),
    "red":    (240/255, 26/255, 55/255,255/255),
    "green":  (0/255, 153/255, 76/255, 255/255),
    "yellow": (243/255, 178/255,  18/255, 255/255),
    "purple": (142/255, 107/255, 181/255, 255/255),
    "cyan": (1/255, 160/255, 157/255, 255/255),
}

PLACE_TARGETS = {
  "cyan block": None,
  "red block": None,
  "green block": None,
  "yellow block": None,
  "purple block": None,

  "cyan bowl": None,
  "red bowl": None,
  "green bowl": None,
  "yellow bowl": None,
  "purple bowl": None,

  "top left corner":     (-0.3 + 0.05, -0.2 - 0.05, 0),
  "top right corner":    (0.3 - 0.05,  -0.2 - 0.05, 0),
  "middle":              (0,           -0.5,        0),
  "bottom left corner":  (-0.3 + 0.05, -0.8 + 0.05, 0),
  "bottom right corner": (0.3 - 0.05,  -0.8 + 0.05, 0),
}


PIXEL_SIZE = 0.00267857
BOUNDS = np.float32([[-0.3, 0.3], [-0.8, -0.2], [0, 0.15]])  # X Y Z

###VILD HYPERPARAMETERS
FLAGS = {
    'prompt_engineering': True,
    'this_is': True,
    'temperature': 100.0,
    'use_softmax': False,
}
FLAGS = EasyDict(FLAGS)


# Define parameters for running vild function
CATEGORY_NAMES = ['blue block',
                  'red block',
                  'green block',
                  'orange block',
                  'yellow block',
                  'purple block',
                  'pink block',
                  'cyan block',
                  'brown block',
                  'gray block',

                  'blue bowl',
                  'red bowl',
                  'green bowl',
                  'orange bowl',
                  'yellow bowl',
                  'purple bowl',
                  'pink bowl',
                  'cyan bowl',
                  'brown bowl',
                  'gray bowl']
IMAGE_PATH = 'imge.jpg'

#@markdown ViLD settings.
CATEGORY_NAME_STRING = ";".join(CATEGORY_NAMES)
MAX_BOXES_TO_DRAW = 8 #@param {type:"integer"}

# Extra prompt engineering: swap A with B for every (A, B) in list.
PROMPT_SWAP = [('block', 'cube')]

NMS_THRESHOLD = 0.4 #@param {type:"slider", min:0, max:0.9, step:0.05}
MIN_RPN_SCORE_THRESH = 0.4  #@param {type:"slider", min:0, max:1, step:0.01}
MIN_BOX_AREA = 10 #@param {type:"slider", min:0, max:10000, step:1.0}
MAX_BOX_AREA = 3000  #@param {type:"slider", min:0, max:10000, step:1.0}
VILD_PARAMS = MAX_BOXES_TO_DRAW, NMS_THRESHOLD, MIN_RPN_SCORE_THRESH, MIN_BOX_AREA, MAX_BOX_AREA