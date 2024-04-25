import numpy as np
from Vild import vild
import constants as c
import imageio
from moviepy.editor import ImageSequenceClip
from PIL import Image
import spacy
import matplotlib.pyplot as plt
from SimulatedEnv import env


task_counter = 0

def describe(found_objects:dict):
    """Create Environment description(list of objects found in the environment)"""
    scene_description = f"objects = {found_objects}"
    scene_description = scene_description.replace("'", "")
    return scene_description


def img_reset():
    """Reset image to match current environment state"""
    image = env.get_camera_image_top()
    image = np.flipud(image.transpose(1, 0, 2))
    imageio.imwrite('imge.jpg', image)


def parse_instruction(instruction):
    """Identify pick target and place location using a Natural Language Understanding technique"""

    # Download the English model if it's not already downloaded
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading the English model...")
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    doc = nlp(instruction)
    objects = []
    destinations = []
    on_the_or_in_the = False
    for token in doc:
        if token.lower_ in ["on", "in"]:
            on_the_or_in_the = True
        elif token.pos_ == "NOUN" or token.pos_ == "ADJ" or token.pos_ == "PROPN":
            if on_the_or_in_the:
                destinations.append(token.text.lower())
            else:
                objects.append(token.text.lower())

    # If objects list is empty, assume the first detected phrase is the destination
    if not objects and destinations:
        objects.append(destinations.pop(0))

    return " ".join(objects), " ".join(destinations)

def extract_object_coordinates(vild_results, pick_objects, place_destinations):
    """Extract object pixel coordinates using bounding boxes drawn on the image upon running ViLD, along with pick target and place location retrieved from parsing the instruction"""
    # Unpack vild results
    category_names, rescaled_detection_boxes = vild_results

    # Initialize object and place coordinates lists
    object_coordinates = []
    place_coordinates = []

    # Ensure pick_objects and place_destinations are iterable
    if not isinstance(pick_objects, (list, tuple)):
        pick_objects = [pick_objects]
    if not isinstance(place_destinations, (list, tuple)):
        place_destinations = [place_destinations]

    # Find object coordinates
    for pick_object in pick_objects:
        for anno_idx, category_name in enumerate(category_names):
            if category_name == pick_object:
                object_coordinates.append(rescaled_detection_boxes[anno_idx])
                break
        else:
            print("Object '{}' not found in the image.".format(pick_object))

    # Find place coordinates
    for place_destination in place_destinations:
        for anno_idx, category_name in enumerate(category_names):
            if category_name == place_destination:
                place_coordinates.append(rescaled_detection_boxes[anno_idx])
                break
        else:
            print("Place destination '{}' not found in the image.".format(place_destination))
    # Extract center coordinates
    def calculate_center(box):
        y_center = box[0] + (box[2] - box[0]) / 2
        x_center = box[1] + (box[3] - box[1]) / 2
        return y_center, x_center

    pick_yx = object_coordinates[0]
    place_yx = calculate_center(place_coordinates[0]) if place_coordinates else None

    return pick_yx, place_yx

def run(obs, instruction):
    """Combining Instruction Parsing, Coordinate Extraction, and Execution"""
    global task_counter
    key_words=["top", "bottom", "middle"]
    before = env.get_camera_image()
    prev_obs = obs['image'].copy()
    
    #ViLD Execution + Coordinate extraction
    vild_results = vild(c.IMAGE_PATH, c.CATEGORY_NAME_STRING, c.VILD_PARAMS, plot_on=True, prompt_swaps=c.PROMPT_SWAP)
    found_objects, _= vild_results

    pick_object, place_destination = parse_instruction(instruction)
    print("Pick object:", pick_object)
    print("Place target:", place_destination)
    # Find the index of the first space character
    first_space_index = place_destination.find(' ')
    
    # Extract the first word from the string
    if first_space_index != -1:
        first_word = place_destination[:first_space_index]
    else:
        first_word = place_destination#
    
    if first_word in key_words:
        place_xyz= np.array(c.PLACE_TARGETS[place_destination])
        pick_yx, _ = extract_object_coordinates(vild_results, pick_object, place_destination)
        
        if pick_yx is not None:
          print("Pick YX coordinates:", pick_yx)     
        pick_xyz = obs['xyzmap'][int(pick_yx[0]), int(pick_yx[1])]
        print("Pick position (x, y, z):", pick_xyz)
        print("Place position (x, y, z):", place_xyz)
    else:
        pick_yx, place_yx = extract_object_coordinates(vild_results, pick_object, place_destination)
        if pick_yx is not None:
          print("Pick YX coordinates:", pick_yx)
        if place_yx is not None:
          print("Place YX coordinates:", place_yx)
        pick_xyz = obs['xyzmap'][int(pick_yx[0]), int(pick_yx[1])]
        place_xyz = obs['xyzmap'][int(place_yx[0]), int(place_yx[1])]
        print("Pick position (x, y, z):", pick_xyz)
        print("Place position (x, y, z):", place_xyz)

        # Show pick and place action.
        plt.title(instruction)
        plt.imshow(prev_obs)
        plt.arrow(pick_yx[1], pick_yx[0], place_yx[1]-pick_yx[1], place_yx[0]-pick_yx[0], color='w', head_starts_at_zero=False, head_width=7, length_includes_head=True)
        plt.show()
    
    act = {'pick': pick_xyz, 'place': place_xyz}
    obs, _, _, _ = env.step(act)
    
    # Show video of environment rollout.
    debug_clip = ImageSequenceClip(env.cache_video, fps=25)
    task_counter+=1
    debug_clip.write_videofile(f"Task{task_counter}.mp4")
    #display(debug_clip.ipython_display(autoplay=1, loop=1, center=False))
    env.cache_video = []
    
    # Show camera image after pick and place.
    plt.subplot(1, 2, 1)
    plt.title('Before')
    plt.imshow(before)
    plt.subplot(1, 2, 2)
    plt.title('After')
    after = env.get_camera_image()
    plt.imshow(after)
    plt.show()
    img_reset()
    


