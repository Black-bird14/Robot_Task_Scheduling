#@title Feasibility Scoring
def check_feasibility(task, steps, environment_description, pickxy, placexy):
  feasibility_threshold= 0.4
  feasibility=0
  # Extract the objects substring from env_description
  objects_substring = environment_description.split("objects =")[1]

  # Remove the square brackets and trailing comma from the substring
  objects_substring = objects_substring.strip("[]").rstrip(",")

  # Split the substring into a list of object names
  object_names = [obj.strip() for obj in objects_substring.split(",")]
  accessible_place_destinations=['bottom right corner', 'bottom left corner', 'top right corner', 'top left corner', 'middle']

  #Check whether each generated step mentions targets for pick and place, that exist within the environment
  feasibility += sum([2 if parse_instruction(step)[0] in object_names else -2 for step in steps])
  feasibility += sum([2 if parse_instruction(step)[1] in object_names or parse_instruction(step)[1] in accessible_place_destinations else -2 for step in steps])

  #Check that pick and place coordinates are within the environment's workspace boundaries
  (min_x, max_x), (min_y, max_y) = env.workspace_boundaries
  feasibility += sum([2 if min_x <= xy[0] <= max_x and min_y <= xy[1] <= max_y else -2 for xy in pickxy])
  feasibility += sum([2 if min_x <= xy[0] <= max_x and min_y <= xy[1] <= max_y else -2 for xy in placexy])

  # Calculate the feasibility score
  feasibility_score = feasibility / (4 * len(steps))
  return feasibility_score

def visualize_task_feasibility(tasks, feasibility_scores):
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(tasks)), feasibility_scores, color='skyblue')
    plt.xlabel('Tasks')
    plt.ylabel('Average Feasibility Score')
    plt.title('Average Feasibility Score for Each Task')
    plt.xticks(range(len(tasks)), tasks, rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# Example usage:
#set_env()
#task="Move all the blocks to the top left corner"
#steps = ["Pick yellow block and Place it on the top left corner", "Pick red block and Place it in the top left corner"]
#environment_description = "objects=[yellow block, red block, red bowl, cyan block]"
#pickxy = [[-0.05784754 -0.30224216],[0.09820628 -0.4529148]]
#placexy = [[-0.25 -0.25],[0, -0.5]]

#feasibility_score = check_feasibility(task, steps, environment_description, pickxy, placexy)
#print("Feasibility score:", feasibility_score)
