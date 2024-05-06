# Robot Task Scheduling

In this project, I integrate Large Language Models (LLMs) with Robotics, i.e. I use human language to give instructions to a robot. The aim is as follows:
When provided with a list of tasks, the robot should break each task down into steps,then print and execute the tasks in order of "fastest" to "slowest"; the "fastest" task being the one requiring the least number of steps.

The project is inspired by SayCan: Do as I can, not as I say: an algorithm that grounds large language models with robotic affordances for long-horizon planning. Given a set of low-level robotic skills (e.g., "Put the green block in the red bowl") and a high-level instruction (e.g., "stack all the blocks"), it scores what a language model believes will help forward the high-level instruction and scores what a robotic affordance model believes is possible. Together these give a task that is useful and possible and the robot executes the command. SayCan allows the robot to describe how it would go about executing a task using the aforementioned affordances. 

For this reason, the environment setup and two of the models employed by SayCan and this project are similar. The main differences are the context for which my project idea came about and the coordinate extraction method.
The context of this project is that of machine tending robots (industrial robots designed to perform tasks related to loading and unloading parts or materials into machines or processing equipment), integrating a more complex level of "reasoning" would take automation to  an entirely other level and help increase productivity in manufacturing environments. 
By "more complex level of reasoning", I am referring to a more advanced notion of the "fastest task", i.e. instead of the robot measuring task completion time in terms of "number of steps", it would perform mathematical estimations involving factors such as distance from objects, weight, motion speed, completion time for each step, etc. Another "more complex level of reasoning" would be classifying the tasks in terms of urgency rather than speed, in other words determining which tasks are the most pressing (could be time wise--which task will bring down productivity if not completed first, or safety wise--maybe some machine parts need to be changed,...).

Coordinate Extraction occurs by first extracting pick and place targets from an instruction, e.g. Pick the blue block and place it in the red bowl (pick target: blue block, place target: red bowl), atop-view image of the environment is fed into a visual language model (ViLD), which detects the objects present and draws bounding boxes around them. Coordinates are extracted by calculating the centre of the bounding box drawn around the target object.
## How to run it
The program can be run in two ways: locally, or entirely in Google Colab.
Running it solely in Colab is pretty straightforward, just ensure to use a GPU runtime.
Running it locally involves:
* Clone this repository `git clone [URL]`.
* Create and activate a python virtual environment, and within that environment run `pip install -r requirements.txt`.
* Run all the cells in Colab (with a GPU runtime), once the demo starts, enter "Yes" for the first question. When the program stops, the LLM's output will be saved in a JSON file and downloaded onto your local filesystem.
* Move to the "python_modules" directory and run the Blender.py python file.
   * If on Windows `python Blender.py`.
   * If on Linux `python3 Blender.py`.

If you want to run the Anvil app Proof of Concept(POC), this can only be done with the local runmode as the server times out when calling the LLM. So make sure to have already run the program locally at least once before trying to test the POC (i.e. make sure to have the `sorted_tasks.json` file on your filesystem). The app can be found at this [repository](https://github.com/Black-bird14/robot-task-scheduling-anvil "Anvil-based Interface"). There, you will also find instructions on how to clone it and run it from the platform. 
To run everything together:
* First install the anvil uplink with `pip install anvil-uplink`.
* Next open "Blender.py" and uncomment all the lines involving the word `anvil`.
* Finally, run the program as instructed above in your terminal, before starting the app from the Anvil platform.
