# Robot Task Scheduling

In this project, I integrate Large Language Models (LLMs) with Robotics, i.e. I use human language to give instructions to a robot. The aim is as follows:
When provided with a list of tasks, the robot should break each task down into steps,then print and execute the tasks in order of "fastest" to "slowest"; the "fastest" task being the one requiring the least number of steps.

The project is inspired by SayCan: Do as I can, not as I say: an algorithm that grounds large language models with robotic affordances for long-horizon planning. Given a set of low-level robotic skills (e.g., "put the green block in the red bowl") and a high-level instruction (e.g., "stack all the blocks"), it scores what a language model believes will help forward the high-level instruction and scores what a robotic affordance model believes is possible. Together these give a task that is useful and possible and the robot executes the command. SayCan allows the robot to describe how it would go about executing a task using the aforementioned affordances. 

For this reason, the environment setup and most of the models employed by SayCan and this project are similar. The main difference is the context for which my project idea came about: In the context of machine tending robots (industrial robots designed to perform tasks related to loading and unloading parts or materials into machines or processing equipment), integrating a more complex level of "reasoning" would take automation to  an entirely other level and help increase productivity in manufacturing envrionments. By "more complex level of reasoning", I am referring to a more advanced notion of the "fastest task", i.e. instead of the robot measuring task completion time in terms of "number of steps", it would perform mathematical estimations involving factors such as distance from objects, weight, motion speed, completion time for each step, etc. Another "more complex level of reasoning" would be classifying the tasks in terms of urgency rather than speed, in other words determining which tasks are the most pressing (could be timewise-which task will bring down productivity if not completed first, or safetywise-maybe some machine parts need to be changed,...).

## Technologies involved
This project involves the use of:
* The PyBullet for the control of the simulated UR5e-2f85 robotic arm and the creation of its envrironment.
* Four machine learning models:
* Llama 2-7B as the Large Language Model used to provide instructions, in human language, to the robot. It is also with the help of this model, that the robot performs some form of reasoning to behave as expected.
    * Three visual learning models to help the robot understand the images captured by the camera, and identify specified objects (i.e. if taskA is "Pickup red block", these models will help the robot identify the red block among other objects in the environment):
        * CLIP (Contrastive Language-Image Pretraining) a computer vision model that learns visual representations from raw text supervision.
        * CLIPort a framework that combines semantic understanding and spatial precision for vision-based manipulation. It is a language-conditioned imitation-learning agent that integrates the semantic understanding of CLIP with the spatial precision of Transporter Networks.
        * ViLD (Vision and Language knowledge Distillation) a training method aiming for advancement of open-vocabulary object detection from object description by arbitrary text inputs.

## How it works
