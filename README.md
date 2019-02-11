
[image1]: ./images/trained_agent.gif "trained agent" 

# Navigation

Project 1 in Udacity deep reinforcement learning nanodegree.

In this project we trained an agent to navigate in a closed space and collect yellow bananas only. 

![alt text][image1]

# Overview

In bananas environment an agent navigates in a closed space and collects bananas. A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

* 0 - move forward
* 1 - move backward
* 2 - turn left
* 3 - turn right

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

This environment is provided by Udacity and is implemented using 
[Unity Machine Learning Agents Toolkit](https://github.com/Unity-Technologies/ml-agents). Bananas environment is similar to, but not identical to the Banana Collector environment of the toolkit. 

# Setup 

1. Follow steps 1, 3 and 4 in the instructions [here](https://github.com/udacity/deep-reinforcement-learning#dependencies) to set up your Python environment. By following these instructions, you will install PyTorch, the ML-Agents toolkit, and a few more Python packages required for this project.

2. Clone this repository.

3. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

4. Place the file in the root folder of this repository, and unzip (or decompress) the file. 

5. (_For Windows users_)  Install VcXsrv from [here](https://sourceforge.net/projects/vcxsrv/). Then launch XLaunch executable and uncheck the option 'Native OpenGL'. 

# Usage   
Follow the instructions in `Navigation.ipynb` to train the agent.

# Files

Navigation.ipynb - notebook that trains an agent and runs it through the environment  
project_report.md - technical details of training environment and process  
checkpoint.pth - weights of the neural network of the trained DQN agent  
code - folder with Python files that implement DQN algorithm  
images - forlder with auxiliary images

