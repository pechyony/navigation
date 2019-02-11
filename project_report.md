
[image1]: ./images/environment.gif "bananas environment" 
[image2]: ./images/graph.jpg "convergence graph" 
[image3]: ./images/trained_agent.gif "trained agent" 

# Project Report

In this project we trained an agent to navigate in a closed space and collect bananas. 

![alt text][image1]

The agent was trained using DQN algorithm. In the next two sections we describe this environment as well as the training process. 

## Bananas Environment

In bananas environment an agent navigates in a closed space and collects bananas. A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

* 0 - move forward
* 1 - move backward
* 2 - turn left
* 3 - turn right

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

This environment is provided by Udacity and is implemented using 
[Unity Machine Learning Agents Toolkit](https://github.com/Unity-Technologies/ml-agents). Bananas environment is similar to, but not identical to the Banana Collector environment of the toolkit.  

## Training an Agent using DQN Algorithm

We trained an agent using a simplified version of [deep Q-network (DQN) algorithm](https://deepmind.com/research/dqn/). This algorithm uses a replay memory that has a finite capacity and works in FIFO way. DQN algorithm also uses local and target action-value functions q and q'. These functions are implemented as feedforward neural networks and have identical architecture. The networks take as input a state vector and output action value of every possible action. We denote by q(S,w,A) the value of action A at state S, computed by local neural network with weights w. Similarly, q'(S,w',A) is the value of action A at state S, computed by target neural network with weights w'. An &epsilon;-greedy policy based on q(S,w,&middot;) chooses a random action with probability &epsilon; and an action that maximizes q(S,w,&middot;) with probability 1-&epsilon;. Out implementation of DQN algorithm is summarized below:

Initialize replay memory with capacity N  
Initialize local action-value function q with weights w  
Initialize target action-value function q' with weights w' = w  
Set number of observed experiences to 0  
Set &epsilon;=&epsilon;<sub>start</sub>   
For the episode e = 1 ... M:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Get from environment initial state S  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;   For time step t = 1 ... T:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Choose action A from state S using &epsilon;-greedy policy q(S,w,&middot;)   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Take action A, observe reward R, next state S' and the indicator if S' is a final state   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Store experience tuple (S,A,R,S',done) in the replay memory   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Increase the number of observed experiences by 1  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;S = S'  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&epsilon; = max(&epsilon;<sub>end</sub>,
&epsilon;<sub>decay</sub>&middot;&epsilon;)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;If 
the number of observed experiences is a multiple of U:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Sample a batch of K experiences (S<sub>i</sub>,A<sub>i</sub>,R<sub>i</sub>,S'<sub>i</sub>,done<sub>i</sub>), i=1 ... K, from the replay memory  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Set target y<sub>i</sub>=r<sub>i</sub>+&gamma;&middot;max<sub>A</sub>q'(S'<sub>i</sub>,w',A), i=1 ... K  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Set &Delta;w=&sum;<sub>i=1 ... K</sub> (y<sub>i</sub>-q(S<sub>i</sub>,w,A<sub>i</sub>))&middot;&nabla;q(S<sub>i</sub>,w,A<sub>i</sub>)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Update w = w + &alpha;&middot;&Delta;w  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Update w'=&tau;&middot;w+(1-&tau;)&middot;w'

The following table lists the architecture of local and target action-value networks.

| Name | Type | Input | Output | Activation function | 
|:-:|:-----:|:-----:|:------:|:-------------------:|  
|Input Layer| Fully Connected  | 37    | 64     | ReLU                |
|Hidden Layer| Fully Connected   | 64    | 64     | ReLU                |
|Output Layer| Fully Connected   | 64    | 4      | Linear              |

Also, the next table summarizes the values of hyperparameters

| Hyperparameter | Description | Value |
|:--------------:|:-----------:|:-----:|
| N | Size of replay memory | 100000 |
| M | Maximum number of eposodes | 2000 |
| T | Maximum number of actions in episode | 1000 |
| U | Frequency of agent learning  | every 4 actions|
| K | Batch size | 64 |
| &gamma; | Discount factor | 0.99 |
| &alpha; | Learning rate | 0.0005 |
| &tau; | Weight of local network when updating target network| 0.001 |
| &epsilon;<sub>start</sub> | Initial value of exploration probability | 1 |
| &epsilon;<sub>end</sub> | Minimal value of exploration probability | 0.01 |
| &epsilon;<sub>decay</sub> | Decay factor of the probability of exploration | 0.995 |


## Software packages

We used PyTorch to train neural network and  the API of Unity Machine Learning Agents Toolkit to interact with bananas environment. 

## Results

The agent solves environment within 600 epochs. The following graph shows the total reward as a function of the number of episode.  

![alt text][image2]

And here is the video of the trained agent collecting yellow bananas:

![alt text][image3]

## Ideas for Future Work
In this project we implemented the basic version of DQN algorithm. To speed up the convergence, we plan to try several advanced techniques, including 
[double DQN](https://arxiv.org/abs/1509.06461), [dueling DQN](https://arxiv.org/abs/1511.06581) and [prioritized experience replay](https://arxiv.org/abs/1511.05952). Also we plan to train an agent in a more challenging enviroment, where the instead of preprocessed 37-dimensional vector, the agent's state is a raw observed image. 



