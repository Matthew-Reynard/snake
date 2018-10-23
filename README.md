# snake
Snake game using Reinforcement Learning (RL) and TensorFlow

<p align="center">
  <img src="https://raw.githubusercontent.com/Matthew-Reynard/snake/master/Images/snake.png" width="200" title="Snake Game" alt="[Snake Game Image]">
</p>

*Note: still under construction*

---

### LIBRARIES USED:
virtualenv/conda (helpful, but not necessary)  
numpy  
pygame  
pandas  
tensorflow
matplotlib  

### ABOUT:
The simple snake game that we all know and love will now be controlled by a machine.   
Using RL, the snake will learn about its environment and learn to choose the best posible action for the current state.

### METHODS:
	
1. Q-Learning - Using a basic Q-Learning lookup table and updating it after every action until convergence. This approach is the most basic and yet the most important as it sets a foundation for Deep Q-Learning. ``` Qlearn.py ```

2. Neural network function approximation - Using a single layer NN, the one hot input vector of the current state is used to find the best action to execute. This is done be using the Q-Learning approach. ``` DQN.py ```

3. A Deep Q Network (DQN) - A 3 layer fully connected NN, with activation functions to add non-linearity in the hopes of outperforming the Linear function model. ``` DQN.py ```

4. A Deep Q network with a Convolutional layer - The input vector is a 3 dimensional tensor with 3 layers of the grid showing the positions of POI (points of interest). After 2 conv layers and max pooling, theres 2 fully connected layers and the output of possible actions. ``` CNN.py ```

5. A DQN with the convolutional layer centered around the head of the snake - This is a similar architecture to the previous network, with only the input state changed. ``` CNN_local.py ```


### RL Methods breakdown:
Q learning - The first method that I implemented was a simple Q learning lookup table where each possible state was allowed 3 actions in the Q matrix. Because the state space can get rather large rather fast, this won't work for the full snake game, therefore it was implemented with just a snakes head and the food. This can be found in ```QLearn.py```  

Deep Q Network - The second method is using a Deep Neural Network (DNN) to approximate the Q values and decide which action would be the best one given the current state. This can be found in ```DQN.py```  

Adding to the DQN: First step was to make the NN deeper from a linear approximator to a non-linear model with the addition of the ReLU activation function. Next step was to add a CNN at the beginning of the network, this will extract the features it thinks impact the output the most and hopefully will be get a better result using this instead of just a fully connected NN. Finally adding a local state space instead of the entire grid as the input will help generalise this and make it fully scalable in terms of the potential grid size.

### FUTURE PLANS: 
Create an IRL environment with an easy executable file or hosted on my website.
Create an executable file (with GUI)
Adding moving obstacles to the environment in the hopes that it would move away.
Use the pixel values instead of the grid positions.

### BUGS:
The controls aren't as responsive as I'd like - it only executes the last button (input) click of the frame. (FIX: maybe make the controls on a separate thread)

### Website
Click [here](http://www.matthewreynard.com) to keep an eye out for the Inverse Reinforcement Learning snake game, I will update this page once it goes live.
