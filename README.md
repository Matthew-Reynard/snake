# snake
Snake game using Reinforcement Learning (RL) and TensorFlow

<p align="center">
  <img src="https://raw.githubusercontent.com/Matthew-Reynard/snake/master/Images/snake.png" width="200" title="Snake Game" alt="[Snake Game Image]">
</p>

*Note: still under construction*

---

### LIBRARIES USED:
virtualenv (helpful, but not necessary)  
numpy  
pygame  
pandas  
tensorflow  

### ABOUT:
The simple snake game that we all know and love will now be controlled by a machine.   
Using RL, the snake will learn about its environment and learn to choose the best posible action for the current state.

### TWO METHODS:
RL - The standard RL that makes the snake do many simulations of the game until it optimises its reward and wins.  
Inverse RL (IRL) - Using data gathered from "experts" to train the snake to win the game (using log files).

### RL:
Q learning - The first method that I implemented was a simple Q learning lookup table where each possible state was allowed 3 actions in the Q matrix. Because the state space can get rather large rather fast, this wont work for the full snake game, therefore it was implemented with just a snakes head and the food. This can be found in QLearn.py  
Deep Q Network - The second method is using a Deep Neural Network (NN) to approximate the Q values and decide which action would be the best one given the current state. This can be found in DQN.py (STILL A WORK IN PROGRESS)

### COMPLETED:
The basics of the snake game is done.  
The creation of log files.  
QLearning (Lookup table) - can run without Tensorflow

### FUTURE PLANS:
Implement the DQN using TensorFlow  
Create an IRL environment  
Create an executable file (with GUI)

### BUGS:
The controls aren't as responsive as I'd like - it only executes the last button (input) click of the frame.  
