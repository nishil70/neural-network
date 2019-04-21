# Neural Network

## Contributors
- Nishil Parmar

## Dataset
- [MNIST database](https://en.wikipedia.org/wiki/MNIST_database)

## Implementions
- Neural Network
  - feed forward neural network with backpropagation learning

## Approach
### Perceptrons
- Perceptrons can represent all of the primitive boolean functions AND, OR, NAND, and NOR. This ability is important because every boolean function can be represented by some network of interconnected units based on these primitives
  
  #### Gradient Descent and delta rule
  - If training examples are not linearly separable, the delta rule converges toward a best-fit approximation to the target concept.
  - Key idea behind delta rule is to use gradient descent to search the hypothesis space of possible weight vectors to find the weights that best fits the training examples.
  - The hypothesis that minimizes E is also the most probable hypothesis in H given the training data
  - Gradient descent search determines a weight vector that minimizes E by starting with an arbitrary initial weight vector, then repeatedly modifying it in small steps. At each step, the weight vector is altered in the direction that produces the steepest descent along the error surface. This process continues until the global minimum error is reached.
  

### The BACK-PROPAGATION algorithm
- The BACKPROPAGATION algorithm learns the weights for a multilayer network
- Assumes the network is a fixed structure that corresponds to a directed graph, possibly containing cycles. Learning corresponds to choosing a weight value for each egde in the graph

## Project Files
- ## Project Files
- [MNIST handwritten digits classification](https://github.com/nishil70/HiddenMarkovModel/blob/master/notebooks/1_preprocessing.ipynb)
- [Neural network implementation](https://github.com/nishil70/HiddenMarkovModel/blob/master/notebooks/2_building-dictionary.ipynb)


## References
- http://www.cs.cmu.edu/~./awm/tutorials/neural.html
- https://www.udemy.com/image-recognition-with-neural-networks-from-scratch/
- http://neuralnetworksanddeeplearning.com/
- Thomas M. Mitchell. 1997. Machine Learning (1 ed.). McGraw-Hill, Inc., New York, NY, USA. 
