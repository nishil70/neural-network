# Neural Network

## Perceptrons
- Perceptrons can represent all of the primitive boolean functions AND, OR, NAND, and NOR. This ability is important because every boolean function can be represented by some network of interconnected units based on these primitives

  ### Perceptorn training rule
  - One way to learn an acceptable weight vector is to begin with random weights, then iteratively apply the perceptron to each training example, modifying the perceptron weights whenever it misclassifies an example, this process is repeated many times until the perceptron classifies all traning examples correctly. Weights are modified at each step according the the perceptron training rule

## The BACK-PROPAGATION algorithm
- Assumes the network is a fixed structure that corresponds to a directed graph, possibly containing cycles. Learning corresponds to choosing a weight value for each egde in the graph
- 
