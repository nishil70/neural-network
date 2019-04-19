# Neural Network

## Perceptrons
- Perceptrons can represent all of the primitive boolean functions AND, OR, NAND, and NOR. This ability is important because every boolean function can be represented by some network of interconnected units based on these primitives

  ### Perceptorn training rule
  - One way to learn an acceptable weight vector is to begin with random weights, then iteratively apply the perceptron to each training example, modifying the perceptron weights whenever it misclassifies an example, this process is repeated many times until the perceptron classifies all traning examples correctly. Weights are modified at each step according the the perceptron training rule.
  - The above learning procedure can be proven to converge to correctly classify all training examples, provided the training examples are linearly separable.
  
  ### Gradient Descent and the Delta Rule
  - If training examples are not linearly separable, the delta rule converges toward a best-fit approximation to the target concept.
  - Key idea behind delta rule is to use gradient descent to search the hypothesis space of possible weight vectors to find the weights that best fits the training examples.
  - The hypothesis that minimizes E is also the most probable hypothesis in H given the training data
  - Gradient descent search determines a weight vector that minimizes E by starting with an arbitrary initial weight vector, then repeatedly modifying it in small steps. At each step, the weight vector is altered in the direction that produces the steepest descent along the error surface. This process continues until the global minimum error is reached.
  

## The BACK-PROPAGATION algorithm
- Assumes the network is a fixed structure that corresponds to a directed graph, possibly containing cycles. Learning corresponds to choosing a weight value for each egde in the graph
- 
