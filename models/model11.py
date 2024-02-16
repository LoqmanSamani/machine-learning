
"""
Steps for building a decision tree

  - Start with all examples at the root node
  - Calculate information gain for splitting on all possible features,
    and pick the one with the highest information gain
  - Split dataset according to the selected feature, 
    and create left and right branches of the tree
  - Keep repeating splitting process until stopping criteria is met

"""


