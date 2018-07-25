# Reinforcement_learning_binary-_graph

### solve the following simple problem:

Generate a binary tree of depth 5. Set the attribute vector of each node to either [0,1] or [1,0] (set that of the root to [1,0]). The goal will be to learn to perform random walks (starting from the root node) where most of the visited nodes have the [1,0] attribute vector. Make sure that there are only a few such paths of length 5 and that we only move from the root to the leaves (cannot go back to nodes we have already visited). After generating a walk, the reward will be equal to (number of nodes in walk whose attribute vector is [1,0] )/5, hence, it will be equal to 1 if the attribute vectors of all nodes is [1,0], equal to 0 if the attribute vectors of none of the nodes is [1,0], and between 0 and 1 if the attribute vectors of some nodes is [1,0].
