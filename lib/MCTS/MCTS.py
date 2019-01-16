import math
import numpy as np
EPS = 1e-8
import utils_mcts
class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, graph_emb, numMCTSSims, cpuct, path_length=100):
        self.game = game
        self.graph_emb = graph_emb
        self.path_length = path_length
        self.nnet = nnet
        self.numMCTSSims = numMCTSSims
        self.cpuct = cpuct
        self.Qsa = {}       # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}       # stores #times edge s,a was visited
        self.Ns = {}        # stores #times board s was visited
        self.Ps = {}        # stores initial policy (returned by neural net)

        #self.Es = {}        # stores game.getGameEnded ended for board s
        self.Vs = {}        # stores game.getValidMoves for board s

    def getActionProb(self, path, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        partial path.
        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        path = tuple(path)
        for i in range(self.numMCTSSims):
            self.search(path)

        counts = [self.Nsa[(path,vertex)] if (path,vertex) in self.Nsa else 0 for vertex in range(self.game.get_action_size())]

        if temp==0:
            bestA = np.argmax(counts)
            probs = [0]*len(counts)
            probs[bestA]=1
            return probs

        counts = [x**(1./temp) for x in counts]
        probs = [x/float(sum(counts)) for x in counts]
        print(probs)
        return probs


    def search(self, path):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.
        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propogated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propogated up the search path. The values of Ns, Nsa, Qsa are
        updated.
        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.
        Returns:
            v: the negative of the value of the current canonicalBoard
        """
        if len(path) == self.path_length:
            _, v = self.nnet.predict(utils_mcts.get_states_emb([list(path)], self.graph_emb))
            return v[0]

        """
        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)
        if self.Es[s]!=0:
            # terminal node
            return -self.Es[s]
        """
        if path not in self.Ps:
            # leaf node
            self.Ps[path], v = self.nnet.predict(utils_mcts.get_states_emb([list(path)], self.graph_emb))
            valids = self.game.get_valid_actions(path[-1])
            self.Ps[path] = self.Ps[path]*valids      # masking invalid moves
            sum_Ps_path = np.sum(self.Ps[path])
            if sum_Ps_path > 0:
                self.Ps[path] /= sum_Ps_path          # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable
                
                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.   
                print("All valid moves were masked, do workaround.")
                self.Ps[path] = self.Ps[path] + valids
                self.Ps[path] /= np.sum(self.Ps[path])

            self.Vs[path] = valids
            self.Ns[path] = 0
            return v[0]

        valids = self.Vs[path]
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(self.game.get_action_size()):
            if valids[a]:
                if (path,a) in self.Qsa:
                    u = self.Qsa[(path,a)] + self.cpuct*self.Ps[path][a]*math.sqrt(self.Ns[path])/(1+self.Nsa[(path,a)])
                else:
                    u = self.cpuct*self.Ps[path][a]*math.sqrt(self.Ns[path] + EPS)     # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = a

        vertex = best_act
        next_path = self.game.get_next_state(list(path), vertex)


        v = self.search(tuple(next_path))

        if (path,vertex) in self.Qsa:
            self.Qsa[(path,vertex)] = (self.Nsa[(path,vertex)]*self.Qsa[(path,vertex)] + v)/(self.Nsa[(path,vertex)]+1)
            self.Nsa[(path,vertex)] += 1
        else:
            self.Qsa[(path,vertex)] = v
            self.Nsa[(path,vertex)] = 1

        self.Ns[path] += 1
        return v


