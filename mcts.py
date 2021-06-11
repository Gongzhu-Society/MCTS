from __future__ import division

import time
import math
import random
import heapq


def randomPolicy(state):
    while not state.isTerminal():
        try:
            action = random.choice(state.getPossibleActions())
        except IndexError:
            raise Exception("Non-terminal state has no possible actions: " + str(state))
        state = state.takeAction(action)
    return state.getReward()


class treeNode():
    def __init__(self, state, parent):
        self.state = state
        self.isTerminal = state.isTerminal()
        self.isFullyExpanded = self.isTerminal
        self.parent = parent
        self.numVisits = 0
        self.totalReward = 0
        self.children = {}

    def __str__(self):
        s=[]
        s.append("totalReward: %s"%(self.totalReward))
        s.append("numVisits: %d"%(self.numVisits))
        s.append("isTerminal: %s"%(self.isTerminal))
        s.append("possibleActions: %s"%(self.children.keys()))
        return "%s: {%s}"%(self.__class__.__name__, ', '.join(s))

class mcts():
    def __init__(self, timeLimit=None, iterationLimit=None, explorationConstant=1 / math.sqrt(2),
                 rolloutPolicy=randomPolicy):
        if timeLimit != None:
            if iterationLimit != None:
                raise ValueError("Cannot have both a time limit and an iteration limit")
            # time taken for each MCTS search in milliseconds
            self.timeLimit = timeLimit
            self.limitType = 'time'
        else:
            if iterationLimit == None:
                raise ValueError("Must have either a time limit or an iteration limit")
            # number of iterations of the search
            if iterationLimit < 1:
                raise ValueError("Iteration limit must be greater than one")
            self.searchLimit = iterationLimit
            self.limitType = 'iterations'
        self.explorationConstant = explorationConstant
        self.rollout = rolloutPolicy

    def search(self, initialState, needDetails=False):
        self.root = treeNode(initialState, None)

        if self.limitType == 'time':
            timeLimit = time.time() + self.timeLimit / 1000
            while time.time() < timeLimit:
                self.executeRound()
        else:
            for i in range(self.searchLimit):
                self.executeRound()

        bestChild = self.getBestChild(self.root, 0)
        action=(action for action, node in self.root.children.items() if node is bestChild).__next__()
        if needDetails:
            return {"action": action, "expectedReward": bestChild.totalReward / bestChild.numVisits}
        else:
            return action

    def executeRound(self):
        """
            execute a selection-expansion-simulation-backpropagation round
        """
        node = self.selectNode(self.root)
        reward = self.rollout(node.state)
        self.backpropogate(node, reward)

    def selectNode(self, node):
        while not node.isTerminal:
            if node.isFullyExpanded:
                node = self.getBestChild(node, self.explorationConstant)
            else:
                return self.expand(node)
        return node

    def expand(self, node):
        actions = node.state.getPossibleActions()
        for action in actions:
            if action not in node.children:
                newNode = treeNode(node.state.takeAction(action), node)
                node.children[action] = newNode
                if len(actions) == len(node.children):
                    node.isFullyExpanded = True
                return newNode
        raise Exception("Should never reach here")

    def backpropogate(self, node, reward):
        while node is not None:
            node.numVisits += 1
            node.totalReward += reward
            node = node.parent

    def getBestChild(self, node, explorationValue):
        bestValue = float("-inf")
        bestNodes = []
        for child in node.children.values():
            nodeValue = node.state.getCurrentPlayer() * child.totalReward / child.numVisits + explorationValue * math.sqrt(
                2 * math.log(node.numVisits) / child.numVisits)
            if nodeValue > bestValue:
                bestValue = nodeValue
                bestNodes = [child]
            elif nodeValue == bestValue:
                bestNodes.append(child)
        return random.choice(bestNodes)

class ismcts(mcts):
    # additional function need to be implemented for Gamestate class:
    # resample()
    # renew_hidden_information(Gamestate.next_hidden_information())
    def executeRound(self):
        # execute a selection-expansion-simulation-backpropagation round
        # notice that in get best child function, we may not resample hidden information
        # thus at the beginning of each round, we resample instead
        self.root.state.resample()
        node = self.selectNode(self.root)
        reward = self.rollout(node.state)
        self.backpropogate(node, reward)

    def isFullyExpandedjudge(self, node):
        # judge: all actions are a subset of children, then fully expanded
        if node.isTerminal:
            return True
        actions = node.state.getPossibleActions()
        if set(actions).issubset(node.children.keys()):
            return True
        else:
            return False

    def selectNode(self, node):
        while not node.isTerminal:
            if self.isFullyExpandedjudge(node):
                node = self.getBestChild(node, self.explorationConstant)
            else:
                return self.expand(node)
        return node

    def getBestChild(self, node, explorationValue):
        # return a node that has already been expanded
        # however, only possible children in this world can be selected as candidates
        # also, renew the hidden information of the node returned, using information in node
        bestValue = float("-inf")
        bestNodes = []
        bestActions = []
        possible_actions = node.state.getPossibleActions()
        for enfant in node.children.keys():
            if enfant not in possible_actions:
                continue
            child = node.children[enfant]
            nodeValue = node.state.getCurrentPlayer() * child.totalReward / child.numVisits + explorationValue * math.sqrt(
                2 * math.log(node.numVisits) / child.numVisits)
            if nodeValue > bestValue:
                bestValue = nodeValue
                #bestNodes = [child]
                bestActions = [enfant]
            elif nodeValue == bestValue:
                #bestNodes.append(child)
                bestActions.append(enfant)
        action_sampled = random.choice(bestActions)
        node_to_return = node.children[action_sampled] #random.choice(bestNodes)
        node_to_return.state.renew_hidden_information(node.state.next_hidden_information(action_sampled))
        #print('get best child')
        return node_to_return

    def expand(self, node):
        # expand unexplored actions for node
        #print('expand')
        actions = node.state.getPossibleActions()
        #print('actions')
        #print(actions)
        #print('children')
        #print(node.children.keys())
        for action in actions:
            if action not in node.children:
                newNode = treeNode(node.state.takeAction(action), node)
                node.children[action] = newNode
                return newNode
        raise Exception("Should never reach here")

def trivialPolicy(state):
    return state.getReward()

class abpruning():
    def __init__(self, deep, rolloutPolicy = trivialPolicy, n_killer = 2, gameinf=65535):
        """
            deep: how many layers to be search, must >= 1
            gameinf: an upper bound of getReward() return values used as "inf" in algorithm
        """
        self.deep = deep
        self.rollout = rolloutPolicy
        self.n_killer = n_killer
        self.gameinf = gameinf
        self.counter = 0

    def search(self, initialState, needDetails=False):
        children = {}
        killers = {} # best actions of brother branches, for killer heuristic optimization
        for action in initialState.getPossibleActions():
            val,ks = self.alphabeta(initialState.takeAction(action), self.deep-1, -1*self.gameinf, self.gameinf, killers = killers)
            children[action] = val
            for k in ks:
                killers[k] = killers.setdefault(k,0) + 1
        self.children = children

        """CurrentPlayer=initialState.getCurrentPlayer()
        if CurrentPlayer==1:
            bestAction = max(self.children.items(),key=lambda x: x[1])
        elif CurrentPlayer==-1:
            bestAction = min(self.children.items(),key=lambda x: x[1])
        else:
            raise Exception("getCurrentPlayer() should return 1 or -1 rather than %s"%(CurrentPlayer,))

        if needDetails:
            return {"action": bestAction[0], "expectedReward": bestAction[1]}
        else:
            return bestAction[0]"""

    def alphabeta(self, node, deep, alpha, beta, killers = {}):
        if deep==0 or node.isTerminal():
            self.counter += 1
            return self.rollout(node),[]

        CurrentPlayer=node.getCurrentPlayer()
        actions = node.getPossibleActions()
        actions.sort(key=lambda x: killers.get(x,-1),reverse=True)
        subkillers = {}
        bestactions = []
        if CurrentPlayer == 1:
            maxeval = -1*self.gameinf
            for action in actions:
                val,ks = self.alphabeta(node.takeAction(action), deep-1, alpha, beta, killers = subkillers)
                maxeval = max(val,maxeval)
                alpha = max(val, alpha)
                bestactions.append((action,val))
                if beta <= alpha:
                    break
                for k in ks:
                    subkillers[k] = subkillers.setdefault(k,0) + 1
            bestactions.sort(key=lambda x: x[1],reverse=True)
            bestactions = [i[0] for i in bestactions[0:min(len(bestactions),self.n_killer)]]
            return maxeval,bestactions
        elif CurrentPlayer == -1:
            mineval = self.gameinf
            for action in actions:
                val,ks = self.alphabeta(node.takeAction(action), deep-1, alpha, beta, killers = subkillers)
                mineval = min(val, mineval)
                beta = min(val, beta)
                bestactions.append((action,val))
                if beta <= alpha:
                    break
                for k in ks:
                    subkillers[k] = subkillers.setdefault(k,0) + 1
            bestactions.sort(key=lambda x: x[1])
            bestactions = [i[0] for i in bestactions[0:min(len(bestactions),self.n_killer)]]
            return mineval,bestactions
        else:
            raise Exception("getCurrentPlayer() should return 1 or -1 rather than %s"%(CurrentPlayer,))