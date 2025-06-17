# pacmanAgents.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from pacman import Directions
from game import Agent
import random
import game
import util

class QLearningAgent(Agent):
    def __init__(self, epsilon=0.1, alpha=0.5, discount=0.9):
        super().__init__()
        self.epsilon = epsilon    
        self.alpha = alpha       
        self.discount = discount 
        self.q_values = {}   

    def getQValue(self, state, action):
        
        return self.q_values.get((state, action), 0.0)

    def update(self, state, action, next_state, reward):
        best_next_q = max([self.getQValue(next_state, a) for a in self.getLegalActions(next_state)], default=0)
        self.q_values[(state, action)] = (1 - self.alpha) * self.getQValue(state, action) + \
                                         self.alpha * (reward + self.discount * best_next_q)

    def getLegalActions(self, state):
        return state.getLegalPacmanActions()

    def chooseAction(self, state):
        legal_actions = self.getLegalActions(state)
        if not legal_actions:
            return None

        if util.flipCoin(self.epsilon):
            return random.choice(legal_actions)
        else:
            q_values = [(self.getQValue(state, action), action) for action in legal_actions]
            max_q_value = max(q_values, key=lambda x: x[0])[0]
            best_actions = [action for q, action in q_values if q == max_q_value]
            return random.choice(best_actions)



class LeftTurnAgent(game.Agent):
    "An agent that turns left at every opportunity"

    def getAction(self, state):
        legal = state.getLegalPacmanActions()
        current = state.getPacmanState().configuration.direction
        if current == Directions.STOP:
            current = Directions.NORTH
        left = Directions.LEFT[current]
        if left in legal:
            return left
        if current in legal:
            return current
        if Directions.RIGHT[current] in legal:
            return Directions.RIGHT[current]
        if Directions.LEFT[left] in legal:
            return Directions.LEFT[left]
        return Directions.STOP


class GreedyAgent(Agent):
    def __init__(self, evalFn="scoreEvaluation"):
        self.evaluationFunction = util.lookup(evalFn, globals())
        assert self.evaluationFunction != None

    def getAction(self, state):
        # Generate candidate actions
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        successors = [(state.generateSuccessor(0, action), action)
                      for action in legal]
        scored = [(self.evaluationFunction(state), action)
                  for state, action in successors]
        bestScore = max(scored)[0]
        bestActions = [pair[1] for pair in scored if pair[0] == bestScore]
        return random.choice(bestActions)


def scoreEvaluation(state):
    return state.getScore()
