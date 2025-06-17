"""
Carolina Dias Lima Maikuma, Felipe Queiroz Rodrigues, 
Maria Eduarda de Moura Eguchi e Matheus Musashi Tanaka 
"""

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState
from multiAgents import MultiAgentSearchAgent


class MinimaxAgent(MultiAgentSearchAgent):
    def __init__(self, evalFn='betterEvaluationFunction', depth='2'): 
        super().__init__(evalFn, int(depth))

    def getAction(self, gameState: GameState):
        """
        Retorna a melhor ação para o Pac-Man usando o algoritmo Minimax.
        """
        def minimax(agentIndex, depth, state):
            if state.isWin() or state.isLose() or depth == self.depth: # Condição de parada
                # Avalia o estado atual usando a função de avaliação
                return betterEvaluationFunction(state)

            numAgents = state.getNumAgents()
            nextAgent = (agentIndex + 1) % numAgents
            nextDepth = depth + 1 if nextAgent == 0 else depth

            legalActions = state.getLegalActions(agentIndex)
            if not legalActions:
                return betterEvaluationFunction(state)

            if agentIndex == 0:
                # Pac-Man (MAX)
                return max(
                    minimax(nextAgent, nextDepth, state.generateSuccessor(agentIndex, action))
                    for action in legalActions
                )
            else:
                # Fantasmas (MIN)
                return min(
                    minimax(nextAgent, nextDepth, state.generateSuccessor(agentIndex, action))
                    for action in legalActions
                )

        # Escolhe a melhor ação entre as possíveis
        bestScore = -float('inf')
        bestAction = None
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            score = minimax(1, 0, successor)
            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestAction


def betterEvaluationFunction(currentGameState: GameState):
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()

    score = currentGameState.getScore()

    # Distância da comida mais próxima
    foodDistances = [manhattanDistance(pos, f) for f in food]
    minFoodDistance = min(foodDistances) if foodDistances else 1

    # Distância para os fantasmas
    ghostDistances = [manhattanDistance(pos, ghost.getPosition()) for ghost in ghostStates]
    minGhostDistance = min(ghostDistances) if ghostDistances else 1

    # Checa se fantasmas estão assustados
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    scaredBonus = sum(scaredTimes)

    # Penalidade se estiver muito perto de um fantasma não assustado
    dangerPenalty = 0
    for ghostState in ghostStates:
        dist = manhattanDistance(pos, ghostState.getPosition())
        if ghostState.scaredTimer == 0 and dist <= 1:
            dangerPenalty -= 200  # distância perigosa

    # Bonus por comer cápsulas e proximidade
    capsuleBonus = 0
    if capsules:
        capsuleDistances = [manhattanDistance(pos, cap) for cap in capsules]
        capsuleBonus += 10 / (min(capsuleDistances) + 1)

    # Fórmula final de avaliação
    return (
        score
        + 10 / (minFoodDistance + 1)
        + capsuleBonus
        + scaredBonus
        + dangerPenalty
    )

# Abreviação opcional
better = betterEvaluationFunction
