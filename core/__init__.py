"""
Module which deals with :
- Federated Learning classes such as aggregation class and node class
- The evaluator class for better aggregation through Reinforcement Learning
- Main functions for training and evaluation
"""

from .federated_learning.aggregation import FederatedAveraging, EvaluatorServer
from .federated_learning.worker import Worker
from .federated_learning import participation

PARTICIPATION = {"all": participation.entirely, "randomly": participation.randomly}
