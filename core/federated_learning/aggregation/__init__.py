"""
Module which deals with all classes for aggregation
- FederatedAveraging, which can be mixed with different "schedulers"
found in `core/federated_learning/participation.py`
- EvaluationV1 and EvaluationV2 which manage the aggregation with an agent
and are different for the definition of the state and the reward
"""
from core.federated_learning.aggregation.fedavg import FederatedAveraging
