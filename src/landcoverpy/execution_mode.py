from enum import Enum


class ExecutionMode(Enum):
    TRAINING = 1
    LAND_COVER_PREDICTION = 2
    SECOND_LEVEL_PREDICTION = 3