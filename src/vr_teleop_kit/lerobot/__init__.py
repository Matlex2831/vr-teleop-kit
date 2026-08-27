"""LeRobot ``Teleoperator`` adapters.

Importing the submodules registers the config types with LeRobot's
``TeleoperatorConfig`` registry, after which
``--teleop.type=bi_quest_teleop`` / ``single_arm_quest_teleop`` /
``so101_quest_teleop`` work in LeRobot CLIs. Requires ``lerobot`` to be
installed.
"""

from .bi_quest_teleop import BiQuestTeleoperator, BiQuestTeleoperatorConfig
from .single_arm_quest_teleop import (
    SingleArmQuestTeleoperator,
    SingleArmQuestTeleoperatorConfig,
)
from .so101_quest_teleop import (
    SO101QuestTeleoperator,
    SO101QuestTeleoperatorConfig,
)

__all__ = [
    "BiQuestTeleoperator",
    "BiQuestTeleoperatorConfig",
    "SingleArmQuestTeleoperator",
    "SingleArmQuestTeleoperatorConfig",
    "SO101QuestTeleoperator",
    "SO101QuestTeleoperatorConfig",
]
