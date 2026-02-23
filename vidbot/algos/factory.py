"""Factory methods for creating models"""

from vidbot.algos.contact_algos import ContactFormerModule
from vidbot.algos.goal_algos import GoalFormerModule
from vidbot.algos.traj_algos import TrajectoryDiffusionModule
from vidbot.algos.traj_rot_algos import TrajectoryRotationDiffusionModule
from vidbot.algos.vq_algos import GoalVectorQuantizationModule


def algomodule_factory(config):
    """
    A factory for creating training algos

    Args:
        config (ExperimentConfig): an ExperimentConfig object,

    Returns:
        algo: pl.LightningModule
    """
    algo_config = config.ALGORITHM
    train_config = config.TRAIN
    algo_name = algo_config.name

    if algo_name == "diffuser":
        algo = TrajectoryDiffusionModule(algo_config=algo_config, train_config=train_config)
    elif algo_name == "diffuser_rot":
        algo = TrajectoryRotationDiffusionModule(
            algo_config=algo_config, train_config=train_config
        )
    elif algo_name == "rqvae":
        algo = GoalVectorQuantizationModule(algo_config=algo_config, train_config=train_config)
    elif algo_name == "goalformer":
        algo = GoalFormerModule(algo_config=algo_config, train_config=train_config)
    elif algo_name == "contactformer":
        algo = ContactFormerModule(algo_config=algo_config, train_config=train_config)

    else:
        raise NotImplementedError("{} is not a valid algorithm" % algo_name)
    return algo
