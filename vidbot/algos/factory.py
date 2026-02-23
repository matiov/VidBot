"""Factory methods for creating models"""

from algos.traj_algos import (
    TrajectoryDiffusionModule,
)

from algos.traj_rot_algos import (
    TrajectoryRotationDiffusionModule,
)

from algos.vq_algos import (
    GoalVectorQuantizationModule,
)

from algos.goal_algos import (
    GoalFormerModule,
)

from algos.contact_algos import (
    ContactFormerModule,
)


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
        algo = TrajectoryDiffusionModule(
            algo_config=algo_config, train_config=train_config
        )
    elif algo_name == "diffuser_rot":
        algo = TrajectoryRotationDiffusionModule(
            algo_config=algo_config, train_config=train_config
        )
    elif algo_name == "rqvae":
        algo = GoalVectorQuantizationModule(
            algo_config=algo_config, train_config=train_config
        )
    elif algo_name == "goalformer":
        algo = GoalFormerModule(algo_config=algo_config, train_config=train_config)
    elif algo_name == "contactformer":
        algo = ContactFormerModule(algo_config=algo_config, train_config=train_config)

    else:
        raise NotImplementedError("{} is not a valid algorithm" % algo_name)
    return algo