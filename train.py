import wandb
from optimisers import Optimiser
import cli_args
arguments = cli_args.argumentsIntake()
# def train():
#     wandb.init()
#     data = wandb.config
#     name = f"Optimiser: {data.optimizer} Activation: {data.activation} OutputActivation: {data.output_activation} BatchSize: {data.batch_size} Epochs: {data.epochs} WeightsInit: {data.weight_init}"
#     wandb.init(name=name)
#     Optimizer = Optimiser(arguments)
#     Optimizer.train()

# wandb.agent("x1ysqe92", function=train, count=20)

Optimizer = Optimiser(arguments)
Optimizer.train()
