import argparse
from pathlib import Path
import logging
from logging.config import fileConfig
fileConfig('logging_config.ini')
import json

from allennlp.training import Trainer
from allennlp.training.learning_rate_schedulers import LearningRateWithoutMetricsWrapper, SlantedTriangular, CosineWithRestarts
from allennlp.data.dataset_readers import DatasetReader
from allennlp.models import Model
from allennlp.data.iterators import BasicIterator
from allennlp.training.optimizers import Optimizer
from allennlp.common import Params

import torch.nn as nn

from src.data.dataset import CourtAndScoreTransform, ImagePathsDataset, TennisDatasetReader
from src.vision.transforms import *
import src.models.models as models

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=30)
    parser.add_argument('--checkpoint-path', type=str, default="")
    parser.add_argument('--dataset-path', type=str, default="")
    parser.add_argument('--param-file', type=str, default="")
    args = parser.parse_args()

    np.random.seed(args.seed)

    use_gpu = args.gpu or torch.cuda.is_available()

    params = Params.from_file(args.param_file)
    dataset_path = Path(args.dataset_path)

    model: models.AnchorBoxModel = Model.from_params(params.pop("model"))

    train_reader = DatasetReader.from_params(params.pop("train_reader"))
    valid_reader = DatasetReader.from_params(params.pop("valid_reader"))

    batch_size = params.pop("batch_size")
    train_path = dataset_path / Path("train").with_suffix(".json")
    with open(train_path, 'rb') as f:
        _json = json.load(f)
        batches_per_epoch = int(len(_json['annotations']) / batch_size)
    logger.info(f"Batches per epoch: {batches_per_epoch}")

    train_instances = train_reader.read(train_path)
    valid_instances = train_reader.read(dataset_path / Path("test.json"))
    iterator = BasicIterator(args.batch_size)
    optimizer = Optimizer.from_params([(n, p) for n, p in model.named_parameters()
                                       if p.requires_grad], params.pop("optim"))

    lr_sched = SlantedTriangular(optimizer, args.epochs, batches_per_epoch,
                                 ratio=100, cut_frac=0.5, gradual_unfreezing=False)

    trainer = Trainer(model, optimizer, iterator, train_instances, valid_instances,
                      learning_rate_scheduler=LearningRateWithoutMetricsWrapper(lr_sched),
                      serialization_dir=args.checkpoint_path,
                      num_epochs=args.epochs,
                      summary_interval=args.log_interval,
                      should_log_learning_rate=True,
                      cuda_device=0 if use_gpu else -1,
                      histogram_interval=args.log_interval * 10,
                      num_serialized_models_to_keep=2)
    trainer.train()

