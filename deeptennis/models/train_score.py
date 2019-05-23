import argparse
from pathlib import Path
import logging
from logging.config import fileConfig
fileConfig('logging_config.ini')
import json
import _jsonnet
import mlflow

from allennlp.training import Trainer
from allennlp.training.learning_rate_schedulers import LearningRateWithoutMetricsWrapper, SlantedTriangular, CosineWithRestarts
from allennlp.data.dataset_readers import DatasetReader
from allennlp.models import Model
from allennlp.data.iterators import BasicIterator
from allennlp.training.optimizers import Optimizer
from allennlp.common import Params

import torch.nn as nn

from deeptennis.data.dataset import CourtAndScoreTransform, ImagePathsDataset, TennisDatasetReader
from deeptennis.vision.transforms import *
import deeptennis.models.models as models
import deeptennis.models.vision_models as models

logger = logging.getLogger(__name__)

def log_params(d, prefix=''):
    for k, v in d.items():
        if isinstance(v, dict):
            log_params(v, prefix + k + '_')
        else:
            mlflow.log_param(prefix + k, v)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=30)
    parser.add_argument('--checkpoint-path', type=str, default="")
    parser.add_argument('--dataset-path', type=str, default="")
    parser.add_argument('--param-file', type=str, default="")
    args = parser.parse_args()

    use_gpu = args.gpu or torch.cuda.is_available()

    params = Params.from_file(args.param_file)
    dataset_path = Path(args.dataset_path)

    model = Model.from_params(params.pop("model"))

    train_reader = DatasetReader.from_params(params.pop("train_reader"))
    valid_reader = DatasetReader.from_params(params.pop("valid_reader"))

    batch_size = params.pop("batch_size")
    epochs = params.pop("epochs")
    train_path = dataset_path / "train.json"
    with open(train_path, 'rb') as f:
        _json = json.load(f)
        batches_per_epoch = int(len(_json['annotations']) / batch_size)
    logger.info(f"Batches per epoch: {batches_per_epoch}")

    train_instances = train_reader.read(train_path)
    valid_instances = train_reader.read(dataset_path / "test.json")
    iterator = BasicIterator(batch_size)
    optimizer = Optimizer.from_params([(n, p) for n, p in model.named_parameters() if p.requires_grad], params.pop("optim"))

    lr_sched = SlantedTriangular(optimizer,
                                 epochs,
                                 batches_per_epoch,
                                 ratio=100,
                                 cut_frac=0.5,
                                 gradual_unfreezing=params.get("gradual_unfreezing", False))

    with mlflow.start_run():
        mlflow.log_artifact(args.param_file)
        trainer = Trainer(model, optimizer, iterator, train_instances, valid_instances,
                          learning_rate_scheduler=LearningRateWithoutMetricsWrapper(lr_sched),
                          serialization_dir=args.checkpoint_path,
                          num_epochs=epochs,
                          summary_interval=args.log_interval,
                          should_log_learning_rate=True,
                          cuda_device=0 if use_gpu else -1,
                          histogram_interval=args.log_interval * 10,
                          validation_metric=params.get("valid_metric", "-loss"),
                          patience=params.get("patience", 2),
                          num_serialized_models_to_keep=2)
        trainer.train()

