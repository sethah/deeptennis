import argparse
from pathlib import Path
import logging
from logging.config import fileConfig
fileConfig('logging_config.ini')
import json
from typing import Iterable, List, Tuple

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
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--n-valid", type=int, default=1)
    parser.add_argument("--freeze-backbone", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=30)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--img-height", type=int, default=224)
    parser.add_argument("--img-width", type=int, default=224)
    parser.add_argument("--img-mean", type=str, default=None)
    parser.add_argument("--img-std", type=str, default=None)
    parser.add_argument("--initial-lr", type=float, default=0.001)
    parser.add_argument('--checkpoint-path', type=str, default="")
    parser.add_argument('--frame-path', type=str, default="")
    parser.add_argument('--court-path', type=str, default="")
    parser.add_argument('--score-path', type=str, default="")
    parser.add_argument('--dataset-path', type=str, default="")
    parser.add_argument('--param-file', type=str, default="")
    parser.add_argument('--action-path', type=str, default="")
    args = parser.parse_args()

    np.random.seed(args.seed)

    im_size = (args.img_height, args.img_width)
    use_gpu = args.gpu or torch.cuda.is_available()

    params = Params.from_file(args.param_file)
    dataset_path = Path(args.dataset_path)

    im_size = (args.img_height, args.img_width)
    model: models.AnchorBoxModel = Model.from_params(params.pop("model"))
    # fpn = models.FPN.from_params(params.pop("model_fpn"))
    # fpn_head = Model.from_params(params.pop("model_fpn_head"))

    train_reader = DatasetReader.from_params(params.pop("train_reader"))
    valid_reader = DatasetReader.from_params(params.pop("valid_reader"))

    batch_size = params.pop("batch_size")
    train_path = dataset_path / Path("train").with_suffix(".json")
    with open(train_path, 'rb') as f:
        _json = json.load(f)
        batches_per_epoch = int(len(_json['annotations']) / batch_size)  # TODO
    logger.info(f"Batches per epoch: {batches_per_epoch}")

    train_instances = train_reader.read(train_path)
    valid_instances = train_reader.read(dataset_path / Path("test").with_suffix(".json"))
    iterator = BasicIterator(args.batch_size)
    optimizer = Optimizer.from_params([(n, p) for n, p in model.named_parameters() if p.requires_grad], params.pop("optim"))

    lr_sched = SlantedTriangular(optimizer, args.epochs, batches_per_epoch,
                                 ratio=100, cut_frac=0.5, gradual_unfreezing=True)

    trainer = Trainer(model, optimizer, iterator, train_instances, valid_instances,
                      learning_rate_scheduler=LearningRateWithoutMetricsWrapper(lr_sched),
                       serialization_dir=args.checkpoint_path,
                       num_epochs=args.epochs,
                       summary_interval=args.log_interval,
                       should_log_learning_rate=True,
                       cuda_device=0 if use_gpu else -1,
                      histogram_interval=args.log_interval * 10,
                      num_serialized_models_to_keep=2)
    # def myhook(module_, inp, outp):
    #     trainer._tensorboard.add_train_scalar("test", outp['loss'], lr_sched.get_lr()[0] * 1000000)
    # model.register_forward_hook(myhook)
    trainer.train()

