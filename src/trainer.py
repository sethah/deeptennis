import logging
from pathlib import Path
import shutil
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import allennlp.nn.util as util
from allennlp.training.learning_rate_schedulers import LearningRateScheduler

import torch
import torch.utils.data as data

from src.model import Model

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
TensorDict = Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]


# class DataIterator(object):
#
#     def __init__(self, batch_size):
#         self.batch_size = batch_size
#
#     def __call__(self,
#                  dataset: data.Dataset,
#                  num_epochs: int,
#                  shuffle: bool = True) -> Iterator[TensorDict]:
#         raise NotImplementedError


# class Trainer(object):
#
#     def __init__(self,
#                  model: Model,
#                  optimizer: torch.optim.Optimizer,
#                  train_dataset: data.Dataset,
#                  validation_dataset: data.Dataset,
#                  iterator: DataIterator,
#                  serialization_dir: Optional[str] = None,
#                  learning_rate_scheduler: Optional[LearningRateScheduler] = None,
#                  device=torch.device("cpu"),
#                  log_interval: int = 100):
#         """
#         TODO: why provide the iterator method and not the loader directly?
#         :param model: A model which also outputs a loss.
#         :param optimizer: Optimizer instantiated with the parameters of the model.
#         :param train_dataset:
#         :param validation_dataset:
#         :param iterator: A function which yields an iterator over batches, given a dataset.
#         :param serialization_dir:
#         :param device:
#         """
#         self.model = model
#         self.optimizer = optimizer
#         self.iterator = iterator
#         self.train_data = train_dataset
#         self._validation_data = validation_dataset
#         self.device = device
#         self._num_epochs = 3
#         self._serialization_dir = Path(serialization_dir)
#         self._validation_metric_decreases = True  # TODO: don't hardcode
#         self._log_interval = log_interval
#         self._learning_rate_scheduler = learning_rate_scheduler
#         self._batches_processed = 0
#         self.model.to(self.device)
#
#     def _train_epoch(self, epoch: int) -> Dict[str, float]:
#         """
#         Trains one epoch and returns metrics.
#         """
#
#         # Set the model to "train" mode.
#         self.model.train()
#
#         train_generator = self.iterator(self.train_data, self._num_epochs, shuffle=True)
#         n = 0.
#         total_loss = 0.0
#         total_batches = 0
#         train_loss = 0.0
#         for batch_idx, batch in enumerate(train_generator):
#             self.optimizer.zero_grad()
#             loss = self.batch_loss(batch, for_training=True)
#             loss.backward()
#             train_loss += loss.item()
#             total_loss += loss.item()
#
#             # This does nothing if batch_num_total is None or you are using an
#             # LRScheduler which doesn't update per batch.
#             if self._learning_rate_scheduler:
#                 self._learning_rate_scheduler.step_batch(self._batches_processed)
#             self.optimizer.step()
#
#             n += 1.
#             total_batches += 1
#             self._batches_processed += 1
#             if (batch_idx + 1) % self._log_interval == 0:
#                 logging.debug('Train Epoch: {} [{}]\tloss/batch: {:.6f}'.format(
#                     epoch, batch_idx, train_loss / n
#                 ))
#                 train_loss = 0.
#                 n = 0.
#         return {"loss": total_loss / total_batches}
#
#     def batch_loss(self, batch: Dict[str, torch.Tensor], for_training: bool) -> torch.Tensor:
#         """
#         Does a forward pass on the given batch and returns the ``loss`` value in the result.
#         If ``for_training`` is `True` also applies regularization penalty.
#         """
#         batch = util.move_to_device(batch, self.device.index)
#
#         output_dict = self.model(**batch)
#         if for_training:
#             # TODO: add regularization
#             pass
#         loss = output_dict['loss']
#         return loss
#
#     def train(self) -> Dict[str, Any]:
#         train_metrics: Dict[str, float] = {}
#         val_metrics: Dict[str, float] = {}
#         metrics: Dict[str, Any] = {}
#
#         epochs_trained = 0
#         validation_metric_per_epoch = []
#
#         for epoch in range(0, self._num_epochs):
#             epoch_start_time = time.time()
#             train_metrics = self._train_epoch(epoch)
#
#             if self._validation_data is not None:
#                 with torch.no_grad():
#                     # We have a validation set, so compute all the metrics on it.
#                     val_loss, num_batches = self._validation_loss()
#                     logging.debug("Validation loss: {:.6f}".format(val_loss))
#                     # val_metrics = self._get_metrics(val_loss, num_batches, reset=True)
#                     val_metrics = {'loss': val_loss / num_batches}
#
#                     # Check validation metric for early stopping
#                     this_epoch_val_metric = val_metrics['loss']
#
#                     # Check validation metric to see if it's the best so far
#                     is_best_so_far = self._is_best_so_far(this_epoch_val_metric, validation_metric_per_epoch)
#                     validation_metric_per_epoch.append(this_epoch_val_metric)
#
#             else:
#                 # No validation set, so just assume it's the best so far.
#                 is_best_so_far = True
#                 val_metrics = {}
#                 this_epoch_val_metric = None
#
#             epoch_elapsed_time = time.time() - epoch_start_time
#             logging.debug("Epoch duration: %s", time.strftime("%H:%M:%S",
#                                                               time.gmtime(epoch_elapsed_time)))
#
#             self._log_learning_rates()
#
#             metrics['epoch'] = epoch
#             metrics["training_epochs"] = epochs_trained
#             epochs_trained += 1
#
#             if self._learning_rate_scheduler:
#                 # The LRScheduler API is agnostic to whether your schedule requires a validation metric -
#                 # if it doesn't, the validation metric passed here is ignored.
#                 self._learning_rate_scheduler.step(this_epoch_val_metric, epoch)
#
#             # TODO: also log validation metrics
#             for metric_name, metric_value in train_metrics.items():
#                 logging.debug(f"Epoch {metric_name}: {metric_value}")
#
#             if is_best_so_far:
#                 # Update all the best_ metrics.
#                 # (Otherwise they just stay the same as they were.)
#                 metrics['best_epoch'] = epoch
#                 for key, value in val_metrics.items():
#                     metrics["best_validation_" + key] = value
#             self._save_checkpoint(epoch, validation_metric_per_epoch, is_best=is_best_so_far)
#
#         return metrics
#
#     def _log_learning_rates(self):
#         names = {param: name for name, param in self.model.named_parameters()}
#         for group in self.optimizer.param_groups:
#             if 'lr' not in group:
#                 continue
#             rate = group['lr']
#             for param in group['params']:
#                 # check whether params has requires grad or not
#                 effective_rate = rate * float(param.requires_grad)
#                 logging.debug(f"learning_rate/{names[param]}: {effective_rate}")
#
#     def _is_best_so_far(self,
#                         this_epoch_val_metric: float,
#                         validation_metric_per_epoch: List[float]):
#         if not validation_metric_per_epoch:
#             return True
#         elif self._validation_metric_decreases:
#             return this_epoch_val_metric < min(validation_metric_per_epoch)
#         else:
#             return this_epoch_val_metric > max(validation_metric_per_epoch)
#
#     def _validation_loss(self) -> Tuple[float, int]:
#         """
#         Computes the validation loss. Returns it and the number of batches.
#         """
#         logger.debug("Validating")
#
#         self.model.eval()
#
#         val_iterator = self.iterator
#
#         val_generator = val_iterator(self._validation_data,
#                                      num_epochs=1,
#                                      shuffle=False)
#         batches_this_epoch = 0
#         val_loss = 0
#         for batch in val_generator:
#
#             loss = self.batch_loss(batch, for_training=False)
#             if loss is not None:
#                 # You shouldn't necessarily have to compute a loss for validation, so we allow for
#                 # `loss` to be None.  We need to be careful, though - `batches_this_epoch` is
#                 # currently only used as the divisor for the loss function, so we can safely only
#                 # count those batches for which we actually have a loss.  If this variable ever
#                 # gets used for something else, we might need to change things around a bit.
#                 batches_this_epoch += 1
#                 val_loss += loss.detach().cpu().numpy()
#
#             # Update the description with the latest metrics
#             # val_metrics = self._get_metrics(val_loss, batches_this_epoch)
#             # description = self._description_from_metrics(val_metrics)
#             # val_generator_tqdm.set_description(description, refresh=False)
#
#         return val_loss, batches_this_epoch
#
#     def _save_checkpoint(self,
#                          epoch: Union[int, str],
#                          val_metric_per_epoch: List[float],
#                          is_best: Optional[bool] = None) -> None:
#         """
#         Saves a checkpoint of the model to self._serialization_dir.
#         Is a no-op if self._serialization_dir is None.
#
#         Parameters
#         ----------
#         epoch : Union[int, str], required.
#             The epoch of training.  If the checkpoint is saved in the middle
#             of an epoch, the parameter is a string with the epoch and timestamp.
#         is_best: bool, optional (default = None)
#             A flag which causes the model weights at the given epoch to
#             be copied to a "best.th" file. The value of this flag should
#             be based on some validation metric computed by your model.
#         """
#         if self._serialization_dir is not None:
#             if not self._serialization_dir.exists():
#                 self._serialization_dir.mkdir(parents=True, exist_ok=False)
#             model_path = self._serialization_dir / "model_state_epoch_{}.th".format(epoch)
#             model_state = self.model.state_dict()
#             torch.save(model_state, model_path)
#
#             training_state = {'epoch': epoch,
#                               'val_metric_per_epoch': val_metric_per_epoch,
#                               'optimizer': self.optimizer.state_dict()}
#             # if self._learning_rate_scheduler is not None:
#             #     training_state["learning_rate_scheduler"] = \
#             #         self._learning_rate_scheduler.lr_scheduler.state_dict()
#             training_path = self._serialization_dir / "training_state_epoch_{}.th".format(epoch)
#             torch.save(training_state, training_path)
#             if is_best:
#                 logger.info("Best validation performance so far. "
#                             "Copying weights to '%s/best.th'.", self._serialization_dir)
#                 shutil.copyfile(model_path, self._serialization_dir / "best.th")
#
#             # if self._num_serialized_models_to_keep and self._num_serialized_models_to_keep >= 0:
#             #     self._serialized_paths.append([time.time(), model_path, training_path])
#             #     if len(self._serialized_paths) > self._num_serialized_models_to_keep:
#             #         paths_to_remove = self._serialized_paths.pop(0)
#             #         # Check to see if we should keep this checkpoint, if it has been longer
#             #         # then self._keep_serialized_model_every_num_seconds since the last
#             #         # kept checkpoint.
#             #         remove_path = True
#             #         if self._keep_serialized_model_every_num_seconds is not None:
#             #             save_time = paths_to_remove[0]
#             #             time_since_checkpoint_kept = save_time - self._last_permanent_saved_checkpoint_time
#             #             if time_since_checkpoint_kept > self._keep_serialized_model_every_num_seconds:
#             #                 # We want to keep this checkpoint.
#             #                 remove_path = False
#             #                 self._last_permanent_saved_checkpoint_time = save_time
#             #         if remove_path:
#             #             for fname in paths_to_remove[1:]:
#             #                 os.remove(fname)