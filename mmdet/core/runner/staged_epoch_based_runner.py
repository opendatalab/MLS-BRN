"""The StagedEpochBasedRunner is used for supporting semi-supervised learning."""

import time

import numpy as np
import torch
from mmcv.parallel import DataContainer as DC
from mmcv.runner import RUNNERS, EpochBasedRunner


@RUNNERS.register_module()
class StagedEpochBasedRunner(EpochBasedRunner):
    """The runner class supports semi-supervised learning."""

    def __init__(self, *args, **kwargs):
        assert "supervised_epochs" in kwargs
        self._supervised_epochs = kwargs.pop("supervised_epochs")
        super().__init__(*args, **kwargs)

    @property
    def supervised_epochs(self):
        """int: Supervised training epochs."""
        return self._supervised_epochs

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = "train"
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(self.data_loader)
        ssl_flag = int(self.epoch >= self.supervised_epochs)
        self.call_hook("before_train_epoch")
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            batch_size = data_batch["img"].data[0].shape[0]
            is_semi_supervised_stage = DC([[torch.from_numpy(np.array([ssl_flag]))] * batch_size])
            data_batch["is_semi_supervised_stage"] = is_semi_supervised_stage
            self.data_batch = data_batch
            self._inner_iter = i
            self.call_hook("before_train_iter")
            self.run_iter(data_batch, train_mode=True, **kwargs)
            self.call_hook("after_train_iter")
            del self.data_batch
            self._iter += 1

        self.call_hook("after_train_epoch")
        self._epoch += 1
