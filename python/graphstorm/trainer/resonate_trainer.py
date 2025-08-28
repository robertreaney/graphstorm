"""
    Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.

    Licensed under the Apache License, Version 2.0 (the "License").
    You may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    GraphStorm trainer for multi-task learning.
"""
from ctypes import ArgumentError
import os
import time
import resource
import logging
import torch as th
from pathlib import Path
import plyvel
import numpy as np
from torch.nn.parallel import DistributedDataParallel
import optuna
from concurrent.futures import ThreadPoolExecutor

from ..config import BUILTIN_TASK_NODE_CLASSIFICATION
from ..eval.evaluator import GSgnnMultiTaskEvaluator
from ..model import (GSgnnModelBase, GSgnnModel,
                     GSgnnMultiTaskModelInterface)
from .gsgnn_trainer import GSgnnTrainer

from ..utils import sys_tracker, rt_profiler, print_mem, get_rank
from ..utils import barrier, is_distributed
from ..model.gnn_encoder_base import prepare_for_wholegraph
from ..model.utils import append_to_dict
from ..utils import is_distributed, get_rank, is_wholegraph



def get_cached_labels(database, fake_labels, device):
    db = database.prefixed_db(b'label-')
    f = lambda x: np.frombuffer(db.get(str(x.item()).encode()))
    with ThreadPoolExecutor() as exe:
        labels = list(exe.map(f, fake_labels))
    return th.tensor(np.array(labels, dtype=np.int32), dtype=th.int32).to(device)

def resonate_mini_batch_gnn_predict(model, loader, task_id, return_proba=True, return_label=True, database=None):
    """ Perform mini-batch prediction on a GNN model.

    Parameters
    ----------
    model : GSgnnModel
        The GraphStorm GNN model
    loader : GSgnnNodeDataLoader
        The GraphStorm dataloader
    return_proba : bool
        Whether or not to return all the predictions or the maximum prediction
    return_label : bool
        Whether or not to return labels

    Returns
    -------
    dict of Tensor :
        GNN prediction results. Return all the results when return_proba is true
        otherwise return the maximum result.
    dict of Tensor : GNN embeddings.
    dict of Tensor : labels if return_labels is True
    """
    if get_rank() == 0:
        logging.debug("Perform mini-batch inference for resonate prediction.")
    device = model.device
    data = loader.data
    g = data.g
    preds = {}
    target_ntypes = set(loader.target_nidx.keys())

    if return_label:
        assert loader.label_field is not None, \
            "Return label is required, but the label field is not provided when" \
            "initlaizing the loader."

    labels = {}

    len_dataloader = max_num_batch = len(loader)
    tensor = th.tensor([len_dataloader], device=device)
    if is_distributed():
        th.distributed.all_reduce(tensor, op=th.distributed.ReduceOp.MAX)
        max_num_batch = tensor[0]

    dataloader_iter = iter(loader)

    with th.no_grad():
        # WholeGraph does not support imbalanced batch numbers across processes/trainers
        # TODO (IN): Fix dataloader to have the same number of minibatches
        for iter_l in range(max_num_batch):
            iter_start = time.time()
            tmp_keys = []
            blocks = None
            if iter_l < len_dataloader:
                input_nodes, seeds, blocks = next(dataloader_iter)
                if not isinstance(input_nodes, dict):
                    assert len(g.ntypes) == 1
                    input_nodes = {g.ntypes[0]: input_nodes}
            if is_wholegraph():
                tmp_keys = [ntype for ntype in g.ntypes if ntype not in input_nodes]
                prepare_for_wholegraph(g, input_nodes)
            nfeat_fields = loader.node_feat_fields
            node_input_feats = data.get_node_feats(input_nodes, nfeat_fields, device)
            # Since v0.4, add the edge features as one input
            # efeat_fields = loader.edge_feat_fields
            # edge_input_feats = data.get_blocks_edge_feats(blocks, efeat_fields, device)

            if blocks is None:
                continue
            # Remove additional keys (ntypes) added for WholeGraph compatibility
            for ntype in tmp_keys:
                del input_nodes[ntype]
            blocks = [block.to(device) for block in blocks]
            
            
            encoder_data = (blocks, node_input_feats, '', input_nodes)
            mini_batch = (encoder_data, '')
            pred = model.predict(task_id, mini_batch, return_proba)


            label_field = loader.label_field
            label = data.get_node_feats(seeds, label_field)
            for k, v in label.items():
                label[k] = get_cached_labels(database, v, device)
            
            
            if return_label:
                append_to_dict(label, labels)

            # pred can be a Tensor or a dict of Tensor
            # emb can be a Tensor or a dict of Tensor
            if isinstance(pred, dict):
                append_to_dict(pred, preds)
            else:
                assert len(seeds) == 1, \
                    f"Expect prediction results of multiple node types {label.keys()}" \
                    f"But only get results of one node type"
                ntype = list(seeds.keys())[0]
                append_to_dict({ntype: pred}, preds)

            if get_rank() == 0 and iter_l % 20 == 0:
                logging.debug("iter %d out of %d: takes %.3f seconds",
                              iter_l, max_num_batch, time.time() - iter_start)

    # MFG for DGL 2.0.0+ return all node and edge type
    preds = {
        ntype: th.cat(preds[ntype])
        for ntype in preds if ntype in target_ntypes
    }
    for ntype, ntype_label in labels.items():
        labels[ntype] = th.cat(ntype_label)
    target_ntype = list(target_ntypes)[0]
    return preds[target_ntype], labels[target_ntype]



class ResonateMultiTaskTrainer(GSgnnTrainer):
    r""" A trainer for multi-task learning

    This class is used to train models for multi-task learning.

    It makes use of the functions provided by `GSgnnTrainer`
    to define two main functions: `fit` that performs the training
    for the model that is provided when the object is created,
    and `eval` that evaluates a provided model against test and
    validation data.

    Parameters
    ----------
    model : GSgnnMultiTaskModel
        The GNN model for prediction.
    topk_model_to_save : int
        The top K model to save.

    .. versionchanged:: 0.4.0
        Add support for edge feature reconstruction tasks.
    """
    def __init__(self, model, part_config, topk_model_to_save=1):
        super(ResonateMultiTaskTrainer, self).__init__(model, topk_model_to_save)
        assert isinstance(model, GSgnnMultiTaskModelInterface) \
            and isinstance(model, GSgnnModelBase), \
                "The input model is not a GSgnnModel model "\
                "or not implement the GSgnnMultiTaskModelInterface." \
                "Please implement GSgnnModelBase."
        
        if get_rank() == 0:
            self.labels_path = Path(part_config).parent / 'levelsdb'
        else:
            self.labels_path = Path(part_config).parent / f'levelsdb{get_rank()}'
        
        try:
            self.db = plyvel.DB(
                self.labels_path.as_posix(),
                create_if_missing=False,  # Read-only, don't create
                error_if_exists=False,     # We expect it to exist
                paranoid_checks=False,     # Skip checks for read-only
                write_buffer_size=0,       # No write buffer needed for read-only
                lru_cache_size= 5 * 1024 * 1024 * 1024,  # 5GB cache per process
            )
            logging.info(f"Opened LevelDB at {self.labels_path} for rank {get_rank()}")
        except Exception as e:
            logging.error(f"Could not open LevelDB at {self.labels_path}: {e}")
            raise e

    def __del__(self):
        """Clean up LevelDB connection when trainer is destroyed."""
        if hasattr(self, 'db') and self.db is not None:
            try:
                self.db.close()
            except:
                pass

    def _prepare_mini_batch(self, data, task_info, mini_batch, device):
        """ prepare mini batch for a single task

        Parameters
        ----------
        data: GSgnnData
            Graph data
        model: GSgnnModel
            Model
        task_info: TaskInfo
            Task meta information
        mini_batch: tuple
            Mini-batch info
        device: torch.device
            Device

        Return
        ------
        tuple: mini-batch
        """

        if task_info.task_type in \
            [BUILTIN_TASK_NODE_CLASSIFICATION]:
            g = data.g
            input_nodes, seeds, blocks = mini_batch
            if not isinstance(input_nodes, dict):
                # This happens on a homogeneous graph.
                assert len(g.ntypes) == 1, \
                    "The graph should be a homogeneous graph, " \
                    f"but it has multiple node types {g.ntypes}"
                input_nodes = {g.ntypes[0]: input_nodes}

            nfeat_fields = task_info.dataloader.node_feat_fields
            label_field = task_info.dataloader.label_field
            input_feats = data.get_node_feats(input_nodes, nfeat_fields, device)
            lbl = data.get_node_feats(seeds, label_field, device)
            
            # RESONATE grab from kv cache
            for k, v in lbl.items():
                lbl[k] = get_cached_labels(self.db, v, device)

            blocks = [block.to(device) for block in blocks] \
                if blocks is not None else None

            # Order follow GSgnnNodeModelInterface.forward
            # TODO: we don't support edge features for now.
            return (blocks, input_feats, None, lbl, input_nodes)
        else:
            raise TypeError(f"Unsupported task {task_info}. Resonate trainer supports only node classification", )

    # pylint: disable=unused-argument
    def fit(self, train_loader,
            num_epochs,
            val_loader=None,
            test_loader=None,
            use_mini_batch_infer=True,
            save_model_path=None,
            save_model_frequency=-1,
            save_perf_results_path=None,
            freeze_input_layer_epochs=0,
            max_grad_norm=None,
            grad_norm_type=2.0,
            is_optuna_run=False,
            optuna_trial=None):
        """ The fit function for multi-task learning.

        Performs the training for `self.model`. Iterates over all the tasks
        and run one mini-batch for each task in an iteration. The loss will be
        accumulated. Performs the backwards step using `self.optimizer`.
        If an evaluator has been assigned to the trainer, it will run evaluation
        at the end of every epoch.

        Parameters
        ----------
        train_loader : GSgnnMultiTaskDataLoader
            The mini-batch sampler for training.
        num_epochs : int
            The max number of epochs to train the model.
        val_loader : GSgnnMultiTaskDataLoader
            The mini-batch sampler for computing validation scores. The validation scores
            are used for selecting models.
        test_loader : GSgnnMultiTaskDataLoader
            The mini-batch sampler for computing test scores.
        use_mini_batch_infer : bool
            Whether or not to use mini-batch inference.
        save_model_path : str
            The path where the model is saved.
        save_model_frequency : int
            The number of iteration to train the model before saving the model.
        save_perf_results_path : str
            The path of the file where the performance results are saved.
            TODO(xiangsx): Add support for saving performance results on disk.
            Reserved for future.
        freeze_input_layer_epochs: int
            Freeze the input layer for N epochs. This is commonly used when
            the input layer contains language models.
            Default: 0, no freeze.
        max_grad_norm: float
            Clip the gradient by the max_grad_norm to ensure stability.
            Default: None, no clip.
        grad_norm_type: float
            Norm type for the gradient clip
            Default: 2.0
        """
        # Check the correctness of configurations.
        if self.evaluator is not None:
            assert val_loader is not None, \
                    "The evaluator is provided but validation set is not provided."
        if not use_mini_batch_infer:
            assert isinstance(self._model, GSgnnModel), \
                    "Only GSgnnModel supports full-graph inference."

        # with freeze_input_layer_epochs is 0, computation graph will not be changed.
        on_cpu = self.device == th.device('cpu')
        if is_distributed():
            model = DistributedDataParallel(self._model,
                                            device_ids=None if on_cpu else [self.device],
                                            output_device=None if on_cpu else self.device,
                                            find_unused_parameters=True,
                                            static_graph=False)
        else:
            model = self._model
        device = model.device
        data = train_loader.data

        # Preparing input layer for training or inference.
        # The input layer can pre-compute node features in the preparing step if needed.
        # For example pre-compute all BERT embeddings
        if freeze_input_layer_epochs > 0:
            self._model.freeze_input_encoder(data)
        # TODO(xiangsx) Support freezing gnn encoder and decoder

        # training loop
        total_steps = 0
        early_stop = False
        sys_tracker.check('start training')
        for epoch in range(num_epochs):
            model.train()
            epoch_start = time.time()
            if freeze_input_layer_epochs <= epoch:
                self._model.unfreeze_input_encoder()
            # TODO(xiangsx) Support unfreezing gnn encoder and decoder

            rt_profiler.start_record()
            batch_tic = time.time()
            for i, task_mini_batches in enumerate(train_loader):
                rt_profiler.record('get_batches')
                total_steps += 1

                mini_batches = []
                for (task_info, mini_batch) in task_mini_batches:
                    mini_batches.append((task_info, \
                        self._prepare_mini_batch(data, task_info, mini_batch, device)))

                rt_profiler.record('model_forward')
                loss, task_losses = model(mini_batches)

                rt_profiler.record('train_forward')
                self.optimizer.zero_grad()
                loss.backward()
                rt_profiler.record('train_backward')
                self.optimizer.step()
                rt_profiler.record('train_step')

                if max_grad_norm is not None:
                    th.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, grad_norm_type)
                self.log_metric("Train loss", loss.item(), total_steps)

                if i % 20 == 0 and get_rank() == 0:
                    rt_profiler.print_stats()
                    per_task_loss = {}
                    for mini_batch, task_loss in zip(mini_batches, task_losses):
                        task_info, _ = mini_batch
                        per_task_loss[task_info.task_id] = task_loss[0].item()
                    logging.info("Epoch %05d | Batch %03d | Train Loss: %.4f | Time: %.4f",
                                 epoch, i, loss.item(), time.time() - batch_tic)
                    logging.debug("Per task Loss: %s", per_task_loss)

                val_score = None
                # if self.can_do_validation(val_loader) and self.evaluator.do_eval(total_steps):
                #     val_score = self.eval(model.module if is_distributed() else model,
                #                           data, val_loader, test_loader, total_steps)
                #     # TODO(xiangsx): Add early stop support

                # Every n iterations, save the model and keep
                # the last k models.
                # TODO(xiangsx): support saving the best top k model.
                if save_model_frequency > 0 and \
                    total_steps % save_model_frequency == 0 and \
                    total_steps != 0:
                    # if val_score is None:
                    #     # not in the same eval_frequncy iteration
                    #     if self.can_do_validation(val_loader):
                    #         # for model saving, force to do evaluation if can
                    #         val_score = self.eval(model.module if is_distributed() else model,
                    #                             data, val_loader, test_loader, total_steps)
                    # We will save the best model when
                    # 1. There is no evaluation, we will keep the
                    #    latest K models.
                    # 2. (TODO) There is evaluaiton, we need to follow the
                    #    guidance of validation score.
                    # So here reset val_score to be None
                    # TODO track the best model
                    val_score = None
                    self.save_topk_models(model, epoch, i, val_score, save_model_path)

                batch_tic = time.time()
                rt_profiler.record('train_eval')

            # ------- end of an epoch -------

            barrier()
            epoch_time = time.time() - epoch_start
            if get_rank() == 0:
                logging.info("Epoch %d take %.3f seconds", epoch, epoch_time)

            val_score = None
            # do evaluation and model saving after each epoch if can
            if self.can_do_validation(val_loader):

                val_score = self.eval(model.module if is_distributed() else model, val_loader, test_loader, total_steps, use_mini_batch_infer=use_mini_batch_infer)
                if self.evaluator.do_early_stop(val_score):
                    early_stop = True
                    
                if is_optuna_run:
                    optuna_trial.report(self.evaluator._get_early_stop_score(val_score), step=epoch)
                    if optuna_trial.should_prune():
                        raise optuna.TrialPruned()


            # After each epoch, check to save the top k models.
            # Will either save the last k model or all models
            # depends on the setting of top k.
            self.save_topk_models(model, epoch, None, val_score, save_model_path)
            rt_profiler.print_stats()
            # make sure saving model finishes properly before the main process kills this training
            # barrier()

            if early_stop is True:
                break

        rt_profiler.save_profile()
        print_mem(device)
        if get_rank() == 0 and self.evaluator is not None:
            # final evaluation
            output = {'best_test_score': self.evaluator.best_test_score,
                       'best_val_score':self.evaluator.best_val_score,
                       'peak_GPU_mem_alloc_MB': th.cuda.max_memory_allocated(device) / 1024 / 1024,
                       'peak_RAM_mem_alloc_MB': \
                           resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                       'best validation iteration': \
                           self.evaluator.best_iter_num,
                       'best model path': \
                           self.get_best_model_path() if save_model_path is not None else None}
            # TODO why doesn't this actually return the path to minimum loss?
            self.log_params(output)

    def eval(self, model, mt_val_loader, mt_test_loader, total_steps,
        use_mini_batch_infer=True, return_proba=True):
        """ do the model evaluation using validation and test sets

        Parameters
        ----------
        model : Pytorch model
            The GNN model.
        data : GSgnnData
            The training dataset
        mt_val_loader: GSgnnMultiTaskDataLoader
            The dataloader for validation data
        mt_test_loader : GSgnnMultiTaskDataLoader
            The dataloader for test data.
        total_steps: int
            Total number of iterations.
        use_mini_batch_infer: bool
            Whether do mini-batch inference
        return_proba: bool
            Whether to return all the predictions or the maximum prediction.

        Returns
        -------
        dict: validation score
        """
        if not return_proba or not use_mini_batch_infer:
            raise ArgumentError(f"Invalid arguments return_proba={return_proba} use_mini_batch_infer={use_mini_batch_infer}")
        test_start = time.time()
        sys_tracker.check('before prediction')

        if mt_val_loader is None and mt_test_loader is None:
            # no need to do validation and test
            # do nothing.
            return None

        model.eval()

        val_dataloaders = mt_val_loader.dataloaders if mt_val_loader is not None else None
        test_dataloaders = mt_test_loader.dataloaders if mt_test_loader is not None else None
        task_infos = mt_val_loader.task_infos if mt_val_loader is not None else mt_test_loader.task_infos

        val_results = dict()
        test_results = dict()

        if val_dataloaders is None: 
            val_dataloaders = [None] * len(task_infos)

        if test_dataloaders is None:
            test_dataloaders = [None] * len(task_infos)
            test_results = None

        # val_results = {'node_classification-rcid-labels': (probs tensor, labels tensor), 'node_classification-hem-labels': (probs tensor, labels tensor)}


        for val_loader, test_loader, task_info \
            in zip(val_dataloaders, test_dataloaders, task_infos):
            pass

            if val_loader is None and test_loader is None:
                # For this task, these is no need to do compute test or val score
                # skip this task
                continue

            if use_mini_batch_infer:
                val_pred, val_label = resonate_mini_batch_gnn_predict(model, val_loader, task_info.task_id, return_proba, database=self.db)
                val_results[task_info.task_id] = (val_pred, val_label)
                
                sys_tracker.check('after_val_score')
                if test_loader is not None:
                    test_pred, test_label = \
                        resonate_mini_batch_gnn_predict(model, test_loader, return_proba,
                                                    return_label=True, database=self.db)
                    test_results[task_info.task_id] = (test_pred, test_label)
                else: # there is no test set
                    test_pred = None
                    test_label = None
                sys_tracker.check('after_test_score')

        sys_tracker.check('after_test_score')
        assert isinstance(self.evaluator, GSgnnMultiTaskEvaluator), \
            "Evaluator must be a GSgnnMultiTaskEvaluator"
        val_score, test_score = self.evaluator.evaluate(
                val_results, test_results, total_steps)
        sys_tracker.check('evaluate validation/test')
        model.train()

        if get_rank() == 0:
            self.log_print_metrics(val_score=val_score,
                                   test_score=test_score,
                                   dur_eval=time.time() - test_start,
                                   total_steps=total_steps)
        return val_score