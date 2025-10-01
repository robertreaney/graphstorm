"""
    Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

    Licensed under the Apache License, Version 2.0 (the "License").
    You may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    GNN inferrer for multi-task learning in GraphStorm
"""
import os
import time
import logging
from typing import Any, Dict, Optional
import plyvel
from pathlib import Path

import torch as th

from ..config import (BUILTIN_TASK_NODE_CLASSIFICATION,
                      BUILTIN_TASK_NODE_REGRESSION,
                      BUILTIN_TASK_EDGE_CLASSIFICATION,
                      BUILTIN_TASK_EDGE_REGRESSION)
from ..dataloading import GSgnnMultiTaskDataLoader
from ..eval.evaluator import GSgnnMultiTaskEvaluator
from .graphstorm_infer import GSInferrer
from ..model.utils import save_full_node_embeddings as save_gsgnn_embeddings
from ..model.utils import (save_node_prediction_results,
                           save_edge_prediction_results,
                           save_relation_embeddings)
from ..model.utils import NodeIDShuffler
from ..model import (do_full_graph_inference,
                     do_mini_batch_inference,
                     gen_emb_for_nfeat_reconstruct)
from ..model.multitask_gnn import multi_task_mini_batch_predict
from ..model.lp_gnn import run_lp_mini_batch_predict

from ..model.edge_decoder import LinkPredictDistMultDecoder

from ..utils import sys_tracker, get_rank, barrier
from ..trainer.resonate_trainer import resonate_mini_batch_gnn_predict, resonate_mini_batch_gnn_predict_wild


def get_predictions(model, mt_loader, use_mini_batch_infer=True, database=None, save_prediction_path=None, node_id_mapping_file=None):
    if not use_mini_batch_infer:
        raise NotImplementedError("Full graph inference is not supported yet.")
    test_start = time.time()
    sys_tracker.check('before prediction')
    model.eval()

    loaders = mt_loader.dataloaders
    task_infos = mt_loader.task_infos
    results = dict()

    for loader, task_info in zip(loaders, task_infos):
        if database is not None:
            pred, label = resonate_mini_batch_gnn_predict(model, loader, task_info.task_id, True, database=database)
        else:
            pred = resonate_mini_batch_gnn_predict_wild(model=model, loader=loader, task_id=task_info.task_id, save_prediction_path=save_prediction_path, node_id_mapping_file=node_id_mapping_file)
            label = {}
        results[task_info.task_id] = (pred, label)

    return results


class ResonateInferrer(GSInferrer):
    """ Multi task inferrer.

    This is a high-level inferrer wrapper that can be used directly
    to do multi task model inference.

    Parameters
    ----------
    model : GSgnnMultiTaskModel
        The GNN model for prediction.
    """
    def __init__(self, model, part_config=None, cached_labels=True):
        super().__init__(model)
        

        if cached_labels:
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
        else:
            self.db = None


    # pylint: disable=unused-argument
    def infer_safe(self, data,
              predict_test_loader: Optional[GSgnnMultiTaskDataLoader] = None,
              save_embed_path=None,
              save_prediction_path=None,
              node_id_mapping_file=None,
              return_proba=True,
              save_embed_format="pytorch",
              infer_batch_size=1024):

        do_eval = self.evaluator is not None
        sys_tracker.check('start inferencing')
        model = self._model
        model.eval()

        # All the tasks share the same GNN encoder so the fanouts are same
        # for different tasks.
        fanout = None
        if predict_test_loader is not None:
            for task_fanout in predict_test_loader.fanout:
                if task_fanout is not None:
                    fanout = task_fanout
                    break
        else:
            raise ValueError("All the test data loaders are None.")

        sys_tracker.check('compute embeddings')
        device = self.device

        g = data.g
        # Note(xiangsx): Save embeddings should happen
        # before conducting prediction results.
        
        barrier()

        # As re-computing node embeddings, for reconstruct node
        # feature evaluation and link prediction evaluation,
        # will directly update the underlying DistTensors,
        # we have to do the evaluation (prediction) of each
        # task in the following priority:
        # 1. node and edge prediction tasks (classificaiton/regression)
        # 2. node feature reconstruction (as it has the chance
        #    to reuse the node embeddings generated at the beginning)
        # 3. link prediction.
        pre_results: Dict[str, Any] = {}
        test_lengths = None

        if predict_test_loader is None:
            logging.warning("There is no prediction tasks."
                            "Will skip saving prediction results.")
            return

        logging.info("Saving prediction results")

        if predict_test_loader is not None:
            # compute prediction results for node classification,
            # node regressoin, edge classification
            # and edge regression tasks.
            get_predictions(model, predict_test_loader, database=self.db, save_prediction_path=save_prediction_path, node_id_mapping_file=node_id_mapping_file)

        return


    # pylint: disable=unused-argument
    def infer(self, data,
              predict_test_loader: Optional[GSgnnMultiTaskDataLoader] = None,
              save_embed_path=None,
              save_prediction_path=None,
              node_id_mapping_file=None,
              return_proba=True,
              save_embed_format="pytorch",
              infer_batch_size=1024):

        do_eval = self.evaluator is not None
        sys_tracker.check('start inferencing')
        model = self._model
        model.eval()

        # All the tasks share the same GNN encoder so the fanouts are same
        # for different tasks.
        fanout = None
        if predict_test_loader is not None:
            for task_fanout in predict_test_loader.fanout:
                if task_fanout is not None:
                    fanout = task_fanout
                    break
        else:
            raise ValueError("All the test data loaders are None.")

        sys_tracker.check('compute embeddings')
        device = self.device

        g = data.g
        # Note(xiangsx): Save embeddings should happen
        # before conducting prediction results.
        if save_embed_path is not None:
            raise NotImplementedError("Saving embeddings is not supported yet ")
            logging.info("Saving node embeddings")
            node_norm_methods = model.node_embed_norm_methods
            # Save the original embs first
            save_gsgnn_embeddings(g,
                                  save_embed_path,
                                  embs,
                                  node_id_mapping_file=node_id_mapping_file,
                                  save_embed_format=save_embed_format)
            barrier()
            for task_id, norm_method in node_norm_methods.items():
                if norm_method is None:
                    continue
                normed_embs = model.normalize_task_node_embs(task_id, embs, inplace=False)
                save_embed_path = os.path.join(save_embed_path, task_id)
                save_gsgnn_embeddings(g,
                                      save_embed_path,
                                      normed_embs,
                                      node_id_mapping_file=node_id_mapping_file,
                                      save_embed_format=save_embed_format)
            sys_tracker.check('save embeddings')

            # save relation embedding if any for link prediction tasks
            if get_rank() == 0:
                decoders = model.task_decoders
                for task_id, decoder in decoders.items():
                    if isinstance(decoder, LinkPredictDistMultDecoder):
                        rel_emb_path = os.path.join(save_embed_path, task_id)
                        os.makedirs(rel_emb_path, exist_ok=True)
                        save_relation_embeddings(rel_emb_path, decoder)

        barrier()

        # As re-computing node embeddings, for reconstruct node
        # feature evaluation and link prediction evaluation,
        # will directly update the underlying DistTensors,
        # we have to do the evaluation (prediction) of each
        # task in the following priority:
        # 1. node and edge prediction tasks (classificaiton/regression)
        # 2. node feature reconstruction (as it has the chance
        #    to reuse the node embeddings generated at the beginning)
        # 3. link prediction.
        pre_results: Dict[str, Any] = {}
        test_lengths = None
        if predict_test_loader is not None:
            # compute prediction results for node classification,
            # node regressoin, edge classification
            # and edge regression tasks.
            pre_results = \
                get_predictions(model, predict_test_loader, database=self.db)

        if do_eval and self.db is not None:
            test_start = time.time()
            assert isinstance(self.evaluator, GSgnnMultiTaskEvaluator)

            val_score, test_score = self.evaluator.evaluate(
                pre_results,
                None,
                0,
            )

            sys_tracker.check('run evaluation')
            if get_rank() == 0:
                self.log_print_metrics(val_score=test_score,
                                       test_score=val_score,
                                       dur_eval=time.time() - test_start,
                                       total_steps=0)

        if save_prediction_path is not None:
            if predict_test_loader is None:
                logging.warning("There is no prediction tasks."
                                "Will skip saving prediction results.")
                return

            logging.info("Saving prediction results")
            target_ntypes = set()
            task_infos = predict_test_loader.task_infos
            dataloaders = predict_test_loader.dataloaders
            for task_info in task_infos:
                if task_info.task_type in \
                    [BUILTIN_TASK_NODE_CLASSIFICATION, BUILTIN_TASK_NODE_REGRESSION]:
                    target_ntypes.add(task_info.task_config.target_ntype)
                else:
                    # task_info.task_type is BUILTIN_TASK_LINK_PREDICTION
                    # or BUILTIN_TASK_RECONSTRUCT_NODE_FEAT
                    # There is no prediction results.
                    continue

            nid_shuffler = NodeIDShuffler(g, node_id_mapping_file, list(target_ntypes)) \
                    if node_id_mapping_file else None

            for task_info, dataloader in zip(task_infos, dataloaders):
                task_id = task_info.task_id
                if task_id not in pre_results:
                    logging.debug("No Prediction results for %s",
                                  task_id)
                    continue

                # Save prediction results
                save_pred_path = os.path.join(save_prediction_path, task_id)
                if task_info.task_type in \
                    [BUILTIN_TASK_NODE_CLASSIFICATION, BUILTIN_TASK_NODE_REGRESSION]:
                    pred, labels = pre_results[task_id]
                    if pred is not None:
                        shuffled_preds = {}

                        target_ntype = task_info.task_config.target_ntype
                        pred_nids = dataloader.target_nidx[target_ntype]
                        if node_id_mapping_file is not None:
                            pred_nids = nid_shuffler.shuffle_nids(
                                target_ntype, pred_nids)

                        shuffled_preds[target_ntype] = (pred, pred_nids, labels)
                        save_node_prediction_results(shuffled_preds, save_pred_path)
                else:
                    # There is no prediction results for link prediction
                    # and feature reconstruction
                    continue

        return
