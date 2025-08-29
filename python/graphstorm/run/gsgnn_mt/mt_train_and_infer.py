"""This script will do training & inference on best-fit model for multitask to avoid loading the model twice."""
import os
import os

from graphstorm.inference.resonate_infer import ResonateInferrer
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import logging
from shutil import copy2
from pathlib import Path
import graphstorm as gs
from graphstorm.config import GSConfig, get_argument_parser
from graphstorm.config import BUILTIN_TASK_LINK_PREDICTION
from graphstorm.dataloading import GSgnnMultiTaskDataLoader, GSgnnData
from graphstorm.model.multitask_gnn import GSgnnMultiTaskSharedEncoderModel
from graphstorm.trainer import ResonateMultiTaskTrainer, GSgnnMultiTaskLearningTrainer
from graphstorm.eval import GSgnnMultiTaskEvaluator
from graphstorm.inference import GSgnnMultiTaskLearningInferrer, ResonateInferrer
from graphstorm.utils import get_device, rt_profiler, sys_tracker, get_device, get_lm_ntypes
from graphstorm.run.gsgnn_mt.gsgnn_mt import create_task_train_dataloader, create_task_val_dataloader, create_task_test_dataloader
from graphstorm.model_introspection import save_mermaid_diagram

# FMT = "%(asctime)s %(levelname)s %(message)s"
# logging.basicConfig(format=FMT, level=logging.DEBUG)

# PART_CONFIG = ".data/debug/graph/debug.json"
# CONFIG_FILE = "experiments/debug/gnn.yaml"
# NUM_TRAINERS = 1
# NUM_SERVERS = 1

# config_args = Namespace(**{
#     'logging_level': 'info', 
#     'yaml_config_file': CONFIG_FILE, 
#     'local_rank': 0, 
#     'profile_path': None, 
#     'construct_feat_ntype': None, 
#     'part_config': PART_CONFIG, 
#     'save_model_path': '.',
#     'use_graphbolt': True
# })

def train(config_args, train_data):

    # reload in case things change
    config = GSConfig(config_args)
    config.verify_arguments(True)

    # task init #
    tasks = config.multi_tasks
    assert tasks is not None, \
        "The multi_task_learning configure block should not be empty."

    train_dataloaders = []
    val_dataloaders = []
    test_dataloaders = []

    for task in tasks:
        train_loader = create_task_train_dataloader(task, config, train_data)
        val_loader = create_task_val_dataloader(task, config, train_data)
        test_loader = create_task_test_dataloader(task, config, train_data)
        train_dataloaders.append(train_loader)
        val_dataloaders.append(val_loader)
        test_dataloaders.append(test_loader)

    train_dataloader = GSgnnMultiTaskDataLoader(train_data, tasks, train_dataloaders)
    val_dataloader = GSgnnMultiTaskDataLoader(train_data, tasks, val_dataloaders)
    test_dataloader = GSgnnMultiTaskDataLoader(train_data, tasks, test_dataloaders)

    ### MODEL TRAINING ###                        
    model = GSgnnMultiTaskSharedEncoderModel(config.alpha_l2norm, config.use_model_residuals)
    gs.gsf.set_encoder(model, train_data.g, config, train_task=True)

    task_evaluators = {}
    encoder_out_dims = model.gnn_encoder.out_dims \
        if model.gnn_encoder is not None \
            else model.node_input_encoder.out_dims
    for task in tasks:
        decoder, loss_func = gs.create_task_decoder(task,
                                                    train_data.g,
                                                    encoder_out_dims,
                                                    train_task=True)
        # For link prediction, lp_embed_normalizer may be used
        # TODO(xiangsx): add embed normalizer for other task types
        # in the future.
        node_embed_norm_method = task.task_config.lp_embed_normalizer \
            if task.task_type in [BUILTIN_TASK_LINK_PREDICTION] \
            else None
        model.add_task(task.task_id,
                        task.task_type,
                        decoder,
                        loss_func,
                        embed_norm_method=node_embed_norm_method)
        if not config.no_validation:
            if val_loader is None:
                logging.warning("The training data do not have validation set.")
            if test_loader is None:
                logging.warning("The training data do not have test set.")

            if val_loader is None and test_loader is None:
                logging.warning("Task %s does not have validation and test sets.", task.task_id)
            else:
                task_evaluators[task.task_id] = \
                    gs.create_evaluator(task)


    model.init_optimizer(lr=config.lr,
                            sparse_optimizer_lr=config.sparse_optimizer_lr,
                            weight_decay=config.wd_l2norm,
                            lm_lr=config.lm_tune_lr)
    
    trainer = ResonateMultiTaskTrainer(model, topk_model_to_save=config.topk_model_to_save, part_config=config.part_config)
    # trainer = GSgnnMultiTaskLearningTrainer(model, topk_model_to_save=config.topk_model_to_save)
    if not config.no_validation:
        evaluator = GSgnnMultiTaskEvaluator(config.eval_frequency,
                                            task_evaluators,
                                            use_early_stop=config.use_early_stop)
        trainer.setup_evaluator(evaluator)
    if config.restore_model_path is not None:
        trainer.restore_model(model_path=config.restore_model_path,
                                model_layer_to_load=config.restore_model_layers)
    trainer.setup_device(device=get_device())

    # Preparing input layer for training or inference.
    # The input layer can pre-compute node features in the preparing step if needed.
    # For example pre-compute all BERT embeddings
    model.prepare_input_encoder(train_data)
    if config.save_model_path is not None:
        save_model_path = config.save_model_path
    elif config.save_embed_path is not None:
        # If we need to save embeddings, we need to save the model somewhere.
        save_model_path = os.path.join(config.save_embed_path, "model")
    else:
        save_model_path = None

    # save the generating configuration for this experiment
    Path(save_model_path).parent.mkdir(parents=True, exist_ok=True)
    copy2(config.yaml_paths, Path(save_model_path).parent / 'config.yaml')

    tracker = gs.create_builtin_task_tracker(config)
    if gs.get_rank() == 0:
        # save some model summary stuff
        try:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        except Exception:
            total_params = None
            trainable_params = None

        logging.info("Model summary:\n%s", model)
        if total_params is not None:
            logging.info("Parameters | total: %s | trainable: %s",
                         f"{total_params:,}", f"{trainable_params:,}")
            
            # print parameters per child module
            for name, child in model.named_children():
                params = [p for p in child.parameters() if p.requires_grad]
                n_params = sum(p.numel() for p in params)
                logging.info("  %-30s params=%s", name, f"{n_params:,}")
                
        logging.info("Configured tasks: %s", ", ".join([t.task_id for t in tasks]))

        # test out mermaid diagram
        try:
            save_mermaid_diagram(model, Path(save_model_path).parent / "model_diagram.md", tasks)
            logging.info("Saved model diagram to model_diagram.md")
        except Exception as e:
            logging.warning("Failed to save model diagram: %s", e)

        tracker.log_params(config.__dict__)
    trainer.setup_task_tracker(tracker)

    # trainer.fit(train_loader=train_dataloader,
    #             val_loader=val_dataloader,
    #             # test_loader=train_dataloader,
    #             num_epochs=config.num_epochs,
    #             save_model_path=save_model_path,
    #             use_mini_batch_infer=config.use_mini_batch_infer,
    #             save_model_frequency=config.save_model_frequency,
    #             save_perf_results_path=config.save_perf_results_path,
    #             freeze_input_layer_epochs=config.freeze_lm_encoder_epochs,
    #             max_grad_norm=config.max_grad_norm,
    #             grad_norm_type=config.grad_norm_type)

    return model, task_evaluators, test_dataloader, trainer

def main(config_args):
    if not bool(os.environ['RESONATE']):
        raise EnvironmentError('need resonate environment for custom code')
    ## main from training script graphstorm.
    config = GSConfig(config_args)
    config.verify_arguments(True)

    gs.initialize(ip_config=config.ip_config, backend=config.backend,
                    local_rank=config.local_rank,
                    use_wholegraph=config.use_wholegraph_embed or False,
                    use_graphbolt=config.use_graphbolt)


    rt_profiler.init(config.profile_path, rank=gs.get_rank())
    sys_tracker.init(config.verbose, rank=gs.get_rank())


    ## LOAD DATA ##
    # THIS IS THE EXPENSIVE CALL
    train_data = GSgnnData(config.part_config,
                            node_feat_field=config.node_feat_name,
                            edge_feat_field=config.edge_feat_name,
                            lm_feat_ntypes=get_lm_ntypes(config.node_lm_configs))

    ##########################################
    ##########################################
    ##########################################
    ##########################################
    ############  below this is fast   #######
    ##########################################
    ##########################################
    ##########################################
    ##########################################

    model, task_evaluators, test_dataloader, trainer = train(config_args, train_data)
    del trainer

    #### LOAD BEST MODEL
    # TODO fetch best path here
    model.restore_model('./.data/hem_rcid/output/hma/save/epoch-15', model_layer_to_load=config.restore_model_layers)
    # model.restore_model(trainer.get_best_model_path(), model_layer_to_load=config.restore_model_layers)
    model = model.to(get_device())

    ### INFERENCE CODE
    infer = ResonateInferrer(model, labels_path='.data/hem_rcid/graph/levelsdb')

    if not config.no_validation:
        evaluator = GSgnnMultiTaskEvaluator(config.eval_frequency, task_evaluators)
        infer.setup_evaluator(evaluator)

    infer.setup_device(device=get_device())
    infer.infer(train_data,
                test_dataloader, 
                save_embed_path=config.save_embed_path,
                save_prediction_path=config.save_prediction_path,
                node_id_mapping_file=config.node_id_mapping_file,
                return_proba=config.return_proba,
                save_embed_format=config.save_embed_format)

    
def generate_parser():
    """ Generate an argument parser
    """
    parser = get_argument_parser()
    return parser

if __name__ == '__main__':
    arg_parser = generate_parser()

    # Ignore unknown args to make script more robust to input arguments
    gs_args, unknown_args = arg_parser.parse_known_args()
    logging.warning("Unknown arguments for command "
                    "graphstorm.run.gs_multi_task_learning: %s",
                    unknown_args)
    main(gs_args)
