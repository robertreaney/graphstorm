"""This script will do training & inference on best-fit model for multitask to avoid loading the model twice."""
import torch
import os
import logging
from shutil import copy2
from pathlib import Path
import graphstorm as gs
from graphstorm.config import GSConfig, get_argument_parser
from graphstorm.config import BUILTIN_TASK_LINK_PREDICTION
from graphstorm.dataloading import GSgnnMultiTaskDataLoader, GSgnnData
from graphstorm.model.multitask_gnn import GSgnnMultiTaskSharedEncoderModel
from graphstorm.trainer import GSgnnMultiTaskLearningTrainer
from graphstorm.eval import GSgnnMultiTaskEvaluator
from graphstorm.inference import GSgnnMultiTaskLearningInferrer
from graphstorm.utils import get_device, rt_profiler, sys_tracker, get_device, get_lm_ntypes
from graphstorm.run.gsgnn_mt.gsgnn_mt import create_task_train_dataloader, create_task_val_dataloader, create_task_test_dataloader
from graphstorm.model_introspection import save_mermaid_diagram
import optuna
from functools import partial

PATH = "/home/ubuntu/src/data-science-research/.data/optimization/0"
STORAGE_PATH = "sqlite:///.data/study_name.db"

def init_model(config, train_data):
    model = GSgnnMultiTaskSharedEncoderModel(config.alpha_l2norm, config.use_model_residuals)
    gs.gsf.set_encoder(model, train_data.g, config, train_task=True)

    encoder_out_dims = model.gnn_encoder.out_dims \
        if model.gnn_encoder is not None \
            else model.node_input_encoder.out_dims
    for task in config.multi_tasks:
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

    model.init_optimizer(lr=config.lr,
                            sparse_optimizer_lr=config.sparse_optimizer_lr,
                            weight_decay=config.wd_l2norm,
                            lm_lr=config.lm_tune_lr)

    # Preparing input layer for training or inference.
    # The input layer can pre-compute node features in the preparing step if needed.
    # For example pre-compute all BERT embeddings
    model.prepare_input_encoder(train_data)

    return model

def init_evaluation(config, train_data):
    task_evaluators = {}
    train_dataloaders = []
    val_dataloaders = []
    test_dataloaders = []

    for task in config.multi_tasks:
        train_loader = create_task_train_dataloader(task, config, train_data)
        val_loader = create_task_val_dataloader(task, config, train_data)
        test_loader = create_task_test_dataloader(task, config, train_data)
        train_dataloaders.append(train_loader)
        val_dataloaders.append(val_loader)
        test_dataloaders.append(test_loader)
    
        # separated this code from model initialization
        if not config.no_validation:
            if val_loader is None:
                raise ValueError("The training data do not have validation set.")
            if test_loader is None:
                logging.warning("The training data do not have test set.")

            if val_loader is None and test_loader is None:
                logging.warning("Task %s does not have validation and test sets.", task.task_id)
            else:
                task_evaluators[task.task_id] = \
                    gs.create_evaluator(task)
        else:
            raise ValueError("Tune mode requires validation to be configured.")


    train_dataloader = GSgnnMultiTaskDataLoader(train_data, config.multi_tasks, train_dataloaders)
    val_dataloader = GSgnnMultiTaskDataLoader(train_data, config.multi_tasks, val_dataloaders)
    test_dataloader = GSgnnMultiTaskDataLoader(train_data, config.multi_tasks, test_dataloaders)

    return task_evaluators, (train_dataloader, val_dataloader, test_dataloader)

def train(trial, config_args=None, train_data=None):
    config = GSConfig(config_args)
    # check this works
    config._hidden_size = trial.suggest_int("hidden_size", 32, 1024, step=4)

    config._topk_model_to_save = 1
    config._save_model_path += f'/{trial.number}'
    # restore = next(Path(config._save_model_path).glob("epoch-*"), None)
    # if restore is not None:
        # config._restore_model_path = restore.as_posix()

    config.verify_arguments(True)

    tasks = config.multi_tasks
    assert tasks is not None, \
        "The multi_task_learning configure block should not be empty."

    task_evaluators, (train_dataloader, val_dataloader, test_dataloader) = init_evaluation(
        config, train_data
    )

    ### MODEL TRAINING ###                        
    model = init_model(config, train_data)

    if config.save_model_path is not None:
        save_model_path = config.save_model_path
    elif config.save_embed_path is not None:
        # If we need to save embeddings, we need to save the model somewhere.
        save_model_path = os.path.join(config.save_embed_path, "model")
    else:
        save_model_path = None

    # save the generating configuration for this experiment
    Path(save_model_path).mkdir(parents=True, exist_ok=True)
    copy2(config.yaml_paths, Path(save_model_path) / 'config.yaml')

    tracker = gs.create_builtin_task_tracker(config)
    if gs.get_rank() == 0:
        tracker.log_params(config.__dict__)
    
    trainer = GSgnnMultiTaskLearningTrainer(model, topk_model_to_save=config.topk_model_to_save)

    if not config.no_validation:
        evaluator = GSgnnMultiTaskEvaluator(config.eval_frequency,
                                            task_evaluators,
                                            use_early_stop=config.use_early_stop)
        trainer.setup_evaluator(evaluator)
    if config.restore_model_path is not None:
        # TODO is epoch number here so i dont have to track it in state.json for ray tune?
        trainer.restore_model(model_path=config.restore_model_path,
                                model_layer_to_load=config.restore_model_layers)
    trainer.setup_device(device=get_device())

    trainer.setup_task_tracker(tracker)

    trainer.fit(train_loader=train_dataloader,
                val_loader=val_dataloader,
                test_loader=test_dataloader,
                num_epochs=config.num_epochs,
                save_model_path=save_model_path,
                use_mini_batch_infer=config.use_mini_batch_infer,
                save_model_frequency=config.save_model_frequency,
                save_perf_results_path=config.save_perf_results_path,
                freeze_input_layer_epochs=config.freeze_lm_encoder_epochs,
                max_grad_norm=config.max_grad_norm,
                grad_norm_type=config.grad_norm_type,
                optuna_trial=trial)

    # TODO how do i also return the path for warm start?
    # return {'best_path': trainer.get_best_model_path()}
    return trainer.evaluator._get_early_stop_score(trainer.evaluator.best_val_score)

def log_model(config, model):
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
            
    # test out mermaid diagram
    try:
        save_mermaid_diagram(model, Path(config.save_model_path).parent / "model_diagram.md", config.multi_tasks)
        logging.info(f"Saved model diagram to {Path(config.save_model_path).parent / 'model_diagram.md'}")
    except Exception as e:
        logging.warning("Failed to save model diagram: %s", e)

def main(config_args):
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

    # ### tuner config
    # search_space = {
    #     "hidden_size": tune.choice([32, 64, 128, 256, 512, 1024]),
    #     "fanout": tune.choice(["1,1", "2,1", "2,2"])
    # }
    
    # TODO  IS NUM TRAINERS AVAILABLE HERE?
    storage = f'sqlite:///{config.save_model_path}/study.db'.replace('./', '')
    Path(storage).parent.mkdir(parents=True, exist_ok=True)
    print('storage', storage)
    trainable = partial(train, config_args=config_args, train_data=train_data)
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.HyperbandPruner(),
        # storage=storage
        storage="sqlite:///.data/study_name.db"
        # storage="sqlite:///.data/debug/output/gnn/save/study.db" # why this doesn't work is beyond me
        # load_if_exists= True  # need some more work for this to be possible
    )
    study.optimize(
        trainable, 
        n_trials=3, 
        n_jobs=1, 
        catch=(Exception,)
    )
    
    # #### LOAD BEST MODEL
    # # TODO sort furthest model
    restore_path = next(Path(config.save_model_path + f'/{study.best_trial.number}').glob("*epoch*")).as_posix()
    for k, v in study.best_params.items():
        setattr(config, f'_{k}', v)

    model = init_model(config, train_data)
    model.restore_model(restore_path, model_layer_to_load=config.restore_model_layers)
    model = model.to(get_device())

    log_model(config, model)

    ### INFERENCE CODE

    infer = GSgnnMultiTaskLearningInferrer(model)

    task_evaluators, (_, _, test_dataloader) = init_evaluation(
        config, train_data
    )

    if not config.no_validation:
        evaluator = GSgnnMultiTaskEvaluator(eval_frequency=config.eval_frequency, task_evaluators=task_evaluators)
        infer.setup_evaluator(evaluator)

    infer.setup_device(device=get_device())
    
    infer.infer(train_data,
                test_dataloader, 
                None, 
                None, 
                None,
                save_embed_path=config.save_embed_path,
                save_prediction_path=config.save_prediction_path,
                use_mini_batch_infer=config.use_mini_batch_infer,
                node_id_mapping_file=config.node_id_mapping_file,
                edge_id_mapping_file=config.edge_id_mapping_file,
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
