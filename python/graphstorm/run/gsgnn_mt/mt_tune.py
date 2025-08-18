"""This script will do training & inference on best-fit model for multitask to avoid loading the model twice."""
import torch as th
import logging
import os
import math
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
from optuna.integration import TorchDistributedTrial
from functools import partial
import json
from torch.distributed import broadcast
from graphstorm.utils import barrier

def suggest_parameters(trial, config):
    search_space = json.loads((Path(config.yaml_paths).parent / 'tune_space.json').read_text())

    params = {
        x[1]['name']: getattr(trial, f'suggest_{x[0]}')(**x[1]) for x in search_space
    }

    return params

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

def update_config(config, params):
    # set number of graph convolutions by fanout param if it varies
    # for num_layers construct the fanout
    fanout = params['fanout']
    params['fanout'] = ','.join([str(fanout)] * params['num_layers'])
    # params['eval_fanout'] = params['fanout']
    params['eval_fanout'] = ','.join(['10'] * params['num_layers'])

    # insert hyperparams into config object
    for k, v in params.items():
        setattr(config, f'_{k}', v)

    # set dim_key and dim_value
    for node_type in ['rcid', 'ip', 'hem']:
        try:
            # look for hma parameters for the query shape and set the key/value dim
            dim_model = getattr(config, f'_hma_{node_type}_dim_model')
            n_heads = getattr(config, f'_hma_{node_type}_attention_heads')
            dim_key = math.floor(dim_model / n_heads)
            setattr(config, f'_hma_{node_type}_dim_key', dim_key)
            setattr(config, f'_hma_{node_type}_dim_value', dim_key)
        except:
            pass

    # set batch size for each task. hem needs half the batch size
    if 'batch_size' in params:
        for ii, task in enumerate(config.multi_tasks):
            if 'hem' in task.task_id:
                bs = int(params['batch_size'] / 2)
            else:
                bs = params['batch_size']
            config.multi_tasks[ii].task_config._batch_size = bs
            config.multi_tasks[ii].task_config._eval_batch_size = int(bs / 8)
    return config

def train(single_trial, config_args=None, train_data=None):
    barrier()
    trial = TorchDistributedTrial(single_trial)
    # broadcast the trial number
    number = th.tensor([int(trial.number)], device=get_device())
    broadcast(number, src=0)

    config = GSConfig(config_args)

    params = suggest_parameters(trial, config)
    # if gs.get_rank() == 0:
    #     params = suggest_parameters(trial, config)
    #     metadata = {
    #         'params': params,
    #         'trial_number': trial.number
    #     }
    #     # params = th.tensor([params[k] for k in keys], device=gs.utils.get_device())
    #     Path('metadata.json').write_text(json.dumps(metadata))

    # barrier()
    # metadata = json.loads(Path('metadata.json').read_text())
    # params = metadata['params']
    # trial_number = metadata['trial_number']
    
    config = update_config(config, params)

    config._topk_model_to_save = 3
    config._save_model_path += f'/{int(number.item())}'
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


    tracker = gs.create_builtin_task_tracker(config)
    if gs.get_rank() == 0:
        tracker.log_params(config.__dict__)
        # save the generating configuration for this experiment
        Path(save_model_path).mkdir(parents=True, exist_ok=True)
        copy2(config.yaml_paths, Path(save_model_path) / 'config.yaml')
        
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

    try:
        trainer.fit(train_loader=train_dataloader,
                val_loader=val_dataloader,
                # test_loader=test_dataloader,
                num_epochs=config.num_epochs,
                save_model_path=save_model_path,
                use_mini_batch_infer=config.use_mini_batch_infer,
                save_model_frequency=config.save_model_frequency,
                save_perf_results_path=config.save_perf_results_path,
                freeze_input_layer_epochs=config.freeze_lm_encoder_epochs,
                max_grad_norm=config.max_grad_norm,
                grad_norm_type=config.grad_norm_type,
                is_optuna_run=True,
                optuna_trial=trial)
    except th.cuda.OutOfMemoryError:
        # if gs.get_rank() == 0:
        trial.set_user_attr("OOM", True)
            # trial.state = optuna.trial.TrialState.COMPLETE
            # return float("inf")
        th.cuda.empty_cache()
        raise
    except optuna.exceptions.TrialPruned:
        # if gs.get_rank() == 0:
        trial.set_user_attr("Pruned", True)
            # trial.state = optuna.trial.TrialState.PRUNED
            # return None
        raise
    except Exception as e:
        # if gs.get_rank() == 0:
        trial.set_user_attr("Exception", str(e))
            # trial.state = optuna.trial.TrialState.FAIL
            # return None
        raise
    finally:
        barrier()
        
    if gs.get_rank() == 0:
        # trial.state = optuna.trial.TrialState.COMPLETE
        return trainer.evaluator._get_early_stop_score(trainer.evaluator.best_val_score)
    return

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

def main(config_args) -> None:
    ## main from training script graphstorm.
    config = GSConfig(config_args)
    config.verify_arguments(True)

    tune_config = (Path(config.yaml_paths).parent / 'tune_space.json')
    Path(config.save_model_path).parent.mkdir(parents=True, exist_ok=True)
    if tune_config.exists():
        copy2(tune_config, Path(config.save_model_path).parent / 'tune_space.json')
    else:
        raise FileNotFoundError(f'tuner requires `tune_space.json` in same dir as the yaml config found at {config.yaml_paths}')

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
    trainable = partial(train, config_args=config_args, train_data=train_data)
    
    if gs.get_rank() == 0:
        study = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.HyperbandPruner(),
            study_name=config.study_name,
            storage="sqlite:///.data/optuna.db",
            load_if_exists= True  # need some more work for this to be possible
        )

        study.optimize(
            trainable,
            n_trials=config.study_n_trials,
            n_jobs=1,
            catch=(th.cuda.OutOfMemoryError,)
        )
    else:
        for _ in range(config.study_n_trials):
            try:
                trainable(None)
            except optuna.exceptions.TrialPruned:
                pass
            except th.cuda.OutOfMemoryError:
                pass

    # #### LOAD BEST MODEL
    # # TODO sort furthest model
    best_metadata_path = Path(config.save_model_path) / 'best_metadata.json'
    if gs.get_rank() == 0:
        metadata = {
            'trial_number': study.best_trial.number,
            'params': study.best_params
        }
        best_metadata_path.write_text(json.dumps((metadata)))

    barrier()
    metadata = json.loads(best_metadata_path.read_text())
    restore_path = next(Path(config.save_model_path + f'/{metadata["trial_number"]}').glob("*epoch*")).as_posix()

    config = update_config(config, metadata['params'])

    model = init_model(config, train_data)
    model.restore_model(restore_path, model_layer_to_load=config.restore_model_layers)
    model = model.to(get_device())

    if gs.get_rank() == 0:
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
