"""This script will do training & inference on best-fit model for multitask to avoid loading the model twice."""
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import logging
import graphstorm as gs
from graphstorm.config import GSConfig, get_argument_parser
from graphstorm.config import BUILTIN_TASK_LINK_PREDICTION
from graphstorm.dataloading import GSgnnMultiTaskDataLoader, GSgnnData
from graphstorm.model.multitask_gnn import GSgnnMultiTaskSharedEncoderModel
from graphstorm.inference import ResonateInferrer
from graphstorm.eval import GSgnnMultiTaskEvaluator
from graphstorm.utils import get_device, rt_profiler, sys_tracker, get_device, get_lm_ntypes
from graphstorm.run.gsgnn_mt.gsgnn_mt import create_task_test_dataloader

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
    # reload in case things change
    config = GSConfig(config_args)
    config.verify_arguments(True)

    # task init #
    tasks = config.multi_tasks
    assert tasks is not None, \
        "The multi_task_learning configure block should not be empty."
    
    test_dataloaders = []
    for task in tasks:
        test_loader = create_task_test_dataloader(task, config, train_data)
        test_dataloaders.append(test_loader)

    test_dataloader = GSgnnMultiTaskDataLoader(train_data, tasks, test_dataloaders)
    #### LOAD BEST MODEL
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
        if config.no_validation:
            raise RuntimeError('no validation is set, cannot do inference')            

        if test_loader is None:
            logging.warning("Task %s does not have test sets.", task.task_id)
            raise RuntimeError('need test data for inference')
        else:
            task_evaluators[task.task_id] = \
                gs.create_evaluator(task)


    # model.restore_model('./.data/hem_rcid/output/hma/save/epoch-15', model_layer_to_load=config.restore_model_layers)
    model.restore_model(config.restore_model_path, model_layer_to_load=config.restore_model_layers)
    model = model.to(get_device())
    
    ### INFERENCE CODE
    infer = ResonateInferrer(model, part_config=config.part_config, cached_labels=False)


    evaluator = GSgnnMultiTaskEvaluator(config.eval_frequency, task_evaluators)
    infer.setup_evaluator(evaluator)


    infer.setup_device(device=get_device())
    infer.infer_safe(train_data,
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
