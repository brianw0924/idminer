import os
import sys
import yaml
from collections import defaultdict
import pytorch_metric_learning.utils.logging_presets as logging_presets
from MyAccuracyCalculator import MyAccuracyCalculator
from datetime import datetime
from utils import *

def main(cfg):

        '''
        Distance
        '''        
        distance = load_distance(cfg["distance_params"])
        
        '''
        Dataset
        '''
        splits_to_eval, dataset_dict = load_identification_datasets(cfg)
        sampler = load_sampler(dataset_dict['train'], cfg)

        '''
        Models
        '''
        models = load_models(cfg)

        '''
        Losses
        '''
        loss_funcs, loss_weights = load_losses(cfg, distance)

        '''
        Optimizers & Schedulers
        '''
        optimizers, lr_schedulers = load_optimizers(cfg, models)
        

        '''
        Miners
        '''
        mining_funcs = load_miner(cfg, distance)



        result_dir = os.path.join(
                cfg["result_dir"],
                str(datetime.now()).replace(' ', '#')
        )
        if not os.path.exists(result_dir):
                os.makedirs(result_dir)
        log_folder = os.path.join(result_dir, "logs")
        model_folder = os.path.join(result_dir, "model")
        os.makedirs(log_folder)
        os.makedirs(model_folder)
        with open(os.path.join(result_dir, 'cfg.yaml'), 'w') as outfile:
                yaml.dump(cfg, outfile, default_flow_style=False)
        
        record_keeper, _, _ = logging_presets.get_record_keeper(
                log_folder
                # tensorboard_folder
        )

        hooks = logging_presets.get_hook_container(
                record_keeper,
                primary_metric = cfg["primary_metric"],
                validation_split_name = cfg["validation_split_name"]
        )

        '''
        Tester
        '''
        tester = get_tester(
                end_of_testing_hook = hooks.end_of_testing_hook, # print results after testing 
                dataloader_num_workers = cfg["num_workers"],
                batch_size = cfg["batch_size"],
                accuracy_calculator = MyAccuracyCalculator(
                        include = cfg["accuracy"],
                        k = cfg["k"]
        ))

        end_of_epoch_hook = hooks.end_of_epoch_hook(
                tester = tester,
                dataset_dict = dataset_dict,
                model_folder = model_folder,
                test_interval = cfg["test_interval"],
                patience = cfg["patience"],
                splits_to_eval = splits_to_eval,
                test_collate_fn = collate_fn
        ) # will access tester.test each epoch

        '''
        Train
        '''
        trainer = get_trainer(
                        models = models,
                        optimizers = optimizers,
                        batch_size = cfg["batch_size"],
                        loss_funcs = loss_funcs,
                        loss_weights = loss_weights,
                        mining_funcs = mining_funcs,
                        dataset = dataset_dict["train"],
                        sampler = sampler,
                        dataloader_num_workers = cfg["num_workers"],
                        lr_schedulers = lr_schedulers,
                        end_of_iteration_hook = hooks.end_of_iteration_hook,
                        end_of_epoch_hook = end_of_epoch_hook,
        )

        trainer.train(num_epochs = cfg["num_epochs"])

if __name__ == "__main__":
        config_path = sys.argv[1]
        with open(config_path, 'r') as f:
                cfg = defaultdict(lambda: None, yaml.load(f, Loader=yaml.Loader))
        main(cfg)