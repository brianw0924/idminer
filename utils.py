import sys
import torch
from pytorch_metric_learning import losses, miners, samplers, testers, trainers, reducers, distances
from tqdm import tqdm
import numpy as np
import os
import matplotlib.pyplot as plt
import glob
import yaml
import pandas as pd
from cycler import cycler
from sklearn import metrics
from collections import defaultdict

from MyAccuracyCalculator import MyAccuracyCalculator
from mytrainer import MyTrainer
from dataset import TrainDataset, TestDataset
from model import FrameEncoder, VideoEncoder

def load_models(cfg):
        return {
                'trunk': FrameEncoder(cfg["trunk_params"]).to(cfg["device"]),
                'embedder': VideoEncoder(cfg["embedder_params"]).to(cfg["device"])
        }

def load_identification_datasets(cfg):
        '''
        splits_to_eval: [('probes', ['gallery', ...]), ...]
        dataset_dict: {'probes': probes, 'gallery': gallery, ...}
        '''
        splits_to_eval, dataset_dict = [], {}
        train_dataset = TrainDataset(cfg["train"])
        dataset_dict["train"] = train_dataset
        for test_cfg in cfg["test"]:
                probes_dataset = TestDataset(test_cfg["fake"]["feature_dir"], test_cfg["probes_txt"])
                gallery_dataset = TestDataset(test_cfg["fake"]["feature_dir"], test_cfg["gallery_txt"])
                dataset_dict[f"probes_{test_cfg['dataset_name']}"] = probes_dataset
                dataset_dict[f"gallery_{test_cfg['dataset_name']}"] = gallery_dataset
                splits_to_eval.append((f"probes_{test_cfg['dataset_name']}", [f"gallery_{test_cfg['dataset_name']}"]))
        print(dataset_dict)
        return splits_to_eval, dataset_dict

def load_dfd_datasets(cfg):
        dataset_dict = {}
        
        for test_cfg in cfg["test"]:
                dataset_dict[test_cfg["dataset_name"]] = {
                        'v1': [],
                        'v2': [],
                        'y': []
                }
                with open(test_cfg["dfd_test_txt"]) as f:
                        next(f)
                        for line in tqdm(f.readlines()):
                                v1, v2, t1, t2, label = line.strip().split(',')
                                label = int(label)
                                emb1 = torch.nn.functional.normalize(torch.from_numpy(np.load(os.path.join(test_cfg[t1]["embedding_dir"], f"{v1}.npy"))), dim=0)
                                emb2 = torch.nn.functional.normalize(torch.from_numpy(np.load(os.path.join(test_cfg[t2]["embedding_dir"], f"{v2}.npy"))), dim=0)
                                dataset_dict[test_cfg["dataset_name"]]['v1'].append(emb1)
                                dataset_dict[test_cfg["dataset_name"]]['v2'].append(emb2)
                                dataset_dict[test_cfg["dataset_name"]]['y'].append(label)
                        dataset_dict[test_cfg["dataset_name"]]['v1'] = torch.stack(dataset_dict[test_cfg["dataset_name"]]['v1'], dim=0)
                        dataset_dict[test_cfg["dataset_name"]]['v2'] = torch.stack(dataset_dict[test_cfg["dataset_name"]]['v2'], dim=0)
        return dataset_dict

def load_optimizers(cfg, models):
        trunk_optimizer = torch.optim.Adam(
                                models["trunk"].parameters(),
                                lr = cfg["trunk_optimizer_params"]["trunk_lr"],
                                weight_decay = cfg["trunk_optimizer_params"]["trunk_weight_decay"]
                )
        embedder_optimizer = torch.optim.Adam(
                                models["embedder"].parameters(),
                                lr = cfg["embedder_optimizer_params"]["embedder_lr"],
                                weight_decay = cfg["embedder_optimizer_params"]["embedder_weight_decay"]
                )
        optimizers =  {
                        "trunk_optimizer": trunk_optimizer,
                        "embedder_optimizer": embedder_optimizer
        }

        lr_schedulers = {
                "trunk_scheduler_by_epoch": torch.optim.lr_scheduler.CosineAnnealingLR(optimizers["trunk_optimizer"], T_max=50, eta_min=1e-6),
                "embedder_scheduler_by_epoch": torch.optim.lr_scheduler.CosineAnnealingLR(optimizers["trunk_optimizer"], T_max=50, eta_min=1e-6)
        }

        return optimizers, lr_schedulers

def load_distance(distance_params):
        return distances.CosineSimilarity()

def load_reducer(reducer_params):
        return reducers.MeanReducer()

def load_losses(cfg, distance):
        identity_anchored_loss = losses.NTXentLoss(**cfg["identity_anchored_loss_params"])
        artifact_agnostic_loss = losses.NTXentLoss(**cfg["artifact_agnostic_loss_params"])
        loss = {
                "artifact_agnostic_loss": artifact_agnostic_loss,
                "identity_anchored_loss": identity_anchored_loss
        }
        loss_weights = {
                "artifact_agnostic_loss": cfg["artifact_agnostic_loss_weight"],
                "identity_anchored_loss": cfg["identity_anchored_loss_weight"]
        }
        return loss, loss_weights

def load_miner(cfg, distance):
        miner = miners.MultiSimilarityMiner(**cfg["miner_params"])
        return {"tuple_miner": miner}

def load_sampler(train_dataset, cfg):
        return samplers.MPerClassSampler(
                labels = train_dataset.y,
                m = cfg["sampler_params"]["m"],
                length_before_new_iter = len(train_dataset)
        )

def collate_fn(batch):
        batch.sort(key=lambda x: len(x[0]), reverse=True)
        datas, labels, anchors, posnegs = map(list, zip(*batch))
        labels = torch.tensor(labels, dtype=int)
        if anchors[0] == None:
                return datas, labels
        else:
                return datas, labels, torch.stack(anchors, 0), torch.stack(posnegs, 0)

def get_trainer(
                models,
                optimizers,
                batch_size,
                loss_funcs,
                loss_weights,
                mining_funcs,
                dataset,
                sampler,
                dataloader_num_workers,
                lr_schedulers,
                end_of_iteration_hook,
                end_of_epoch_hook
        ):

        return MyTrainer(
                        models = models,
                        optimizers = optimizers,
                        batch_size = batch_size,
                        loss_funcs = loss_funcs,
                        loss_weights = loss_weights,
                        mining_funcs = mining_funcs,
                        dataset = dataset,
                        sampler = sampler,
                        collate_fn = collate_fn,
                        dataloader_num_workers = dataloader_num_workers,
                        lr_schedulers = lr_schedulers,
                        end_of_iteration_hook = end_of_iteration_hook,
                        end_of_epoch_hook = end_of_epoch_hook,
                )


def get_tester(
                end_of_testing_hook,
                dataloader_num_workers,
                batch_size,
                accuracy_calculator
        ):
        return  testers.GlobalEmbeddingSpaceTester(
                        end_of_testing_hook = end_of_testing_hook, # print results after testing 
                        dataloader_num_workers = dataloader_num_workers,
                        batch_size = batch_size,
                        accuracy_calculator = accuracy_calculator
        )


def identification_test(cfg):

        '''
        Load model
        '''
        trunk = torch.nn.Identity().to(cfg["device"])
        trunk.eval()

        '''
        Load dataset
        '''
        print("Load dataset")
        splits_to_eval, dataset_dict = load_identification_datasets(cfg)
        
        '''
        Build tester
        '''
        print("Build tester")
        tester = testers.GlobalEmbeddingSpaceTester(
                        dataloader_num_workers = cfg["num_workers"],
                        batch_size = cfg["batch_size"],
                        accuracy_calculator = MyAccuracyCalculator
        )
        '''
        Identification test
        '''
        print("Test")
        acc = tester.test(
                dataset_dict = dataset_dict,
                epoch=-1,
                trunk_model = trunk,
                splits_to_eval = splits_to_eval
        )
        
        return acc

def dfd_test(cfg):
        '''
        Load dataset
        '''
        dataset_dict = load_dfd_datasets(cfg)
        
        acc = {}
        distance = load_distance(cfg["distance_params"])
        for dataset_name, data in dataset_dict.items():
                tqdm.write(f"testing: {dataset_name}")
                pred = distance.pairwise_distance(data['v1'], data['v2'])
                # pred = [1/x for x in pred]
                fpr, tpr, thresholds = metrics.roc_curve(data['y'], pred, pos_label=1)
                acc[dataset_name] = {'auc': metrics.auc(fpr, tpr)}
                
        return acc

def visualizer_hook(umapper, umap_embeddings, labels, split_name, keyname, *args):
    label_set = np.unique(labels)
    num_classes = len(label_set)
    plt.figure(figsize=(20, 15))
    plt.gca().set_prop_cycle(
        cycler(
            "color", [plt.cm.nipy_spectral(i) for i in np.linspace(0, 0.9, num_classes)]
        )
    )
    for i in range(num_classes):
        idx = labels == label_set[i]
        plt.plot(umap_embeddings[idx, 0], umap_embeddings[idx, 1], ".", markersize=1)
    plt.savefig('visualize.png')

def __extract_embedding(cfg, trunk, embedder):
        AUs = ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r']
        for test_cfg in cfg["test"]:
                if not os.path.exists(test_cfg["fake"]["embedding_dir"]):
                        os.makedirs(test_cfg["fake"]["embedding_dir"])
                if not os.path.exists(test_cfg["real"]["embedding_dir"]):
                        os.makedirs(test_cfg["real"]["embedding_dir"])
                with open(test_cfg["dfd_test_txt"]) as f:
                        next(f)
                        for line in tqdm(f.readlines()):
                                v1, v2, t1, t2, _= line.strip().split(',')
                                if embedder:
                                        embedding_save_path = os.path.join(
                                                test_cfg[t1]["embedding_dir"],
                                                f"{v1}.npy"
                                        )
                                        if not os.path.exists(embedding_save_path):
                                                df = pd.read_csv(os.path.join(test_cfg[t1]["feature_dir"], f"{v1.replace('.mp4', '')}.csv"))
                                                feature = torch.tensor(df[AUs].values).float().to('cuda:0') # (T, num_fau)
                                                f_emb = trunk(feature)
                                                emb =  embedder(f_emb.unsqueeze(0)).squeeze(0).detach().cpu().numpy()
                                                np.save(embedding_save_path, emb)
                                                tqdm.write(f"Save: {embedding_save_path}")

                                        embedding_save_path = os.path.join(
                                                test_cfg[t2]["embedding_dir"],
                                                f"{os.path.basename(v2)}.npy"
                                        )
                                        if not os.path.exists(embedding_save_path):
                                                df = pd.read_csv(os.path.join(test_cfg[t2]["feature_dir"], f"{v2.replace('.mp4', '')}.csv"))
                                                feature = torch.tensor(df[AUs].values).float().to('cuda:0') # (T, num_fau)
                                                f_emb = trunk(feature)
                                                emb =  embedder(f_emb.unsqueeze(0)).squeeze(0).detach().cpu().numpy()
                                                np.save(embedding_save_path, emb)
                                                tqdm.write(f"Save: {embedding_save_path}")
        return

def extract_embedding(cfg):
        models = load_models(cfg)
        trunk = models["trunk"]
        trunk.load_state_dict(torch.load(cfg["trunk_pth"]))
        trunk.eval()
        embedder = models["embedder"]
        embedder.load_state_dict(torch.load(cfg["embedder_pth"]))
        embedder.eval()
        __extract_embedding(cfg, trunk, embedder)

if __name__ == "__main__":
        config_path = sys.argv[1]
        with open(config_path, 'r') as f:
                cfg = defaultdict(lambda: None, yaml.load(f, Loader=yaml.Loader))
        extract_embedding(cfg)