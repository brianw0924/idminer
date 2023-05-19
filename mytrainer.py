import torch
from pytorch_metric_learning.trainers import BaseTrainer
from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu

class MyTrainer(BaseTrainer):
    def calculate_loss(self, curr_batch):
        datas, labels, anchors, posnegs = curr_batch
        '''
        Artifact Agnostic
        '''
        self.losses["artifact_agnostic_loss"] = \
            self.artifact_agnostic_loss(anchors, posnegs, labels)

        '''
        Identity Anchored
        '''
        self.losses["identity_anchored_loss"] = \
            self.identity_anchored_loss(datas, labels)
        # trunk_output = self.get_trunk_output(data)
        # embeddings = self.get_final_embeddings(trunk_output)
        
    def artifact_agnostic_loss(self, anchors, posnegs, labels):
        embeddings = (
            self.get_trunk_output(anchors),
            self.get_trunk_output(posnegs)
        )

        indices_tuple = self.maybe_mine_frame_embeddings(embeddings, labels)
        return self.maybe_get_artifact_agnostic_loss(
            embeddings, labels, indices_tuple
        )

    def maybe_mine_frame_embeddings(self, embeddings, labels):
        # for both get_all_triplets_indices and mining_funcs
        # we need to clone labels and pass them as ref_labels
        # to ensure triplets are generated between anchors and posnegs
        if "tuple_miner" in self.mining_funcs:
            (anchors_embeddings, posnegs_embeddings) = embeddings
            return self.mining_funcs["tuple_miner"](
                anchors_embeddings, labels, posnegs_embeddings, labels.clone()
            )
        else:
            labels = labels.to(embeddings[0].device)
            return lmu.get_all_triplets_indices(labels, labels.clone())

    def maybe_get_artifact_agnostic_loss(self, embeddings, labels, indices_tuple):
        if self.loss_weights.get("artifact_agnostic_loss", 0) > 0:
            current_batch_size = embeddings[0].shape[0]
            indices_tuple = c_f.shift_indices_tuple(indices_tuple, current_batch_size)
            all_labels = torch.cat([labels, labels], dim=0)
            all_embeddings = torch.cat(embeddings, dim=0)
            return self.loss_funcs["artifact_agnostic_loss"](
                all_embeddings, all_labels, indices_tuple
            )
        return 0

    def identity_anchored_loss(self, datas, labels):
        trunk_output = self.models["trunk"](datas)
        embeddings = self.models["embedder"](trunk_output)
        indices_tuple = self.maybe_mine_video_embeddings(embeddings, labels)
        return self.maybe_get_identity_anchored_loss(
            embeddings, labels, indices_tuple
        )
    
    # def compute_embeddings(self, datas):
    #     videos = []
    #     for data in datas:
    #         videos.append(self.get_trunk_output(data))
    #     videos = pack_sequence(sequences=videos)
    #     videos.device = videos.data.device
    #     embeddings = self.get_final_embeddings(videos)
    #     return embeddings

    def maybe_mine_video_embeddings(self, data, labels):
        return self.mining_funcs["tuple_miner"](data, labels)

    def maybe_get_identity_anchored_loss(self, embeddings, labels, indices_tuple):
        if self.loss_weights.get("identity_anchored_loss", 0) > 0:
            return self.loss_funcs["identity_anchored_loss"](embeddings, labels, indices_tuple)
        return 0

    def get_batch(self):
        self.dataloader_iter, curr_batch = c_f.try_next_on_generator(
            self.dataloader_iter, self.dataloader
        )
        return curr_batch

    def modify_schema(self):
        self.schema["loss_funcs"].keys += ["artifact_agnostic_loss"]
        self.schema["loss_funcs"].keys += ["identity_anchored_loss"]
