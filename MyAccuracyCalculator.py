import torch
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
class MyAccuracyCalculator(AccuracyCalculator):
        # method name must start with "calculate_"

        def calculate_rank_1(self, knn_labels, query_labels, **kwargs):
                # knn_labels is a 2d tensor, where knn_labels[i, j] is the label of the jth nearest neighbor of the ith query embedding
                # query_labels is a 1d tensor where query_labels[i] is the label of the ith query embedding
                assert len(knn_labels[0]) >= 1
                cmc_curve = torch.zeros(1)
                for i, query_label in enumerate(query_labels):
                        for rank, knn_label in enumerate(knn_labels[i][:1]):
                                if query_label == knn_label:
                                        # print(query_label)
                                        cmc_curve[rank:] +=1
                                        break
                cmc_curve /= len(query_labels)
                return cmc_curve[0]

        def calculate_rank_5(self, knn_labels, query_labels, **kwargs):
                # knn_labels is a 2d tensor, where knn_labels[i, j] is the label of the jth nearest neighbor of the ith query embedding
                # query_labels is a 1d tensor where query_labels[i] is the label of the ith query embedding
                assert len(knn_labels[0]) >= 5
                cmc_curve = torch.zeros(5)
                for i, query_label in enumerate(query_labels):
                        for rank, knn_label in enumerate(knn_labels[i][:5]):
                                if query_label == knn_label:
                                        cmc_curve[rank:] +=1
                                        break
                cmc_curve /= len(query_labels)
                return cmc_curve[4]

        def calculate_rank_10(self, knn_labels, query_labels, **kwargs):
                assert len(knn_labels[0]) >= 10
                # knn_labels is a 2d tensor, where knn_labels[i, j] is the label of the jth nearest neighbor of the ith query embedding
                # query_labels is a 1d tensor where query_labels[i] is the label of the ith query embedding
                cmc_curve = torch.zeros(10)
                for i, query_label in enumerate(query_labels):
                        for rank, knn_label in enumerate(knn_labels[i][:10]):
                                if query_label == knn_label:
                                        # print(query_label)
                                        cmc_curve[rank:] +=1
                                        break
                cmc_curve /= len(query_labels)
                return cmc_curve[9]

        def calculate_mAP(self, knn_labels, query_labels, **kwargs):
                # knn_labels is a 2d tensor, where knn_labels[i, j] is the label of the jth nearest neighbor of the ith query embedding
                # query_labels is a 1d tensor where query_labels[i] is the label of the ith query embedding
                ap = torch.zeros(len(query_labels))
                for i, query_label in enumerate(query_labels):
                        gt_count = 0
                        for rank, knn_label in enumerate(knn_labels[i]):
                                if query_label == knn_label:
                                        # print(query_label)
                                        gt_count += 1
                                        ap[i] += gt_count / (rank + 1)
                        if gt_count != 0:
                                ap[i] /= gt_count
                return torch.mean(ap)
        
        def requires_knn(self):
                return super().requires_knn() + ["rank_1", "rank_5", "rank_10", "mAP"]
         
