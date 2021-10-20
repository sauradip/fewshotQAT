#!/usr/bin/env python3

import torch
import torch.nn.functional as F
from src.util import batch_intersectionAndUnionGPU
from typing import List
from gtad_lib.util import to_one_hot
from collections import defaultdict
from typing import Tuple
from visdom_logger import VisdomLogger
from gtad_lib.transformer import MultiHeadAttention

def findTAL(logits,target,sub_cls, vid_name):
    n_tasks, shot, num_classes, H, W = logits.size()
    temp_list=[]
    for task in range(n_tasks):
        for shot1 in range(shot):
            # print("clss",cnames[sub_cls[task][shot1]])
            # print("query_vid", vid_name[0])
            fg = logits[task][shot1][0].detach().cpu().numpy()
            bg = logits[task][shot1][1].detach().cpu().numpy()
            
            # thres = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
            thres = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
            # thres = [x / 100.0 for x in range(0, 100, 5)]
            # print(thres)
            for i in thres:
                fg_thres = fg > i
                integer_map = map(int,fg_thres)
                filtered_seq_int = list(integer_map)
                filled_fg = ndimage.binary_fill_holes(filtered_seq_int).astype(int).tolist()
                if 1 in filled_fg:            
                    start_pt = filled_fg.index(1)
                    end_pt = len(filled_fg) - 1 - filled_fg[::-1].index(1)
                    if (end_pt/100) > (start_pt/100) and ((end_pt - start_pt)/100) > 0.1 :
                        reg_vals = np.mean(fg[start_pt:end_pt])
                        conf_vals = np.amax(fg[start_pt:end_pt])
                        # print("conf",conf_vals)
                        temp_list.append([start_pt/100,end_pt/100,conf_vals,conf_vals])
                # else:
                    
            # print("temp_list", temp_list)
            # print(target[task][shot1].detach().cpu().numpy().tolist())
    new_props = np.stack(temp_list)
    video_name = vid_name[0]
    col_name = ["xmin", "xmax", "clr_score", "reg_socre"]
    new_df = pd.DataFrame(new_props, columns=col_name)
    

    

    new_df.to_csv(opt["output"]+"/results1/" + video_name + ".csv", index=False)

class Classifier(object):
    def __init__(self):
        self.num_classes = 2
        self.temperature = 20.0
        self.adapt_iter = 50
        # self.weights = [1.0, 'auto', 'auto']
        self.weights = [1.5, 0.2, 0.8]
        self.lr = 0.025
        self.FB_param_update = [10]
        self.visdom_freq = 5
        self.FB_param_type = "soft"
        self.FB_param_noise = 0
        self.self_attn = MultiHeadAttention(1, 9216, 9216, 9216, dropout=0.5)
        self.use_att = True
        # print(self.self_attn)
        # temp 20.0
        # adap 50
        # weights [1.0, 'auto', 'auto']
        # lr 0.025
        # FB [10]
        # FB_param_type soft
        # FB_param_noise 0


    def init_prototypes(self, features_s: torch.tensor, features_s_grad: torch.tensor, features_q: torch.tensor, features_q_grad: torch.tensor,
                        gt_s: torch.tensor, gt_q: torch.tensor, subcls: List[int], callback ) -> None:
        """
        inputs:
            features_s : shape [n_task, shot, c, h, w] [1 ,2, 9560, 100, 100]
            features_q : shape [n_task, 1, c, h, w] [1 , 9560, 100 , 100]
            gt_s : shape [n_task, shot, H, W]
            gt_q : shape [n_task, 1, H, W]

        returns :
            prototypes : shape [n_task, c]
            bias : shape [n_task]
        """
        

        ### set Transformer gradient to True ##

        for param in self.self_attn.parameters():
            param.requires_grad = True

        # DownSample support masks
        # n_task, shot, c, h, w = features_s.size()
        
        n_task, shot, c, h, w = features_s.size()
        # print("features_s", features_s.size())
        # f_s_att = features_s.view(n_task,shot,c*h*w)
        # f_q_att = features_q.view(n_task,shot,c*h*w)
        # set_s_q = torch.cat((f_s_att,f_q_att), 2)

        # print("feat_s", features_s_grad.requires_grad)
        ds_gt_s = F.interpolate(gt_s.float(), size=features_s.shape[-2:], mode='nearest')
        ds_gt_s = ds_gt_s.long().unsqueeze(2)  # [n_task, shot, 1, h, w]

        ds_gt_q = F.interpolate(gt_q.float(), size=features_q.shape[-2:], mode='nearest')
        ds_gt_q = ds_gt_q.long().unsqueeze(2)  # [n_task, shot, 1, h, w]
        
        # attn_weights = self.self_attn(features_s,features_s,features_s)
        # attn_weights = self.self_attn(features_q,features_s,features_s)
        # Computing prototypes
        fg_mask = (ds_gt_s == 1)
        fg_mask_q = (ds_gt_q == 1)
        # print("fg_mask", fg_mask)
        if self.use_att : 
            fg_prototype = (features_s * fg_mask).sum(dim=(3, 4))
            fg_prototype /= (fg_mask.sum(dim=(3, 4)) + 1e-10)  # [n_task, c] 
            proto_ = fg_prototype
            attn_prototype = self.self_attn(proto_,proto_, proto_).cuda().detach()
            self.prototype = torch.mean(attn_prototype,1).squeeze(1)
        else : 
            fg_prototype = (features_s * fg_mask).sum(dim=(1, 3, 4))
            fg_prototype /= (fg_mask.sum(dim=(1, 3, 4)) + 1e-10)  # [n_task, c] 
            self.prototype = fg_prototype
        # print("fg_proto", fg_prototype.size())

        # fg_prototype_q = (features_q * fg_mask_q).sum(dim=(1, 3, 4))
        # fg_prototype_q /= (fg_mask_q.sum(dim=(1, 3, 4)) + 1e-10)  # [n_task, c] 
        # set_ = torch.cat((fg_prototype,fg_prototype_q), 1)
        # self.prototype = fg_prototype
        # print("before attn",fg_prototype.requires_grad)
        
        # self.prototype = attn_prototype.squeeze(1)
        
        # self.prototype_attn = self.self_attn(proto_,proto_,proto_).squeeze(1).cuda()
        # print("after attn",self.self_attn.parameters())
        # self.proto_attn =  self.self_attn(proto_,proto_,proto_).cuda().detach()
        # self.prototype = self.self_attn(proto_,proto_,proto_).squeeze(1).cuda().detach()

        # proto = F.normalize(self.prototype, dim=-1) # normalize for cosine distance

        # query = query.view(num_batch, -1, emb_dim) # (Nbatch,  Nq*Nw, d)
        # logits = torch.bmm(query, proto.permute([0,2,1])) / self.args.temperature
        # logits = logits.view(-1, num_proto)
        
        # self.prototype.requires_grad = False
        # self.prototype = self.prototype(requires_grad=True)
        # print("prototype", self.prototype.size())
        logits_q = self.get_logits(features_q)  # [n_tasks, shot, h, w]
        self.bias = logits_q.mean(dim=(1, 2, 3))

        assert self.prototype.size() == (n_task, c), self.prototype.size()
        assert torch.isnan(self.prototype).sum() == 0, self.prototype

        if callback is not None:
            self.update_callback(callback, 0, features_s, features_q, subcls, gt_s, gt_q)

    def get_logits(self, features: torch.tensor) -> torch.tensor:

        """
        Computes the cosine similarity between self.prototype and given features
        inputs:
            features : shape [n_tasks, shot, c, h, w]

        returns :
            logits : shape [n_tasks, shot, h, w]
        """

        # Put prototypes and features in the right shape for multiplication

        features = features.permute((0, 1, 3, 4, 2))  # [n_task, shot, h, w, c]
        prototype = self.prototype.unsqueeze(1).unsqueeze(2)  # [n_tasks, 1, 1, c]

        # Compute cosine similarity
        cossim = features.matmul(prototype.unsqueeze(4)).squeeze(4)  # [n_task, shot, h, w]
        cossim /= ((prototype.unsqueeze(3).norm(dim=4) * \
                    features.norm(dim=4)) + 1e-10)  # [n_tasks, shot, h, w]

        return self.temperature * cossim

    def get_probas(self, logits: torch.tensor) -> torch.tensor:
        """
        inputs:
            logits : shape [n_tasks, shot, h, w]

        returns :
            probas : shape [n_tasks, shot, num_classes, h, w]
        """
        logits_fg = logits - self.bias.unsqueeze(1).unsqueeze(2).unsqueeze(3)  # [n_tasks, shot, h, w]
        probas_fg = torch.sigmoid(logits_fg).unsqueeze(2)
        probas_bg = 1 - probas_fg
        probas = torch.cat([probas_bg, probas_fg], dim=2)
        return probas

    def compute_FB_param(self, features_q: torch.tensor, gt_q: torch.tensor) -> torch.tensor:
        """
        inputs:
            features_q : shape [n_tasks, shot, c, h, w]
            gt_q : shape [n_tasks, shot, h, w]

        updates :
             self.FB_param : shape [n_tasks, num_classes]
        """
        ds_gt_q = F.interpolate(gt_q.float(), size=features_q.size()[-2:], mode='nearest').long()
        valid_pixels = (ds_gt_q != 0).unsqueeze(2)  # [n_tasks, shot, num_classes, h, w]
        assert (valid_pixels.sum(dim=(1, 2, 3, 4)) == 0).sum() == 0, valid_pixels.sum(dim=(1, 2, 3, 4))

        one_hot_gt_q = to_one_hot(ds_gt_q, self.num_classes)  # [n_tasks, shot, num_classes, h, w]

        oracle_FB_param = (valid_pixels * one_hot_gt_q).sum(dim=(1, 3, 4)) / valid_pixels.sum(dim=(1, 3, 4))

        if self.FB_param_type == 'oracle':
            self.FB_param = oracle_FB_param
            # Used to assess influence of delta perturbation
            if self.FB_param_noise != 0:
                perturbed_FB_param = oracle_FB_param
                perturbed_FB_param[:, 1] += self.FB_param_noise * perturbed_FB_param[:, 1]
                perturbed_FB_param = torch.clamp(perturbed_FB_param, 0, 1)
                perturbed_FB_param[:, 0] = 1.0 - perturbed_FB_param[:, 1]
                self.FB_param = perturbed_FB_param

        else:
            logits_q = self.get_logits(features_q)
            probas = self.get_probas(logits_q).detach()
            self.FB_param = (valid_pixels * probas).sum(dim=(1, 3, 4))
            self.FB_param /= valid_pixels.sum(dim=(1, 3, 4))

        # Compute the relative error
        deltas = self.FB_param[:, 1] / oracle_FB_param[:, 1] - 1
        return deltas

    def get_entropies(self,
                      valid_pixels: torch.tensor,
                      probas: torch.tensor,
                      reduction='sum') -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        """
        inputs:
            probas : shape [n_tasks, shot, num_class, h, w]
            valid_pixels: shape [n_tasks, shot, h, w]

        returns:
            d_kl : FB proportion kl [n_tasks,]
            cond_entropy : Entropy of predictions [n_tasks,]
            marginal : Current marginal distribution over labels [n_tasks, num_classes]
        """
        n_tasks, shot, num_classes, h, w = probas.size()
        # print(valid_pixels)
        assert (valid_pixels.sum(dim=(1, 2, 3)) == 0).sum() == 0, \
               (valid_pixels.sum(dim=(1, 2, 3)) == 0).sum()  # Make sure all tasks have a least 1 valid pixel

        cond_entropy = - ((valid_pixels.unsqueeze(2) * (probas * torch.log(probas + 1e-10))).sum(2))
        cond_entropy = cond_entropy.sum(dim=(1, 2, 3))
        cond_entropy /= valid_pixels.sum(dim=(1, 2, 3))

        marginal = (valid_pixels.unsqueeze(2) * probas).sum(dim=(1, 3, 4))
        marginal /= valid_pixels.sum(dim=(1, 2, 3)).unsqueeze(1)

        d_kl = (marginal * torch.log(marginal / (self.FB_param + 1e-10))).sum(1)

        if reduction == 'sum':
            cond_entropy = cond_entropy.sum(0)
            d_kl = d_kl.sum(0)
            assert not torch.isnan(cond_entropy), cond_entropy
            assert not torch.isnan(d_kl), d_kl
        elif reduction == 'mean':
            cond_entropy = cond_entropy.mean(0)
            d_kl = d_kl.mean(0)
        return d_kl, cond_entropy, marginal

    def get_ce(self,
               probas: torch.tensor,
               valid_pixels: torch.tensor,
               one_hot_gt: torch.tensor,
               reduction: str = 'sum') -> torch.tensor:
        """
        inputs:
            probas : shape [n_tasks, shot, c, h, w]
            one_hot_gt: shape [n_tasks, shot, num_classes, h, w]
            valid_pixels : shape [n_tasks, shot, h, w]

        updates :
             ce : Cross-Entropy between one_hot_gt and probas, shape [n_tasks,]
        """
        ce = - ((valid_pixels.unsqueeze(2) * (one_hot_gt * torch.log(probas + 1e-10))).sum(2))  # [n_tasks, shot, h, w]
        ce = ce.sum(dim=(1, 2, 3))  # [n_tasks]
        ce /= valid_pixels.sum(dim=(1, 2, 3))
        if reduction == 'sum':
            ce = ce.sum(0)
        elif reduction == 'mean':
            ce = ce.mean(0)
        return ce
    
    def infer(self, q_features, target):
        self.prototype.requires_grad = False
        self.bias.requires_grad = False

        logits_q = self.get_logits(q_features)
        proba_q = self.get_probas(logits_q)
        findTAL(proba_q,target)


    def RePRI(self,
              features_s: torch.tensor,
              features_q: torch.tensor,
              gt_s: torch.tensor,
              gt_q: torch.tensor,
              subcls: List,
              n_shots: torch.tensor,
              callback: VisdomLogger) -> torch.tensor:
        """
        Performs RePRI inference

        inputs:
            features_s : shape [n_tasks, shot, c, h, w]
            features_q : shape [n_tasks, shot, c, h, w]
            gt_s : shape [n_tasks, shot, h, w]
            gt_q : shape [n_tasks, shot, h, w]
            subcls : List of classes present in each task
            n_shots : # of support shots for each task, shape [n_tasks,]

        updates :
            prototypes : torch.Tensor of shape [n_tasks, num_class, c]

        returns :
            deltas : Relative error on FB estimation right after first update, for each task,
                     shape [n_tasks,]
        """
        deltas = torch.zeros_like(n_shots)
        l1, l2, l3 = self.weights
        if l2 == 'auto':
            l2 = 1 / n_shots
        else:
            l2 = l2 * torch.ones_like(n_shots)
        if l3 == 'auto':
            l3 = 1 / n_shots
        else:
            l3 = l3 * torch.ones_like(n_shots)
        if self.use_att :
            param = list(self.self_attn.parameters())
            self.prototype.requires_grad_()
            self.bias.requires_grad_()
            param.append(self.prototype)
            param.append(self.bias)
        else:
            param=[]
            self.prototype.requires_grad_()
            self.bias.requires_grad_()
            param.append(self.prototype)
            param.append(self.bias)
        # optimizer = torch.optim.SGD([self.prototype, self.bias,], lr=self.lr)
        optimizer = torch.optim.SGD(param, lr=self.lr)
        ds_gt_q = F.interpolate(gt_q.float(), size=features_s.size()[-2:], mode='nearest').long()
        ds_gt_s = F.interpolate(gt_s.float(), size=features_s.size()[-2:], mode='nearest').long()

        valid_pixels_q = (ds_gt_q != 0).float()  # [n_tasks, shot, h, w]
        valid_pixels_s = (ds_gt_s != 0).float()  # [n_tasks, shot, h, w]

        one_hot_gt_s = to_one_hot(ds_gt_s, self.num_classes)  # [n_tasks, shot, num_classes, h, w]

        for iteration in range(1, self.adapt_iter):

            logits_s = self.get_logits(features_s)  # [n_tasks, shot, num_class, h, w]
            logits_q = self.get_logits(features_q)  # [n_tasks, 1, num_class, h, w]
            proba_q = self.get_probas(logits_q)
            proba_s = self.get_probas(logits_s)

            d_kl, cond_entropy, marginal = self.get_entropies(valid_pixels_q,
                                                              proba_q,
                                                              reduction='none')
            ce = self.get_ce(proba_s, valid_pixels_s, one_hot_gt_s, reduction='none')
            # loss = l1 * ce + l2 * d_kl + l3 * cond_entropy

            loss = l1 * ce + l3 * cond_entropy

            optimizer.zero_grad()
            loss.sum(0).backward()
            optimizer.step()

            # Update FB_param
            if (iteration + 1) in self.FB_param_update  \
                    and ('oracle' not in self.FB_param_type) and (l2.sum().item() != 0):
                deltas = self.compute_FB_param(features_q, gt_q).cpu()
                l2 += 1

            if callback is not None and (iteration + 1) % self.visdom_freq == 0:
                self.update_callback(callback, iteration, features_s, features_q, subcls, gt_s, gt_q)
        return deltas

    def get_mIoU(self,
                 probas: torch.tensor,
                 gt: torch.tensor,
                 subcls: torch.tensor,
                 reduction: str = 'mean') -> torch.tensor:
        """
        Computes the mIoU over the current batch of tasks being processed

        inputs:
            probas : shape [n_tasks, shot, num_class, h, w]
            gt : shape [n_tasks, shot, h, w]
            subcls : List of classes present in each task


        returns :
            class_IoU : Classwise IoU (or mean of it), shape
        """
        intersection, union, _ = batch_intersectionAndUnionGPU(probas, gt, self.num_classes)  # [num_tasks, shot, num_class]
        inter_count = defaultdict(int)
        union_count = defaultdict(int)

        for i, classes_ in enumerate(subcls):
            inter_count[0] += intersection[i, 0, 0]
            union_count[0] += union[i, 0, 0]
            for j, class_ in enumerate(classes_):
                inter_count[class_] += intersection[i, 0, j + 1]  # Do not count background
                union_count[class_] += union[i, 0, j + 1]
        class_IoU = torch.tensor([inter_count[subcls] / union_count[subcls] for subcls in inter_count if subcls != 0])
        if reduction == 'mean':
            return class_IoU.mean()
        elif reduction == 'none':
            return class_IoU

    def update_callback(self, callback, iteration: int, features_s: torch.tensor,
                        features_q: torch.tensor, subcls: List[int],
                        gt_s: torch.tensor, gt_q: torch.tensor) -> None:
        """
        Updates the visdom callback in case live visualization of metrics is desired

        inputs:
            iteration: Current inference iteration
            features_s : shape [n_tasks, shot, c, h, w]
            features_q : shape [n_tasks, shot, c, h, w]
            gt_s : shape [n_tasks, shot, h, w]
            gt_q : shape [n_tasks, shot, h, w]
            subcls : List of classes present in each task


        returns :
            callback : Visdom logger
        """
        logits_q = self.get_logits(features_q)  # [n_tasks, shot, num_class, h, w]
        logits_s = self.get_logits(features_s)  # [n_tasks, shot, num_class, h, w]
        proba_q = self.get_probas(logits_q).detach()  # [n_tasks, shot, num_class, h, w]
        proba_s = self.get_probas(logits_s).detach()  # [n_tasks, shot, num_class, h, w]

        f_resolution = features_s.size()[-2:]
        ds_gt_q = F.interpolate(gt_q.float(), size=f_resolution, mode='nearest').long()
        ds_gt_s = F.interpolate(gt_s.float(), size=f_resolution, mode='nearest').long()

        valid_pixels_q = (ds_gt_q != 0).float()  # [n_tasks, shot, h, w]
        valid_pixels_s = (ds_gt_s != 0).float()  # [n_tasks, shot, h, w]

        one_hot_gt_q = to_one_hot(ds_gt_q, self.num_classes)  # [n_tasks, shot, num_classes, h, w]
        oracle_FB_param = (valid_pixels_q * one_hot_gt_q).sum(dim=(1, 3, 4))
        oracle_FB_param /= (valid_pixels_q).sum(dim=(1, 3, 4))

        one_hot_gt_s = to_one_hot(ds_gt_s, self.num_classes)  # [n_tasks, shot, num_classes, h, w]
        ce_s = self.get_ce(proba_s, valid_pixels_s, one_hot_gt_s)
        ce_q = self.get_ce(proba_q, valid_pixels_q, one_hot_gt_q)

        mIoU_q = self.get_mIoU(proba_q, gt_q, subcls)

        callback.scalar('mIoU_q', iteration, mIoU_q, title='mIoU')
        if iteration > 0:
            d_kl, cond_entropy, marginal = self.get_entropies(valid_pixels_q,
                                                              proba_q,
                                                              reduction='mean')
            marginal2oracle = (oracle_FB_param * torch.log(oracle_FB_param / marginal + 1e-10)).sum(1).mean()
            FB_param2oracle = (oracle_FB_param * torch.log(oracle_FB_param / self.FB_param + 1e-10)).sum(1).mean()
            callback.scalars(['Cond', 'marginal2oracle', 'FB_param2oracle'], iteration,
                             [cond_entropy, marginal2oracle, FB_param2oracle], title='Entropy')
        callback.scalars(['ce_s', 'ce_q'], iteration, [ce_s, ce_q], title='CE')