import os
import math
import numpy as np
import pandas as pd
import torch.nn.parallel
from tqdm import tqdm
from gtad_lib import opts
from gtad_lib.models import GTAD
from gtad_lib.dataset_fs import VideoDataSet, VideoEpisodicDataSet
import torch.nn.functional as F
from gtad_lib.classifier_v2 import Classifier ## changed
from visdom_logger import VisdomLogger
from gtad_lib.util import AverageMeter, batch_intersectionAndUnionGPU, get_model_dir, main_process
from gtad_lib.util import find_free_port, setup, cleanup, to_one_hot, intersectionAndUnionGPU
from collections import defaultdict

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt = opts.parse_opt()
    opt = vars(opt)
    if not os.path.exists(opt['output'] + "/results1"):
        os.makedirs(opt['output'] + "/results1")

    data_path = opt["output"]+"/results1/"
    for fls in os.listdir(data_path):
        os.remove(os.path.join(data_path,fls))

    model = GTAD(opt)
    model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
    checkpoint = torch.load(opt["output"] + "/GTAD_best.pth.tar")
    model.load_state_dict(checkpoint['state_dict'])
    # print("bttle_dim", model.module.bottleneck_dim)
    model.eval()

    valid_mode = "Episodic" ## "Standard" or "Episodic"

    if valid_mode == "Standard" : 
        test_loader = torch.utils.data.DataLoader(VideoDataSet(opt, subset="validation", mode='inference'),
                                                batch_size=1, shuffle=False,
                                                num_workers=8, pin_memory=True, drop_last=False)
    else :
        test_loader = torch.utils.data.DataLoader(VideoEpisodicDataSet(opt, subset="validation", mode='inference'),
                                                batch_size=1, shuffle=False,
                                                num_workers=8, pin_memory=True, drop_last=False)

    tscale = opt["temporal_scale"]
    print("Inference start")
    if valid_mode == "Standard": 
        with torch.no_grad():
            for idx, input_data in test_loader:
                video_name = test_loader.dataset.video_list[idx[0]]
                # offset = min(test_loader.dataset.data['indices'][idx[0]])
                # video_name = video_name+'_{}'.format(math.floor(offset/250))
                input_data = input_data.cuda()

                # forward pass
                confidence_map, start, end = model(input_data)

                start_scores = start[0].detach().cpu().numpy()
                end_scores = end[0].detach().cpu().numpy()
                clr_confidence = (confidence_map[0][1]).detach().cpu().numpy()
                reg_confidence = (confidence_map[0][0]).detach().cpu().numpy()

                max_start = max(start_scores)
                max_end = max(end_scores)

                # use BMN post-processing to boost performance
                start_bins = np.zeros(len(start_scores))
                start_bins[0] = 1  # [1,0,0...,0,1]
                for idx in range(1, tscale - 1):
                    if start_scores[idx] > start_scores[idx + 1] and start_scores[idx] > start_scores[idx - 1]:
                        start_bins[idx] = 1
                    elif start_scores[idx] > (0.5 * max_start):
                        start_bins[idx] = 1

                end_bins = np.zeros(len(end_scores))
                end_bins[-1] = 1
                for idx in range(1, tscale - 1):
                    if end_scores[idx] > end_scores[idx + 1] and end_scores[idx] > end_scores[idx - 1]:
                        end_bins[idx] = 1
                    elif end_scores[idx] > (0.5 * max_end):
                        end_bins[idx] = 1

                # enumerate sub-graphs as proposals
                new_props = []
                for idx in range(opt["max_duration"]):
                    for jdx in range(opt["temporal_scale"]):
                        start_index = jdx
                        end_index = start_index + idx+1
                        if end_index < opt["temporal_scale"] and start_bins[start_index] == 1 and end_bins[end_index] == 1:
                            xmin = start_index / opt['temporal_scale']
                            xmax = end_index / opt['temporal_scale']
                            clr_score = clr_confidence[idx, jdx]
                            reg_score = reg_confidence[idx, jdx]
                            new_props.append([xmin, xmax, clr_score, reg_score])
                new_props = np.stack(new_props)

                col_name = ["xmin", "xmax", "clr_score", "reg_socre"]
                new_df = pd.DataFrame(new_props, columns=col_name)
                new_df.to_csv(opt["output"]+"/results/" + video_name + ".csv", index=False)

    else:
        # with torch.no_grad():
            ## NO_GRAD SHOULD BE REMPOVED
            ## to do : we take features from sub_graph module of dimension [batch, c, 100, 100] c--> 9216
            nb_episodes = opt["episode"] 
            # nb_episodes = 50
            testing_mode = False
            # model_feat = model.module.goi_align
            # int. feat_dim : 100 x 100 x 9216
            ## 100 X 1
            # model.module.extract_features(input_data)
            # c = model.module.bottleneck_dim
            # h = model.module.feature_res[0]
            # w = model.module.feature_res[1]
            h = model.module.features_dim[0]
            w = model.module.features_dim[1]
            c = 256 # 9216
            batch_size_val = 2 # consumes gpu-ram
            shot = opt["shot"]
            norm_feat = True
            n_runs=2
            H= 100
            W = 1
            runtimes = torch.zeros(n_runs)
            deltas_init = torch.zeros((n_runs, nb_episodes, batch_size_val))
            deltas_final = torch.zeros((n_runs, nb_episodes, batch_size_val))
            val_IoUs = np.zeros(n_runs)
            val_losses = np.zeros(n_runs) 
            use_callback = False

            # ========== Perform the runs  ==========
            for run in tqdm(range(n_runs)): ## epochs


                loss_meter = AverageMeter()
                iter_num = 0
                cls_intersection = defaultdict(int)  # Default value is 0
                cls_union = defaultdict(int)
                IoU = defaultdict(int)
    # =============== episode = group of tasks ===============
                # batch_size_val
                # shot
                for e in tqdm(range(nb_episodes)):
                    features_s = torch.zeros(batch_size_val, shot, c, h, w).to(device)
                    features_s_grad = torch.zeros(batch_size_val, shot, c, h, w).to(device)
                    features_q = torch.zeros(batch_size_val, 1, c, h, w).to(device)
                    features_q_grad = torch.zeros(batch_size_val, 1, c, h, w).to(device)
                    gt_s = torch.zeros(batch_size_val, shot, 100,
                                            1).long().to(device)
                    gt_q = torch.zeros(batch_size_val, 1, 100,
                                            1).long().to(device)
                    n_shots = torch.zeros(batch_size_val).to(device)
                    classes = []  # All classes considered in the tasks
                    iter_loader = iter(test_loader)
        # =========== Generate tasks and extract features for each task ===============
                    # i=0 
                    ## batch_size_ means no of task per episode
                    for i in range(batch_size_val):
                        try:
                            idx, input_data, qry_img, q_label, spprt_imgs, s_label, subcls, video_name = iter_loader.next()

                        except:
                            iter_loader = iter(test_loader)
                            idx, input_data, qry_img, q_label, spprt_imgs, s_label, subcls, video_name = iter_loader.next()


                        # iter_loader = iter(test_loader)
                        # idx, input_data, qry_img, q_label, spprt_imgs, s_label, subcls, video_name = iter_loader.next()
                        # idx, input_data, qry_img, q_label, spprt_imgs, s_label, subcls = test_loader.next()
                        input_data = input_data.cuda()
                        qry_img = qry_img.cuda()
                        # print("vid_name", video_name)
                        q_label = q_label.cuda()
                        spprt_imgs = spprt_imgs.cuda()
                        s_label = s_label.cuda()
                        iter_num += 1
                        # print(spprt_imgs.size())
                        # print(len(subcls))
                        # print("qry_img", qry_img.size())
                        # print("support img", spprt_imgs.size())
                        # split = spprt_imgs.size(2)/shot
                        # f_s = model.module.extract_features(spprt_imgs[:,i*400:(i+1)*400,:])
                        f_s = model.module.extract_features(spprt_imgs.squeeze(0)) ## spprt_imgs --? [1,shot,400,100]
                        f_q = model.module.extract_features(qry_img)
                        # print("suppport", f_s.size())
                        # print("query", f_q.size())
                        shot = f_s.size(0)
                        n_shots[i] = shot
                        features_s[i,:shot] = f_s.detach()
                        features_s_grad[i,:shot] = f_s
                        # print("suppport", features_s.size())
                        features_q[i] = f_q.detach()
                        features_q_grad[i] = f_q
                        # gt_s[i,:shot] = s_label[:,i*100:(i+1)*100,:]
                        gt_s[i,:shot] = s_label
                        gt_q[i,0] = q_label
                        classes.append([class_.item() for class_ in subcls])
                        
                            
                    # print("class", classes)#

        # =========== Normalize features along channel dimension ===============
                    ## feat_dim : [task=batch, shot=1, c, h, w]
                    if norm_feat:
                        features_s = F.normalize(features_s, dim=1)
                        features_q = F.normalize(features_q, dim=1)    
        # =========== Create a callback is args.visdom_port != -1 ===============
                    callback = VisdomLogger(port=2) if use_callback else None
        # ===========  Initialize the classifier + prototypes + F/B parameter Π ===============                    
                    classifier = Classifier()   

                    # n_task, shot, c, h, w = features_s.size()
                    # print("before", features_s.size())
                    # features_s = F.adaptive_avg_pool2d(features_s.view(n_task*shot,c, h, w),[10 , 10])
                    # features_q = F.adaptive_avg_pool2d(features_q.view(n_task,c, h, w),[10 , 10])
                    # b, c, h, w = features_s.size()
                    # features_s = features_s.view(n_task,shot,c, h, w)
                    # features_q = features_q.view(n_task,1,c, h, w)
                    # print("after", features_s.size())

                    classifier.init_prototypes(features_s, features_s_grad, features_q, features_q_grad, gt_s, gt_q, classes, callback)
                    batch_deltas = classifier.compute_FB_param(features_q=features_q, gt_q=gt_q)
                    # print("delta", batch_deltas)
                    deltas_init[run, e, :] = batch_deltas.cpu()

        # =========== Perform RePRI inference ===============
                    
                    batch_deltas = classifier.RePRI(features_s, features_q, gt_s, gt_q, classes, n_shots, callback) ## trained W,b and Transformer weights
                    print("Linear classifier in epoch "+ str(n_runs)+" trained")
                    deltas_final[run, e, :] = batch_deltas
                    # t1 = time.time()
                    # runtime += t1 - t0
                    ### after trained W , pass it to train transformer using CE loss and 
                    frozen_trans = classifier.TransformerRePRI(features_q,gt_q) ## added
                    logits_u = classifier.get_logits_frozen(features_q,frozen_trans) ## added
                    ### after trained prototypes doing test time logits pass 
                    # logits = classifier.get_logits(features_q)  # [n_tasks, shot, h, w] # used transformer here to get updated feature values
                    logits = F.interpolate(logits_u,
                                        size=(H, W),
                                        mode='bilinear',
                                        align_corners=True)
                    probas = classifier.get_probas(logits).detach()
                    # print(probas.size()) # [task=2, shot=1, class=2, h=100, w=1]
                    intersection, union, _ = batch_intersectionAndUnionGPU(probas, gt_q, 2, classes, video_name)  # [n_tasks, shot, num_class]
                    intersection, union = intersection.cpu(), union.cpu() ### intersection_size : [50,1,2] --> has values of one hot
                    # print(intersection.size())

        # ================== Log metrics ==================
                    one_hot_gt = to_one_hot(gt_q, 2) ## gt_q --> had value among 1 and 255 , one_hot_gt --> contains value among 1 and 0 
                    # print(one_hot_gt)
                    valid_pixels = gt_q != 0
                    loss = classifier.get_ce(probas, valid_pixels, one_hot_gt, reduction='mean')
                    loss_meter.update(loss.item())
                    for i, task_classes in enumerate(classes):
                        # print("task_classes" , task_classes)
                        for j, class_ in enumerate(task_classes):
                            cls_intersection[class_] += intersection[i, 0, j + 1]  # Do not count background
                            cls_union[class_] += union[i, 0, j + 1]

                    for class_ in cls_union:
                        IoU[class_] = cls_intersection[class_] / (cls_union[class_] + 1e-10)
                    # print("num",iter_num)
                    if (iter_num % 100 == 0):
                        mIoU = np.mean([IoU[i] for i in IoU])
                        print('Test: [{}/{}] '
                            'mIoU {:.4f} '
                            'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '.format(iter_num,
                                                                                        1000,
                                                                                        mIoU,
                                                                                        loss_meter=loss_meter,
                                                                                        ))

                mIoU = np.mean(list(IoU.values()))
                print('mIoU---Val result: mIoU {:.4f}.'.format(mIoU))
                for class_ in cls_union:
                    print("Class {} : {:.4f}".format(class_, IoU[class_]))

                val_IoUs[run] = mIoU
                val_losses[run] = loss_meter.avg

                if testing_mode:
                    classifier = Classifier()
                    for j in range(0,500):
                        idx, input_data, qry_img, q_label, spprt_imgs, s_label, subcls, video_name = iter_loader.next()
                        qry_img = qry_img.cuda()
                        # print("vid_name", video_name)
                        q_label = q_label.cuda()
                        f_q = model.module.extract_features(qry_img)
                        classifier.infer(f_q,q_label)
                                      
            # for idx, input_data in test_loader:
            #     video_name = test_loader.dataset.video_list[idx[0]]
            #     # offset = min(test_loader.dataset.data['indices'][idx[0]])
            #     # video_name = video_name+'_{}'.format(math.floor(offset/250))
            #     input_data = input_data.cuda()

            #     # forward pass
            #     confidence_map, start, end = model(input_data)

            #     start_scores = start[0].detach().cpu().numpy()
            #     end_scores = end[0].detach().cpu().numpy()
            #     clr_confidence = (confidence_map[0][1]).detach().cpu().numpy()
            #     reg_confidence = (confidence_map[0][0]).detach().cpu().numpy()

            #     max_start = max(start_scores)
            #     max_end = max(end_scores)

            #     # use BMN post-processing to boost performance
            #     start_bins = np.zeros(len(start_scores))
            #     start_bins[0] = 1  # [1,0,0...,0,1]
            #     for idx in range(1, tscale - 1):
            #         if start_scores[idx] > start_scores[idx + 1] and start_scores[idx] > start_scores[idx - 1]:
            #             start_bins[idx] = 1
            #         elif start_scores[idx] > (0.5 * max_start):
            #             start_bins[idx] = 1

            #     end_bins = np.zeros(len(end_scores))
            #     end_bins[-1] = 1
            #     for idx in range(1, tscale - 1):
            #         if end_scores[idx] > end_scores[idx + 1] and end_scores[idx] > end_scores[idx - 1]:
            #             end_bins[idx] = 1
            #         elif end_scores[idx] > (0.5 * max_end):
            #             end_bins[idx] = 1

            #     # enumerate sub-graphs as proposals
            #     new_props = []
            #     for idx in range(opt["max_duration"]):
            #         for jdx in range(opt["temporal_scale"]):
            #             start_index = jdx
            #             end_index = start_index + idx+1
            #             if end_index < opt["temporal_scale"] and start_bins[start_index] == 1 and end_bins[end_index] == 1:
            #                 xmin = start_index / opt['temporal_scale']
            #                 xmax = end_index / opt['temporal_scale']
            #                 clr_score = clr_confidence[idx, jdx]
            #                 reg_score = reg_confidence[idx, jdx]
            #                 new_props.append([xmin, xmax, clr_score, reg_score])
            #     new_props = np.stack(new_props)

            #     col_name = ["xmin", "xmax", "clr_score", "reg_socre"]
            #     new_df = pd.DataFrame(new_props, columns=col_name)
            #     new_df.to_csv(opt["output"]+"/results/" + video_name + ".csv", index=False)


    print("Inference finished")