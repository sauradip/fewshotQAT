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
from scipy import ndimage

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
    if opt["cross_domain"]:
        checkpoint = torch.load("/home/phd/Desktop/sauradip_research/TAL/gtad_thumos/gtad/output/default/GTAD_best.pth.tar")
    else :
        checkpoint = torch.load(opt["output"] + "/GTAD_best.pth.tar")
    model.load_state_dict(checkpoint['state_dict'])
    # print("bttle_dim", model.module.bottleneck_dim)
    model.eval()

    if opt["meta_learn"]:
        valid_mode = "Episodic_train" ## "Standard" or "Episodic"
    else:
        valid_mode = "Episodic_test"

    if valid_mode == "Standard" : 
        test_loader = torch.utils.data.DataLoader(VideoDataSet(opt, subset="validation", mode='inference'),
                                                batch_size=1, shuffle=False,
                                                num_workers=8, pin_memory=True, drop_last=False)
    elif valid_mode == "Episodic_train" :
        test_loader = torch.utils.data.DataLoader(VideoEpisodicDataSet(opt, subset="training" , mode="train"),
                                                batch_size=1, shuffle=True,
                                                num_workers=8, pin_memory=True, drop_last=False)
    
    elif valid_mode == "Episodic_test" :
        test_loader1 = torch.utils.data.DataLoader(VideoEpisodicDataSet(opt, subset="validation", mode='inference'),
                                                batch_size=1, shuffle=False,
                                                num_workers=8, pin_memory=True, drop_last=False)

    tscale = opt["temporal_scale"]
    print("Inference start")






    def findTAL(logits,target, vid_name):
        
        n_tasks, shot, num_classes, H, W = logits.size()
        temp_list=[]
        for task in range(n_tasks):
            for shot1 in range(shot):
                # print("clss",cnames[sub_cls[task][shot1]])
                # print("query_vid", vid_name[0])
                fg = logits[task][shot1][0].detach().cpu().numpy()
                bg = logits[task][shot1][1].detach().cpu().numpy()
                # print(fg)
                # thres = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
                thres = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
                # thres = [x / 100.0 for x in range(0, 100, 5)]
                # print(thres)
                for i in thres:
                    fg_thres = fg > i
                    integer_map = map(int,fg_thres)
                    filtered_seq_int = list(integer_map)
                    filled_fg_1 = ndimage.binary_fill_holes(filtered_seq_int).astype(int).tolist() # [001111111000] --> [0.2 0.1 0.4 0.4 0.5 0.5 0.6 0.1 0.1] --->  
                    filled_fg = filtered_seq_int[9:97]
                    # print(filled_fg)
                    if 1 in filled_fg:            
                        start_pt = filled_fg.index(1)
                        end_pt = len(filled_fg) - 1 - filled_fg[::-1].index(1)
                        if (end_pt/100) > (start_pt/100) and ((end_pt - start_pt)/100) > 0.1 :
                            # reg_vals = np.mean(fg[start_pt:end_pt])
                            # reg_vals = stats.mode(fg[start_pt:end_pt])
                            conf_vals = np.amax(fg[start_pt:end_pt])
                            # print("conf",conf_vals)
                            temp_list.append([start_pt/100,end_pt/100,conf_vals,conf_vals])
                            
                    if 1 in filled_fg_1:            
                        start_pt = filled_fg_1.index(1)
                        end_pt = len(filled_fg_1) - 1 - filled_fg_1[::-1].index(1)
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
        new_df.to_csv(opt["output"]+"/results2/" + video_name + ".csv", index=False)

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
            ## use base split
            meta_learn = opt["meta_learn"]
            if meta_learn :
                nb_episodes = opt["episode"] # 200
                # nb_episodes = 50
                # testing_mode = False
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
                batch_size_val = 1 # consumes gpu-ram
                shot = opt["shot"]
                norm_feat = True
                n_runs=10 
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
                            # print("suppport", f_s.parameters())
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
                        # batch_deltas = classifier.compute_FB_param(features_q=features_q, gt_q=gt_q)
                        # # print("delta", batch_deltas)
                        # deltas_init[run, e, :] = batch_deltas.cpu()

            # =========== Perform RePRI inference ===============
            
                        batch_deltas = classifier.RePRI(features_s, features_q, gt_s, gt_q, classes, n_shots, callback) ## trained W,b 
                        # print("Linear classifier in epoch "+ str(run)+" trained \n")
                        deltas_final[run, e, :] = batch_deltas
                        # t1 = time.time()
                        # runtime += t1 - t0
                        ### after trained W , pass it to train transformer using CE loss and 
                        if meta_learn:
                            model_trans = classifier.TransformerRePRI(features_s,features_q,gt_q,gt_s,mode="train") ## added
                            # print(model_trans)
                            # print("Transformer in epoch "+ str(run)+" trained \n")
                            state = {'epoch': n_runs + 1,'state_dict': model_trans.state_dict()}
                            # print(state)
                            torch.save(state, opt["output"] + "/Transformer_checkpoint_shot_"+str(opt["shot"])+".pth.tar")
                            # print("--------------------------------------------- \n")
            else : 
                if not os.path.exists(opt['output'] + "/results2"):
                    os.makedirs(opt['output'] + "/results2")
                data_path = opt["output"]+"/results2/"
                for fls in os.listdir(data_path):
                    os.remove(os.path.join(data_path,fls))
                
                nb_episodes = opt["episode"] 
                # nb_episodes = 50
                # testing_mode = False
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
                batch_size_val = 1 # consumes gpu-ram
                shot = opt["shot"]
                norm_feat = False
                n_runs=10
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
                    cnt=0
                    cls_intersection = defaultdict(int)  # Default value is 0
                    cls_union = defaultdict(int)
                    IoU = defaultdict(int)
        # =============== episode = group of tasks ===============
                    # batch_size_val
                    # shot
                    for e in tqdm(range(nb_episodes)):
                        cnt+=1
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
                        iter_loader = iter(test_loader1)
            # =========== Generate tasks and extract features for each task ===============
                        # i=0 
                        ## batch_size_ means no of task per episode
                        for i in range(batch_size_val):
                            try:
                                idx, input_data, qry_img, q_label, spprt_imgs, s_label, subcls, video_name = iter_loader.next()

                            except:
                                iter_loader = iter(test_loader1)
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
                            # print("suppport", f_s.())
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
                        # print("support_feat",features_s)
                        classifier.init_prototypes(features_s, features_s_grad, features_q, features_q_grad, gt_s, gt_q, classes, callback)
                        # batch_deltas = classifier.compute_FB_param(features_q=features_q, gt_q=gt_q)
                        # # print("delta", batch_deltas)
                        # deltas_init[run, e, :] = batch_deltas.cpu()

            # =========== Perform RePRI inference ===============
                        batch_deltas = classifier.RePRI(features_s, features_q, gt_s, gt_q, classes, n_shots, callback) ## trained W,b 
                        pred_q = classifier.TransformerRePRI(features_s,features_q,gt_q,gt_s,cnt,mode="test") ## added pred_q per episode
                        findTAL(pred_q,gt_q,video_name)




                    # print("done")
        #             logits_u = classifier.get_logits_frozen(features_q,frozen_trans) ## added
        #             ### after trained prototypes doing test time logits pass 
        #             # logits = classifier.get_logits(features_q)  # [n_tasks, shot, h, w] # used transformer here to get updated feature values
        #             logits = F.interpolate(logits_u,
        #                                 size=(H, W),
        #                                 mode='bilinear',
        #                                 align_corners=True)
        #             probas = classifier.get_probas(logits).detach()
        #             # print(probas.size()) # [task=2, shot=1, class=2, h=100, w=1]
        #             intersection, union, _ = batch_intersectionAndUnionGPU(probas, gt_q, 2, classes, video_name)  # [n_tasks, shot, num_class]
        #             intersection, union = intersection.cpu(), union.cpu() ### intersection_size : [50,1,2] --> has values of one hot
        #             # print(intersection.size())

        # # ================== Log metrics ==================
        #             one_hot_gt = to_one_hot(gt_q, 2) ## gt_q --> had value among 1 and 255 , one_hot_gt --> contains value among 1 and 0 
        #             # print(one_hot_gt)
        #             valid_pixels = gt_q != 0
        #             loss = classifier.get_ce(probas, valid_pixels, one_hot_gt, reduction='mean')
        #             loss_meter.update(loss.item())
        #             for i, task_classes in enumerate(classes):
        #                 # print("task_classes" , task_classes)
        #                 for j, class_ in enumerate(task_classes):
        #                     cls_intersection[class_] += intersection[i, 0, j + 1]  # Do not count background
        #                     cls_union[class_] += union[i, 0, j + 1]

        #             for class_ in cls_union:
        #                 IoU[class_] = cls_intersection[class_] / (cls_union[class_] + 1e-10)
        #             # print("num",iter_num)
        #             if (iter_num % 100 == 0):
        #                 mIoU = np.mean([IoU[i] for i in IoU])
        #                 print('Test: [{}/{}] '
        #                     'mIoU {:.4f} '
        #                     'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '.format(iter_num,
        #                                                                                 1000,
        #                                                                                 mIoU,
        #                                                                                 loss_meter=loss_meter,
        #                                                                                 ))

        #         mIoU = np.mean(list(IoU.values()))
        #         print('mIoU---Val result: mIoU {:.4f}.'.format(mIoU))
        #         for class_ in cls_union:
        #             print("Class {} : {:.4f}".format(class_, IoU[class_]))

        #         val_IoUs[run] = mIoU
        #         val_losses[run] = loss_meter.avg

        #         if testing_mode:
        #             classifier = Classifier()
        #             for j in range(0,500):
        #                 idx, input_data, qry_img, q_label, spprt_imgs, s_label, subcls, video_name = iter_loader.next()
        #                 qry_img = qry_img.cuda()
        #                 # print("vid_name", video_name)
        #                 q_label = q_label.cuda()
        #                 f_q = model.module.extract_features(qry_img)
        #                 classifier.infer(f_q,q_label)
                                      
        #     # for idx, input_data in test_loader:
        #     #     video_name = test_loader.dataset.video_list[idx[0]]
        #     #     # offset = min(test_loader.dataset.data['indices'][idx[0]])
        #     #     # video_name = video_name+'_{}'.format(math.floor(offset/250))
        #     #     input_data = input_data.cuda()

        #     #     # forward pass
        #     #     confidence_map, start, end = model(input_data)

        #     #     start_scores = start[0].detach().cpu().numpy()
        #     #     end_scores = end[0].detach().cpu().numpy()
        #     #     clr_confidence = (confidence_map[0][1]).detach().cpu().numpy()
        #     #     reg_confidence = (confidence_map[0][0]).detach().cpu().numpy()

        #     #     max_start = max(start_scores)
        #     #     max_end = max(end_scores)

        #     #     # use BMN post-processing to boost performance
        #     #     start_bins = np.zeros(len(start_scores))
        #     #     start_bins[0] = 1  # [1,0,0...,0,1]
        #     #     for idx in range(1, tscale - 1):
        #     #         if start_scores[idx] > start_scores[idx + 1] and start_scores[idx] > start_scores[idx - 1]:
        #     #             start_bins[idx] = 1
        #     #         elif start_scores[idx] > (0.5 * max_start):
        #     #             start_bins[idx] = 1

        #     #     end_bins = np.zeros(len(end_scores))
        #     #     end_bins[-1] = 1
        #     #     for idx in range(1, tscale - 1):
        #     #         if end_scores[idx] > end_scores[idx + 1] and end_scores[idx] > end_scores[idx - 1]:
        #     #             end_bins[idx] = 1
        #     #         elif end_scores[idx] > (0.5 * max_end):
        #     #             end_bins[idx] = 1

        #     #     # enumerate sub-graphs as proposals
        #     #     new_props = []
        #     #     for idx in range(opt["max_duration"]):
        #     #         for jdx in range(opt["temporal_scale"]):
        #     #             start_index = jdx
        #     #             end_index = start_index + idx+1
        #     #             if end_index < opt["temporal_scale"] and start_bins[start_index] == 1 and end_bins[end_index] == 1:
        #     #                 xmin = start_index / opt['temporal_scale']
        #     #                 xmax = end_index / opt['temporal_scale']
        #     #                 clr_score = clr_confidence[idx, jdx]
        #     #                 reg_score = reg_confidence[idx, jdx]
        #     #                 new_props.append([xmin, xmax, clr_score, reg_score])
        #     #     new_props = np.stack(new_props)

        #     #     col_name = ["xmin", "xmax", "clr_score", "reg_socre"]
        #     #     new_df = pd.DataFrame(new_props, columns=col_name)
        #     #     new_df.to_csv(opt["output"]+"/results/" + video_name + ".csv", index=False)


    print("Inference finished")