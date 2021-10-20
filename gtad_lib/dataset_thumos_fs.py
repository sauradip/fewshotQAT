# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import json, pickle
import torch.utils.data as data
import torch
import h5py

def ioa_with_anchors(anchors_min, anchors_max, box_min, box_max):
    # calculate the overlap proportion between the anchor and all bbox for supervise signal,
    # the length of the anchor is 0.01
    len_anchors = anchors_max - anchors_min
    int_xmin = np.maximum(anchors_min, box_min)
    int_xmax = np.minimum(anchors_max, box_max)
    inter_len = np.maximum(int_xmax - int_xmin, 0.)
    scores = np.divide(inter_len, len_anchors)
    return scores


def iou_with_anchors(anchors_min, anchors_max, box_min, box_max):
    """Compute jaccard score between a box and the anchors.
    """
    len_anchors = anchors_max - anchors_min
    int_xmin = np.maximum(anchors_min, box_min)
    int_xmax = np.minimum(anchors_max, box_max)
    inter_len = np.maximum(int_xmax - int_xmin, 0.)
    union_len = len_anchors - inter_len + box_max - box_min
    # print inter_len,union_len
    jaccard = np.divide(inter_len, union_len)
    return jaccard


def load_json(file):
    with open(file) as json_file:
        json_data = json.load(json_file)
        return json_data

class VideoEpisodicDataSet(data.Dataset):
    def __init__(self, opt, subset="val", mode="test"):
        self.temporal_scale = opt["temporal_scale"]  # 100
        self.temporal_gap = 1. / self.temporal_scale
        self.subset = subset
        self.mode = mode
        self.feature_path = opt["feature_path"]
        self.video_info_path = "/home/phd/Desktop/sauradip_research/TAL/gtad/data/activitynet_annotations/video_info_new.csv"
        self.video_anno_path = "/home/phd/Desktop/sauradip_research/TAL/gtad/data/activitynet_annotations/anet_anno_action.json"
        self._getDatasetDictEpisode() # to do ---> same as self.data_list in RePri
        # self._get_match_map()
        self.shot = opt["shot"] # to do
        # self.transform =  transform # to do 
        actions = pd.read_csv("/media/phd/SAURADIP5TB/ACLPT/activitynet_annotations/action_name.csv")
        # self.cnames = dict_annts['list_classes']
        # self.cnames = actions.action.values.tolist()
        self.cnames = list(set(test_class).intersection(set(actions.action.values.tolist())))
        # print(self.cnames)
        self.db = load_json("/media/phd/SAURADIP5TB/ACLPT/activitynet_annotations/anet_anno_action.json")
        
    def __len__(self):
        return len(self.video_list)

    def __getitem__(self,index):
        video_data = self._load_file(index) ## to do : add load_file method --> gives features
        query_vid, target , support_vid , support_label , subcls_list , video_name = self.load_episodic_data(index)
        
        return index, video_data, query_vid, target , support_vid , support_label , subcls_list , video_name

    def _load_file(self, index):
        video_name = self.video_list[index]
        if self.feature_path[-1]=='/':  # h5 files are in seperated files
            if ',' in self.feature_path: # concatenation of two
                feat = []
                for feature_path in self.feature_path.split(','):
                    with h5py.File(feature_path+video_name+'.h5', 'r') as f:
                        feat.append(f[video_name][:])
                feat = np.concatenate(feat,axis=1)
            else:
                # print(video_name, 'not found!!!!!!!!!!!')
                # feat = torch.randn((100,512))
                with h5py.File(self.feature_path+video_name+'.h5', 'r') as f:
                    feat = f[video_name][:]
        else:
            with h5py.File(self.feature_path, 'r') as features_h5:
                feat = features_h5[video_name][()]
        # video_data = torch.randn((100,400))
        video_data = torch.Tensor(feat)
        video_data = torch.transpose(video_data, 0, 1)
        if video_data.shape[0]!=self.temporal_scale: # rescale to fixed shape
            video_data = F.interpolate(video_data.unsqueeze(0), size=self.temporal_scale, mode='linear',align_corners=False)[0,...]
        video_data.float()
        return video_data

    def _load_file_by_name(self, vid_name):
        video_name = vid_name
        if self.feature_path[-1]=='/':  # h5 files are in seperated files
            if ',' in self.feature_path: # concatenation of two
                feat = []
                for feature_path in self.feature_path.split(','):
                    with h5py.File(feature_path+video_name+'.h5', 'r') as f:
                        feat.append(f[video_name][:])
                feat = np.concatenate(feat,axis=1)
            else:
                # print(video_name, 'not found!!!!!!!!!!!')
                # feat = torch.randn((100,512))
                with h5py.File(self.feature_path+video_name+'.h5', 'r') as f:
                    feat = f[video_name][:]
        else:
            with h5py.File(self.feature_path, 'r') as features_h5:
                feat = features_h5[video_name][()]
        # video_data = torch.randn((100,400))
        video_data = torch.Tensor(feat)
        video_data = torch.transpose(video_data, 0, 1)
        if video_data.shape[0]!=self.temporal_scale: # rescale to fixed shape
            video_data = F.interpolate(video_data.unsqueeze(0), size=self.temporal_scale, mode='linear',align_corners=False)[0,...]
        video_data.float()
        return video_data

    def _getAnntsCwise(self,annts):
        annts_cwise ={}
        for i, v in enumerate(self.video_list):
            video_frame = annts[v]['duration_frame']
            video_second = annts[v]['duration_second']
            feature_frame = annts[v]['feature_frame']
            corrected_second = float(feature_frame) / video_frame * video_second  # there are some frames not used

            for a in annts[v]['annotations']:
                cn = a['label']
                corr_start = max(min(1, a['segment'][0] / corrected_second), 0)
                corr_end = max(min(1, a['segment'][1] / corrected_second), 0)
                if cn in test_class and (corr_end - corr_start) > 0.7: 
                    
                    if cn not in annts_cwise:
                        annts_cwise[cn] = []
                    # annts_cwise[cn].append([i, a['segment'][0], a['segment'][1]])
                    annts_cwise[cn].append(v) ##just index
        
        return annts_cwise

    def _getDatasetDictEpisode(self):
        anno_df = pd.read_csv(self.video_info_path)
        anno_database = load_json(self.video_anno_path)
        self.video_dict = {}
        for i in range(len(anno_df)):
            video_name = anno_df.video.values[i]
            video_info = anno_database[video_name]
            video_subset = anno_df.subset.values[i]
            ## to do : 1) check unique label class 2) check FG class in test split
            ## 3) check if < 768 frames or not 
            if len(video_info["annotations"]) > 0:
                labels = video_info["annotations"][0]["label"]

            if labels in test_class and self.subset in video_subset:
                self.video_dict[video_name] = video_info
                
        self.video_list = list(self.video_dict.keys())
        print("%s subset video numbers: %d" % (self.subset, len(self.video_list)))

    def _getLabel(self,idx):
        video_name = self.video_list[idx] ## contains video 
        video_info = self.video_dict[video_name]
        video_frame = video_info['duration_frame']
        video_second = video_info['duration_second']
        feature_frame = video_info['feature_frame']
        corrected_second = float(feature_frame) / video_frame * video_second  # there are some frames not used
        video_labels = video_info['annotations']
        gt_bbox = []
        gt_iou_map = []
        for j in range(len(video_labels)):
            tmp_info = video_labels[j]
            tmp_start = max(min(1, tmp_info['segment'][0] / corrected_second), 0)
            tmp_end = max(min(1, tmp_info['segment'][1] / corrected_second), 0)
            gt_bbox.append([tmp_start, tmp_end])

        gt_bbox = np.array(gt_bbox)
        gt_xmins = gt_bbox[:, 0]
        gt_xmaxs = gt_bbox[:, 1]
        action_mask = np.zeros([100,1])

        start_indexes = []
        end_indexes = []
        for idx in range(len(gt_xmins)):
          start_indexes.append(math.floor(100*gt_xmins[idx]))
          end_indexes.append(math.floor(100*gt_xmaxs[idx]))

        for idx in range(len(start_indexes)):
          action_mask[start_indexes[idx]:end_indexes[idx]] = 1
        
        return torch.Tensor(action_mask)

    def _getLabel_by_name(self,vid_name):
        video_name = vid_name ## contains video 
        video_info = self.video_dict[video_name]
        video_frame = video_info['duration_frame']
        video_second = video_info['duration_second']
        feature_frame = video_info['feature_frame']
        corrected_second = float(feature_frame) / video_frame * video_second  # there are some frames not used
        video_labels = video_info['annotations']
        gt_bbox = []
        gt_iou_map = []
        for j in range(len(video_labels)):
            tmp_info = video_labels[j]
            tmp_start = max(min(1, tmp_info['segment'][0] / corrected_second), 0)
            tmp_end = max(min(1, tmp_info['segment'][1] / corrected_second), 0)
            gt_bbox.append([tmp_start, tmp_end])

        gt_bbox = np.array(gt_bbox)
        gt_xmins = gt_bbox[:, 0]
        gt_xmaxs = gt_bbox[:, 1]
        action_mask = np.zeros([100,1])

        start_indexes = []
        end_indexes = []
        for idx in range(len(gt_xmins)):
          start_indexes.append(math.floor(100*gt_xmins[idx]))
          end_indexes.append(math.floor(100*gt_xmaxs[idx]))

        for idx in range(len(start_indexes)):
          action_mask[start_indexes[idx]:end_indexes[idx]] = 1
        
        return torch.Tensor(action_mask)
        
    def load_episodic_data(self,index):

        # video_name = self.video_list[index] ## contains video 
        # video_info = self.video_dict[video_name]
        # video_frame = video_info['duration_frame']
        # video_second = video_info['duration_second']
        # feature_frame = video_info['feature_frame']
        # corrected_second = float(feature_frame) / video_frame * video_second  # there are some frames not used
        # video_labels = video_info['annotations']  # the measurement is second, not frame
        # labels = video_labels[0]["label"]
        
        shot = self.shot
        # gt_bbox = []
        # gt_iou_map = []
        # for j in range(len(video_labels)):
        #     tmp_info = video_labels[j]
        #     tmp_start = max(min(1, tmp_info['segment'][0] / corrected_second), 0)
        #     tmp_end = max(min(1, tmp_info['segment'][1] / corrected_second), 0)
        #     gt_bbox.append([tmp_start, tmp_end])

        # gt_bbox = np.array(gt_bbox)
        # gt_xmins = gt_bbox[:, 0]
        # gt_xmaxs = gt_bbox[:, 1]
        # action_mask = np.zeros([100,1])

        # start_indexes = []
        # end_indexes = []
        # for idx in range(len(gt_xmins)):
        #   start_indexes.append(math.floor(100*gt_xmins[idx]))
        #   end_indexes.append(math.floor(100*gt_xmaxs[idx]))

        # for idx in range(len(start_indexes)):
        #   action_mask[start_indexes[idx]:end_indexes[idx]] = 1

        # vid_label = np.zeros([len(test_class)])
        # vid_label[chosen_class] = 1
        

        ## action_mask : FG/BG ( 2 class )


        # ========= Read Query Video + Choose Label ======= #
        cwise_anno = self._getAnntsCwise(self.db)
        # print("cwise", cwise_anno)
        class_chosen = random.choice(test_class) ## query label
        subcls_list = [self.cnames.index(class_chosen)] ## index of chosen class among novel class 
        # print("class chosen", subcls_list)
        rand_vid_name = random.choice(cwise_anno[class_chosen])
        query_label = self._getLabel_by_name(rand_vid_name) ## action mask --> only FG / BG
        query_data = self._load_file_by_name(rand_vid_name)
        # class_chosen = labels ## query label
        
        # == From classes in query video, chose one randomly ===
        ## to do : get a dict with {class_id:video_ids} --> put in self.sub_class
        # class_chosen = np.random.choice(label_class) ## to do: 
        # label = vid_label
        # file_class_chosen = self.sub_class[class_chosen] ## to do : The sub_class has video_id having chosen class 
        # num_file = len(file_class_chosen)
        # self.db = load_json("/media/phd/SAURADIP5TB/ACLPT/activitynet_annotations/anet_anno_action.json")
        
        # print(cwise_anno)
        # print("class_chosen", class_chosen)
        file_class_chosen = set(cwise_anno[class_chosen])
        # print("file_class_chosen", set(cwise_anno[class_chosen]))
        # print("length", len(file_class_chosen))
        num_file = len(file_class_chosen)
        support_image_list = []
        support_label_list = []
        ## ========== Build SUpport ===========
        ## same class for S+Q : todo : support label and query label is same
        for k in range(shot):
            support_index = random.randint(1, num_file) - 1 ## -1 because one sample is already taken in query fom indexes
            support_vname = cwise_anno[class_chosen][support_index]
            support_data = self._load_file_by_name(support_vname)
            support_label = self._getLabel_by_name(support_vname)
            support_image_list.append(support_data)
            support_label_list.append(support_label)
        assert len(support_label_list) == shot and len(support_image_list) == shot


        spprt_vids = torch.cat(support_image_list,0).view(shot,400,100)
        spprt_labels = torch.cat(support_label_list,0).view(shot,100,1)
        # print(spprt_vids.size())
        ## query_data --> [batch,feat_dim,temp], query_label --> [batch,100,1], spprt_vids --> [batch,nshot,feat_dim, temp]
        ## spprt_labels --> [batch,nshot,temp,1]
        return query_data, query_label , spprt_vids , spprt_labels , subcls_list , rand_vid_name







class VideoDataSet(data.Dataset):  # thumos
    def __init__(self, opt, subset="train", mode="train"):
        self.temporal_scale = opt["temporal_scale"]  # 128
        self.temporal_gap = 1. / self.temporal_scale # 1/128
        self.subset = subset
        self.mode = mode
        self.feature_path = opt["feature_path"]
        self.video_info_path = opt["video_info"]
        self.video_anno_path = opt["video_anno"]
        self.feat_dim = opt['feat_dim']
        # Assuming someone wont outsmart this by mutating the dict 😝.
        # Consider to use YACS and FB code structure in the future.
        self.cfg = opt

        #### THUMOS
        self.skip_videoframes = opt['skip_videoframes']
        self.num_videoframes = opt['temporal_scale']
        self.max_duration = opt['max_duration']
        self.min_duration = opt['min_duration']
        if self.feature_path[-3:]=='200':
            self.feature_dirs = [self.feature_path + "/flow/csv", self.feature_path + "/rgb/csv"]
        else:
            self.feature_dirs = [self.feature_path]
        self._get_data()
        self.video_list = self.data['video_names']
        # self._getDatasetDict()
        self._get_match_map()

    def _getDatasetDict(self):
        anno_df = pd.read_csv(self.video_info_path)
        anno_database = load_json(self.video_anno_path)
        self.video_dict = {}
        for i in range(len(anno_df)):
            video_name = anno_df.video.values[i]
            video_info = anno_database[video_name]
            video_subset = anno_df.subset.values[i]
            if self.subset in video_subset:
                self.video_dict[video_name] = video_info
        self.video_list = list(self.video_dict.keys())
        print("%s subset video numbers: %d" % (self.subset, len(self.video_list)))

    def _get_video_data(self, data, index):
        return data['video_data'][index]


    def __getitem__(self, index):
        video_data = self._get_video_data(self.data, index) # get one from 2793
        video_data = torch.tensor(video_data.transpose())
        if self.mode == "train":
            match_score_start, match_score_end, confidence_score = self._get_train_label(index)
            return video_data, confidence_score, match_score_start, match_score_end
        else:
            return index, video_data

    def _get_match_map(self):
        match_map = []
        for idx in range(self.num_videoframes):
            tmp_match_window = []
            xmin = self.temporal_gap * idx
            for jdx in range(1, self.max_duration + 1):
                xmax = xmin + self.temporal_gap * jdx
                tmp_match_window.append([xmin, xmax])  # [0,0.01], [0,0.02], ... 64 x 2
            match_map.append(tmp_match_window)  # 128 x 64 x 2
        match_map = np.array(match_map)  # 128 x 64 x 2
        match_map = np.transpose(match_map, [1, 0, 2])  # [0,1] [1,2] [2,3].....[99,100], 64 x 128 x 2
        match_map = np.reshape(match_map, [-1, 2])  # [0,2] [1,3] [2,4].....[99,101]   # (duration x start) x 2
        self.match_map = match_map  # duration is same in row, start is same in col
        self.anchor_xmin = [self.temporal_gap * (i-0.5) for i in range(self.temporal_scale)]
        self.anchor_xmax = [self.temporal_gap * (i+0.5) for i in range(1, self.temporal_scale + 1)]

    def _get_train_label(self, index):
        # change the measurement from second to percentage
        # gt_bbox = []
        gt_iou_map = []
        gt_bbox = self.data['gt_bbox'][index]
        anchor_xmin = self.data['anchor_xmins'][index]
        anchor_xmax = self.data['anchor_xmaxs'][index]
        offset = int(min(anchor_xmin))
        for j in range(len(gt_bbox)):
            # tmp_info = video_labels[j]
            tmp_start = max(min(1, (gt_bbox[j][0]-offset)*self.temporal_gap/self.skip_videoframes), 0)
            tmp_end =   max(min(1, (gt_bbox[j][1]-offset)*self.temporal_gap/self.skip_videoframes), 0)
            # gt_bbox.append([tmp_start, tmp_end])
            tmp_gt_iou_map = iou_with_anchors(
                self.match_map[:, 0], self.match_map[:, 1], tmp_start, tmp_end)
            tmp_gt_iou_map = np.reshape(tmp_gt_iou_map,
                                        [self.max_duration,self.num_videoframes])
            gt_iou_map.append(tmp_gt_iou_map)
        gt_iou_map = np.array(gt_iou_map)
        gt_iou_map = np.max(gt_iou_map, axis=0)
        # if not gt_iou_map.max()>0.9:
        #     raise ValueError
        gt_iou_map = torch.Tensor(gt_iou_map)

        # generate R_s and R_e
        gt_bbox = np.array(gt_bbox)
        gt_xmins = gt_bbox[:, 0]
        gt_xmaxs = gt_bbox[:, 1]
        gt_lens = gt_xmaxs - gt_xmins
        gt_len_small = 3 * self.skip_videoframes
        gt_start_bboxs = np.stack((gt_xmins - gt_len_small / 2, gt_xmins + gt_len_small / 2), axis=1)
        gt_end_bboxs = np.stack((gt_xmaxs - gt_len_small / 2, gt_xmaxs + gt_len_small / 2), axis=1)

        # calculate the ioa for all timestamp
        match_score_start = []
        for jdx in range(len(anchor_xmin)):
            match_score_start.append(np.max(
                ioa_with_anchors(anchor_xmin[jdx], anchor_xmax[jdx], gt_start_bboxs[:, 0], gt_start_bboxs[:, 1])))
        match_score_end = []
        for jdx in range(len(anchor_xmin)):
            match_score_end.append(np.max(
                ioa_with_anchors(anchor_xmin[jdx], anchor_xmax[jdx], gt_end_bboxs[:, 0], gt_end_bboxs[:, 1])))
        match_score_start = torch.Tensor(match_score_start)
        match_score_end = torch.Tensor(match_score_end)

        return match_score_start, match_score_end, gt_iou_map

    def __len__(self):
        return len(self.video_list)

    def _get_data(self):
        if 'train' in self.subset:
            anno_df = pd.read_csv(self.video_info_path+'val_Annotation.csv')
        elif 'val' in self.subset:
            anno_df = pd.read_csv(self.video_info_path+'test_Annotation.csv')

        video_name_list = sorted(list(set(anno_df.video.values[:])))

        video_info_dir = '/'.join(self.video_info_path.split('/')[:-1])
        saved_data_path = os.path.join(video_info_dir, 'saved.%s.%s.nf%d.sf%d.num%d.%s.pkl' % (
            self.feat_dim, self.subset, self.num_videoframes, self.skip_videoframes,
            len(video_name_list), self.mode)
                                       )
        print(saved_data_path)
        if not self.cfg['override'] and os.path.exists(saved_data_path):
            print('Got saved data.')
            with open(saved_data_path, 'rb') as f:
                self.data, self.durations = pickle.load(f)
            print('Size of data: ', len(self.data['video_names']), flush=True)
            return

        if self.feature_path:
            list_data = []

        list_anchor_xmins = []
        list_anchor_xmaxs = []
        list_gt_bbox = []
        list_videos = []
        list_indices = []

        num_videoframes = self.num_videoframes
        skip_videoframes = self.skip_videoframes
        start_snippet = int((skip_videoframes + 1) / 2)
        stride = int(num_videoframes / 2)

        self.durations = {}

        self.flow_val = h5py.File(self.feature_path+'/flow_val.h5', 'r')
        self.rgb_val = h5py.File(self.feature_path+'/rgb_val.h5', 'r')
        self.flow_test = h5py.File(self.feature_path+'/flow_test.h5', 'r')
        self.rgb_test = h5py.File(self.feature_path+'/rgb_test.h5', 'r')

        for num_video, video_name in enumerate(video_name_list):
            print('Getting video %d / %d' % (num_video, len(video_name_list)), flush=True)
            anno_df_video = anno_df[anno_df.video == video_name]
            if self.mode == 'train':
                gt_xmins = anno_df_video.startFrame.values[:]
                gt_xmaxs = anno_df_video.endFrame.values[:]

            if 'val' in video_name:
                feature_h5s = [
                    self.flow_val[video_name][::self.skip_videoframes,...],
                    self.rgb_val[video_name][::self.skip_videoframes,...]
                ]
            elif 'test' in video_name:
                feature_h5s = [
                    self.flow_test[video_name][::self.skip_videoframes,...],
                    self.rgb_test[video_name][::self.skip_videoframes,...]
                ]
            num_snippet = min([h5.shape[0] for h5 in feature_h5s])
            df_data = np.concatenate([h5[:num_snippet, :]
                                      for h5 in feature_h5s],
                                     axis=1)

            # df_snippet = [start_snippet + skip_videoframes * i for i in range(num_snippet)] 
            df_snippet = [skip_videoframes * i for i in range(num_snippet)] 
            num_windows = int((num_snippet + stride - num_videoframes) / stride)
            windows_start = [i * stride for i in range(num_windows)]
            if num_snippet < num_videoframes:
                windows_start = [0]
                # Add on a bunch of zero data if there aren't enough windows.
                tmp_data = np.zeros((num_videoframes - num_snippet, self.feat_dim))
                df_data = np.concatenate((df_data, tmp_data), axis=0)
                df_snippet.extend([
                    df_snippet[-1] + skip_videoframes * (i + 1)
                    for i in range(num_videoframes - num_snippet)
                ])
            elif num_snippet - windows_start[-1] - num_videoframes > int(num_videoframes / skip_videoframes):
                windows_start.append(num_snippet - num_videoframes)

            for start in windows_start:
                tmp_data = df_data[start:start + num_videoframes, :]

                tmp_snippets = np.array(df_snippet[start:start + num_videoframes])
                if self.mode == 'train':
                    tmp_anchor_xmins = tmp_snippets - skip_videoframes / 2.
                    tmp_anchor_xmaxs = tmp_snippets + skip_videoframes / 2.
                    tmp_gt_bbox = []
                    tmp_ioa_list = []
                    for idx in range(len(gt_xmins)):
                        tmp_ioa = ioa_with_anchors(gt_xmins[idx], gt_xmaxs[idx],
                                                   tmp_anchor_xmins[0],
                                                   tmp_anchor_xmaxs[-1])
                        tmp_ioa_list.append(tmp_ioa)
                        if tmp_ioa > 0:
                            tmp_gt_bbox.append([gt_xmins[idx], gt_xmaxs[idx]])

                    if len(tmp_gt_bbox) > 0 and max(tmp_ioa_list) > 0.9:
                        list_gt_bbox.append(tmp_gt_bbox)
                        list_anchor_xmins.append(tmp_anchor_xmins)
                        list_anchor_xmaxs.append(tmp_anchor_xmaxs)
                        list_videos.append(video_name)
                        list_indices.append(tmp_snippets)
                        if self.feature_dirs:
                            list_data.append(np.array(tmp_data).astype(np.float32))
                elif "infer" in self.mode:
                    list_videos.append(video_name)
                    list_indices.append(tmp_snippets)
                    list_data.append(np.array(tmp_data).astype(np.float32))

        print("List of videos: ", len(set(list_videos)), flush=True)
        self.data = {
            'video_names': list_videos,
            'indices': list_indices
        }
        if self.mode == 'train':
            self.data.update({
                'gt_bbox': list_gt_bbox,
                'anchor_xmins': list_anchor_xmins,
                'anchor_xmaxs': list_anchor_xmaxs,
            })
        if self.feature_dirs:
            self.data['video_data'] = list_data
        print('Size of data: ', len(self.data['video_names']), flush=True)
        with open(saved_data_path, 'wb') as f:
            pickle.dump([self.data, self.durations], f)
        print('Dumped data...')


if __name__ == '__main__':
    import opts
    opt = opts.parse_opt()
    opt = vars(opt)
    train_loader = torch.utils.data.DataLoader(VideoDataSet(opt, subset="train"),
                                               batch_size=opt["batch_size"], shuffle=True,
                                               num_workers=8, pin_memory=True)
    for a, b, c, d in train_loader:
        print(a.shape,b.shape,c.shape,d.shape)
        break