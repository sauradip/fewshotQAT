# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import json
import torch.utils.data as data
import torch
import h5py
from torch.functional import F
import math
import random

base_class = ['Fun sliding down', ' Beer pong', ' Getting a piercing', ' Shoveling snow', ' Kneeling', ' Tumbling', ' Playing water polo', ' Washing dishes', ' Blowing leaves', ' Playing congas', ' Making a lemonade', ' Playing kickball', ' Removing ice from car', ' Playing racquetball', ' Swimming', ' Playing bagpipes', ' Painting', ' Assembling bicycle', ' Playing violin', ' Surfing', ' Making a sandwich', ' Welding', ' Hopscotch', ' Gargling mouthwash', ' Baking cookies', ' Braiding hair', ' Capoeira', ' Slacklining', ' Plastering', ' Changing car wheel', ' Chopping wood', ' Removing curlers', ' Horseback riding', ' Smoking hookah', ' Doing a powerbomb', ' Playing ten pins', ' Getting a haircut', ' Playing beach volleyball', ' Making a cake', ' Clean and jerk', ' Trimming branches or hedges', ' Drum corps', ' Windsurfing', ' Kite flying', ' Using parallel bars', ' Doing kickboxing', ' Cleaning shoes', ' Playing field hockey', ' Playing squash', ' Rollerblading', ' Playing drums', ' Playing rubik cube', ' Sharpening knives', ' Zumba', ' Raking leaves', ' Bathing dog', ' Tug of war', ' Ping-pong', ' Using the balance beam', ' Playing lacrosse', ' Scuba diving', ' Preparing pasta', ' Brushing teeth', ' Playing badminton', ' Mixing drinks', ' Discus throw', ' Playing ice hockey', ' Doing crunches', ' Wrapping presents', ' Hand washing clothes', ' Rock climbing', ' Cutting the grass', ' Wakeboarding', ' Futsal', ' Playing piano', ' Baton twirling', ' Mooping floor', ' Triple jump', ' Longboarding', ' Polishing shoes', ' Doing motocross', ' Arm wrestling', ' Doing fencing', ' Hammer throw', ' Shot put', ' Playing pool', ' Blow-drying hair', ' Cricket', ' Spinning', ' Running a marathon', ' Table soccer', ' Playing flauta', ' Ice fishing', ' Tai chi', ' Archery', ' Shaving', ' Using the monkey bar', ' Layup drill in basketball', ' Spread mulch', ' Skateboarding', ' Canoeing', ' Mowing the lawn', ' Beach soccer', ' Hanging wallpaper', ' Tango', ' Disc dog', ' Powerbocking', ' Getting a tattoo', ' Doing nails', ' Snowboarding', ' Putting on shoes', ' Clipping cat claws', ' Snow tubing', ' River tubing', ' Putting on makeup', ' Decorating the Christmas tree', ' Fixing bicycle', ' Hitting a pinata', ' High jump', ' Doing karate', ' Kayaking', ' Grooming dog', ' Bungee jumping', ' Washing hands', ' Painting fence', ' Doing step aerobics', ' Installing carpet', ' Playing saxophone', ' Long jump', ' Javelin throw', ' Playing accordion', ' Smoking a cigarette', ' Belly dance', ' Playing polo', ' Throwing darts', ' Roof shingle removal', ' Tennis serve with ball bouncing', ' Skiing', ' Peeling potatoes', ' Elliptical trainer', ' Building sandcastles', ' Drinking beer', ' Rock-paper-scissors', ' Using the pommel horse', ' Croquet', ' Laying tile', ' Cleaning windows', ' Fixing the roof', ' Springboard diving', ' Waterskiing', ' Using uneven bars', ' Having an ice cream', ' Sailing', ' Washing face', ' Knitting', ' Bullfighting', ' Applying sunscreen', ' Painting furniture', ' Grooming horse', ' Carving jack-o-lanterns']
val_class = ['Swinging at the playground', ' Dodgeball', ' Ballet', ' Playing harmonica', ' Paintball', ' Cumbia', ' Rafting', ' Hula hoop', ' Cheerleading', ' Vacuuming floor', ' Playing blackjack', ' Waxing skis', ' Curling', ' Using the rowing machine', ' Ironing clothes', ' Playing guitarra', ' Sumo', ' Putting in contact lenses', ' Brushing hair', ' Volleyball']
test_class = ['Hurling', 'Polishing forniture', 'BMX', 'Riding bumper cars', 'Starting a campfire', 'Walking the dog', 'Preparing salad', 'Plataform diving', 'Breakdancing', 'Camel ride', 'Hand car wash', 'Making an omelette', 'Shuffleboard', 'Calf roping', 'Shaving legs', 'Snatch', 'Cleaning sink', 'Rope skipping', 'Drinking coffee', 'Pole vault']




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
        self.shot = 1 # to do
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
        # if self.feature_path[-1]=='/':  # h5 files are in seperated files
        #     if ',' in self.feature_path: # concatenation of two
        #         feat = []
        #         for feature_path in self.feature_path.split(','):
        #             with h5py.File(feature_path+video_name+'.h5', 'r') as f:
        #                 feat.append(f[video_name][:])
        #         feat = np.concatenate(feat,axis=1)
        #     else:
        #         # print(video_name, 'not found!!!!!!!!!!!')
        #         # feat = torch.randn((100,512))
        #         with h5py.File(self.feature_path+video_name+'.h5', 'r') as f:
        #             feat = f[video_name][:]
        # else:
        #     with h5py.File(self.feature_path, 'r') as features_h5:
        #         feat = features_h5[video_name][()]
        with h5py.File("/media/phd/SAURADIP5TB/dataset/C3D/sub_activitynet_v1-3.c3d.hdf5", 'r') as features_h5:
            feat = features_h5[video_name]["c3d_features"][:]
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
                    with h5py.File("/media/phd/SAURADIP5TB/dataset/C3D/sub_activitynet_v1-3.c3d.hdf5", 'r') as features_h5:
                        feat = features_h5[video_name]["c3d_features"][:]
                feat = np.concatenate(feat,axis=1)
            else:
                # print(video_name, 'not found!!!!!!!!!!!')
                # feat = torch.randn((100,512))
                with h5py.File("/media/phd/SAURADIP5TB/dataset/C3D/sub_activitynet_v1-3.c3d.hdf5", 'r') as features_h5:
                    feat = features_h5[video_name]["c3d_features"][:]
        else:
            with h5py.File("/media/phd/SAURADIP5TB/dataset/C3D/sub_activitynet_v1-3.c3d.hdf5", 'r') as features_h5:
                feat = features_h5[video_name]["c3d_features"][:]
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
                if cn in test_class and (corr_end - corr_start) > 0.4: 
                    
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
        # print(subcls_list)
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


        spprt_vids = torch.cat(support_image_list,0)
        spprt_labels = torch.cat(support_label_list,0)

        ## query_data --> [batch,feat_dim,temp], query_label --> [batch,100,1], spprt_vids --> [batch,nshot*feat_dim, temp]
        ## spprt_labels --> [batch,nshot*temp,1]
        return query_data, query_label , spprt_vids , spprt_labels , subcls_list , rand_vid_name







class VideoDataSet(data.Dataset):
    def __init__(self, opt, subset="train", mode="train"):
        self.temporal_scale = opt["temporal_scale"]  # 100
        self.temporal_gap = 1. / self.temporal_scale
        self.subset = subset
        self.mode = mode
        self.feature_path = opt["feature_path"]
        self.video_info_path = opt["video_info"]
        self.video_anno_path = opt["video_anno"]
        self._getDatasetDict()
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
                if len(video_info["annotations"]) > 0:
                    labels = video_info["annotations"][0]["label"]
                    if labels in base_class:
                        self.video_list = list(self.video_dict.keys())
        print("%s subset video numbers: %d" % (self.subset, len(self.video_list)))

    def __getitem__(self, index):
        video_data = self._load_file(index)
        if self.mode == "train":
            match_score_start, match_score_end, confidence_score = self._get_train_label(index, self.anchor_xmin,
                                                                                         self.anchor_xmax)
            return video_data,confidence_score, match_score_start, match_score_end
        else:
            return index, video_data

    def _get_match_map(self):
        match_map = []
        for idx in range(self.temporal_scale):
            tmp_match_window = []
            xmin = self.temporal_gap * idx
            for jdx in range(1, self.temporal_scale + 1):
                xmax = xmin + self.temporal_gap * jdx
                tmp_match_window.append([xmin, xmax])
            match_map.append(tmp_match_window)
        match_map = np.array(match_map)  # 100x100x2
        match_map = np.transpose(match_map, [1, 0, 2])  # [0,1] [1,2] [2,3].....[99,100]
        match_map = np.reshape(match_map, [-1, 2])  # [0,2] [1,3] [2,4].....[99,101]   # duration x start
        self.match_map = match_map  # duration is same in row, start is same in col
        self.anchor_xmin = [self.temporal_gap * (i-0.5) for i in range(self.temporal_scale)]
        self.anchor_xmax = [self.temporal_gap * (i+0.5) for i in range(1, self.temporal_scale + 1)]

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

    def _get_train_label(self, index, anchor_xmin, anchor_xmax):
        video_name = self.video_list[index]
        video_info = self.video_dict[video_name]
        video_frame = video_info['duration_frame']
        video_second = video_info['duration_second']
        feature_frame = video_info['feature_frame']
        corrected_second = float(feature_frame) / video_frame * video_second  # there are some frames not used
        video_labels = video_info['annotations']  # the measurement is second, not frame

        ##############################################################################################
        # change the measurement from second to percentage
        gt_bbox = []
        gt_iou_map = []
        for j in range(len(video_labels)):
            tmp_info = video_labels[j]
            tmp_start = max(min(1, tmp_info['segment'][0] / corrected_second), 0)
            tmp_end = max(min(1, tmp_info['segment'][1] / corrected_second), 0)
            gt_bbox.append([tmp_start, tmp_end])
            tmp_gt_iou_map = iou_with_anchors(
                self.match_map[:, 0], self.match_map[:, 1], tmp_start, tmp_end)
            tmp_gt_iou_map = np.reshape(tmp_gt_iou_map,
                                        [self.temporal_scale, self.temporal_scale])
            gt_iou_map.append(tmp_gt_iou_map)
        gt_iou_map = np.array(gt_iou_map)
        gt_iou_map = np.max(gt_iou_map, axis=0)
        gt_iou_map = torch.Tensor(gt_iou_map)
        ##############################################################################################

        ####################################################################################################
        # generate R_s and R_e
        gt_bbox = np.array(gt_bbox)
        gt_xmins = gt_bbox[:, 0]
        gt_xmaxs = gt_bbox[:, 1]
        gt_lens = gt_xmaxs - gt_xmins
        gt_len_small = 3 * self.temporal_gap  # np.maximum(self.temporal_gap, self.boundary_ratio * gt_lens)
        gt_start_bboxs = np.stack((gt_xmins - gt_len_small / 2, gt_xmins + gt_len_small / 2), axis=1)
        gt_end_bboxs = np.stack((gt_xmaxs - gt_len_small / 2, gt_xmaxs + gt_len_small / 2), axis=1)
        #####################################################################################################

        ##########################################################################################################
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
        ############################################################################################################

        return match_score_start, match_score_end, gt_iou_map

    def __len__(self):
        return len(self.video_list)


if __name__ == '__main__':
    import opts

    opt = opts.parse_opt()
    opt = vars(opt)
    train_loader = torch.utils.data.DataLoader(VideoEpisodicDataSet(opt, subset="train"),
                                               batch_size=opt["batch_size"], shuffle=True,
                                               num_workers=8, pin_memory=True)
    for a,b,c,d, e in train_loader:
        print(a.shape,b.shape,c.shape,d.shape, len(e))
        break
