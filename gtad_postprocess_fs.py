# import sys
# import numpy as np
# import pandas as pd
# import json
# import os
# from joblib import Parallel, delayed

# from gtad_lib import opts



base_class = ['Fun sliding down', ' Beer pong', ' Getting a piercing', ' Shoveling snow', ' Kneeling', ' Tumbling', ' Playing water polo', ' Washing dishes', ' Blowing leaves', ' Playing congas', ' Making a lemonade', ' Playing kickball', ' Removing ice from car', ' Playing racquetball', ' Swimming', ' Playing bagpipes', ' Painting', ' Assembling bicycle', ' Playing violin', ' Surfing', ' Making a sandwich', ' Welding', ' Hopscotch', ' Gargling mouthwash', ' Baking cookies', ' Braiding hair', ' Capoeira', ' Slacklining', ' Plastering', ' Changing car wheel', ' Chopping wood', ' Removing curlers', ' Horseback riding', ' Smoking hookah', ' Doing a powerbomb', ' Playing ten pins', ' Getting a haircut', ' Playing beach volleyball', ' Making a cake', ' Clean and jerk', ' Trimming branches or hedges', ' Drum corps', ' Windsurfing', ' Kite flying', ' Using parallel bars', ' Doing kickboxing', ' Cleaning shoes', ' Playing field hockey', ' Playing squash', ' Rollerblading', ' Playing drums', ' Playing rubik cube', ' Sharpening knives', ' Zumba', ' Raking leaves', ' Bathing dog', ' Tug of war', ' Ping-pong', ' Using the balance beam', ' Playing lacrosse', ' Scuba diving', ' Preparing pasta', ' Brushing teeth', ' Playing badminton', ' Mixing drinks', ' Discus throw', ' Playing ice hockey', ' Doing crunches', ' Wrapping presents', ' Hand washing clothes', ' Rock climbing', ' Cutting the grass', ' Wakeboarding', ' Futsal', ' Playing piano', ' Baton twirling', ' Mooping floor', ' Triple jump', ' Longboarding', ' Polishing shoes', ' Doing motocross', ' Arm wrestling', ' Doing fencing', ' Hammer throw', ' Shot put', ' Playing pool', ' Blow-drying hair', ' Cricket', ' Spinning', ' Running a marathon', ' Table soccer', ' Playing flauta', ' Ice fishing', ' Tai chi', ' Archery', ' Shaving', ' Using the monkey bar', ' Layup drill in basketball', ' Spread mulch', ' Skateboarding', ' Canoeing', ' Mowing the lawn', ' Beach soccer', ' Hanging wallpaper', ' Tango', ' Disc dog', ' Powerbocking', ' Getting a tattoo', ' Doing nails', ' Snowboarding', ' Putting on shoes', ' Clipping cat claws', ' Snow tubing', ' River tubing', ' Putting on makeup', ' Decorating the Christmas tree', ' Fixing bicycle', ' Hitting a pinata', ' High jump', ' Doing karate', ' Kayaking', ' Grooming dog', ' Bungee jumping', ' Washing hands', ' Painting fence', ' Doing step aerobics', ' Installing carpet', ' Playing saxophone', ' Long jump', ' Javelin throw', ' Playing accordion', ' Smoking a cigarette', ' Belly dance', ' Playing polo', ' Throwing darts', ' Roof shingle removal', ' Tennis serve with ball bouncing', ' Skiing', ' Peeling potatoes', ' Elliptical trainer', ' Building sandcastles', ' Drinking beer', ' Rock-paper-scissors', ' Using the pommel horse', ' Croquet', ' Laying tile', ' Cleaning windows', ' Fixing the roof', ' Springboard diving', ' Waterskiing', ' Using uneven bars', ' Having an ice cream', ' Sailing', ' Washing face', ' Knitting', ' Bullfighting', ' Applying sunscreen', ' Painting furniture', ' Grooming horse', ' Carving jack-o-lanterns']
val_class = ['Swinging at the playground', ' Dodgeball', ' Ballet', ' Playing harmonica', ' Paintball', ' Cumbia', ' Rafting', ' Hula hoop', ' Cheerleading', ' Vacuuming floor', ' Playing blackjack', ' Waxing skis', ' Curling', ' Using the rowing machine', ' Ironing clothes', ' Playing guitarra', ' Sumo', ' Putting in contact lenses', ' Brushing hair', ' Volleyball']
test_class = ['Hurling', ' Polishing forniture', ' BMX', ' Riding bumper cars', ' Starting a campfire', ' Walking the dog', ' Preparing salad', ' Plataform diving', ' Breakdancing', ' Camel ride', ' Hand car wash', ' Making an omelette', ' Shuffleboard', ' Calf roping', ' Shaving legs', ' Snatch', ' Cleaning sink', ' Rope skipping', ' Drinking coffee', ' Pole vault']





# def load_json(file):
#     with open(file) as json_file:
#         data = json.load(json_file)
#         return data


# def get_infer_dict(opt):
#     df = pd.read_csv(opt["video_info"])
#     json_data = load_json(opt["video_anno"])
#     database = json_data
#     video_dict = {}
#     for i in range(len(df)):
#         video_name = df.video.values[i]
#         video_info = database[video_name]
#         video_new_info = {}
#         video_new_info['duration_frame'] = video_info['duration_frame']
#         video_new_info['duration_second'] = video_info['duration_second']
#         video_new_info["feature_frame"] = video_info['feature_frame']
#         video_subset = df.subset.values[i]
#         video_new_info['annotations'] = video_info['annotations']
#         if video_subset == 'validation':
#             # video_dict[video_name] = video_new_info
#             if len(video_info["annotations"]) > 0:
#                 labels = video_info["annotations"][0]["label"]
#                 if labels in base_class:
#                     video_dict[video_name] = video_new_info

#     return video_dict


# def Soft_NMS(df, nms_threshold=1e-5, num_prop=100):
#     '''
#     From BSN code
#     :param df:
#     :param nms_threshold:
#     :return:
#     '''
#     df = df.sort_values(by="score", ascending=False)

#     tstart = list(df.xmin.values[:])
#     tend = list(df.xmax.values[:])
#     tscore = list(df.score.values[:])

#     rstart = []
#     rend = []
#     rscore = []

#     # # frost: I use a trick here, remove the detection
#     # # which is longer than 300
#     # for idx in range(0, len(tscore)):
#     #     if tend[idx] - tstart[idx] >= 300:
#     #         tscore[idx] = 0

#     while len(tscore) > 1 and len(rscore) < num_prop and max(tscore)>0:
#         max_index = tscore.index(max(tscore))
#         for idx in range(0, len(tscore)):
#             if idx != max_index:
#                 tmp_iou = IOU(tstart[max_index], tend[max_index], tstart[idx], tend[idx])
#                 if tmp_iou > 0:
#                     tscore[idx] = tscore[idx] * np.exp(-np.square(tmp_iou) / nms_threshold)

#         rstart.append(tstart[max_index])
#         rend.append(tend[max_index])
#         rscore.append(tscore[max_index])
#         tstart.pop(max_index)
#         tend.pop(max_index)
#         tscore.pop(max_index)

#     newDf = pd.DataFrame()
#     newDf['score'] = rscore
#     newDf['xmin'] = rstart
#     newDf['xmax'] = rend
#     return newDf


# def IOU(s1, e1, s2, e2):
#     if (s2 > e1) or (s1 > e2):
#         return 0
#     Aor = max(e1, e2) - min(s1, s2)
#     Aand = min(e1, e2) - max(s1, s2)
#     return float(Aand) / Aor

# def _gen_detection_video(video_name, video_score, video_cls, video_info, opt, num_prop=200, topk = 2):
    
#     score_1 = np.max(video_score)
#     class_1 = video_cls[np.argmax(video_score)]
#     video_score[np.argmax(video_score)] = -1
#     score_2 = np.max(video_score)
#     class_2 = video_cls[np.argmax(video_score)]
#     proposal_list = []
#     # print("name",video_name )
#     if os.path.exists(os.path.join(opt["output"], 'results/' + video_name + ".csv")):
#         df = pd.read_csv(os.path.join(opt["output"], 'results/' + video_name + ".csv"))
#         df['score'] = df.clr_score.values[:] * df.reg_socre.values[:]
#         if len(df) > 1:
#             df = Soft_NMS(df, opt["nms_thr"])

#         df = df.sort_values(by="score", ascending=False)
#         video_duration=float((video_info["duration_frame"]/16)*16)/video_info["duration_frame"]*video_info["duration_second"]
#         # video_duration = video_info["duration_second"]
#         # proposal_list = []

#         for j in range(min(100, len(df))):
#             tmp_proposal = {}
#             tmp_proposal["label"] = str(class_1)
#             tmp_proposal["score"] = float(df.score.values[j])
#             tmp_proposal["segment"] = [max(0, df.xmin.values[j]) * video_duration,
#                                     min(1, df.xmax.values[j]) * video_duration]
#             proposal_list.append(tmp_proposal)
#         for j in range(min(100, len(df))):
#             tmp_proposal = {}
#             tmp_proposal["label"] = str(class_2)
#             tmp_proposal["score"] = float(df.score.values[j])
#             tmp_proposal["segment"] = [max(0, df.xmin.values[j]) * video_duration,
#                                     min(1, df.xmax.values[j]) * video_duration]
#             proposal_list.append(tmp_proposal)

#         # print('The {}-th video {} is finished'.format(idx, video_name))

#     return {video_name: proposal_list}

# def gen_detection_multicore(opt):
#     # get video duration
#     infer_dict = get_infer_dict(opt)

#     # load class name and video level classification
#     cls_data = load_json("data/cuhk_val_simp_share.json")
#     # cls_data = load_json("/home/phd/Desktop/sauradip_research/TAL/A2CL-PT/result_a2clpt.json")
#     vd_list = cls_data["results"]
#     cls_list = list(vd_list.keys())
#     # cls_data_score, cls_data_action = cls_data["results"], cls_data["class"]
#     cls_data_score, cls_data_cls = {}, {}
#     for idx, vid in enumerate(infer_dict.keys()):
#         vid = vid[2:]
#         # if "v_"+ vid in cls_list:
#         cls_data_score[vid] = np.array(cls_data["results"][vid])
#         cls_data_cls[vid] = cls_data["class"] #[np.argmax(cls_data_score[vid])] # find the max class



#     parallel = Parallel(n_jobs=15, prefer="processes")
#     detection = parallel(delayed(_gen_detection_video)(vid, cls_data_score[vid], video_cls, infer_dict["v_"+vid], opt)
#                         for vid, video_cls in cls_data_cls.items())
#     detection_dict = {}
#     [detection_dict.update(d) for d in detection]
#     output_dict = {"version": "ANET v1.3, GTAD", "results": detection_dict, "external_data": {}}

#     with open(opt["output"] + '/detection_result_nms{}.json'.format(opt['nms_thr']), "w") as out:
#         json.dump(output_dict, out)


# if __name__ == "__main__":
#     opt = opts.parse_opt()
#     opt = vars(opt)

#     print("Detection post processing start")
#     gen_detection_multicore(opt)
#     print("Detection Post processing finished")


#     from evaluation.eval_detection import ANETdetection
#     anet_detection = ANETdetection(
#         ground_truth_filename="./evaluation/activity_net_1_3_new.json",
#         prediction_filename=os.path.join(opt['output'], "detection_result_nms{}.json".format(opt['nms_thr'])),
#         subset='validation', verbose=True, check_status=False)
#     anet_detection.evaluate()

#     mAP_at_tIoU = [f'mAP@{t:.2f} {mAP*100:.3f}' for t, mAP in zip(anet_detection.tiou_thresholds, anet_detection.mAP)]
#     results = f'Detection: average-mAP {anet_detection.average_mAP*100:.3f} {" ".join(mAP_at_tIoU)}'
#     print(results)
#     with open(os.path.join(opt['output'], 'results.txt'), 'a') as fobj:
#         fobj.write(f'{results}\n')


import sys
import numpy as np
import pandas as pd
import json
import os
from joblib import Parallel, delayed

from gtad_lib import opts

def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data

vid_cls_dict = {}

def get_infer_dict(opt):
    df = pd.read_csv(opt["video_info"])
    json_data = load_json(opt["video_anno"])
    cls_data = load_json("/home/phd/Desktop/sauradip_research/TAL/A2CL-PT/output/result_a2clpt_lbl_4418.json")
    database = json_data
    video_dict = {}
    for i in range(len(df)):
        video_name = df.video.values[i]
        video_info = database[video_name]
        video_new_info = {}
        video_new_info['duration_frame'] = video_info['duration_frame']
        video_new_info['duration_second'] = video_info['duration_second']
        video_new_info["feature_frame"] = video_info['feature_frame']
        video_subset = df.subset.values[i]
        video_new_info['annotations'] = video_info['annotations']
        if video_subset == 'validation':
            if os.path.exists(os.path.join(opt["output"], 'results2/' + video_name + ".csv")):
                video_dict[video_name] = video_new_info
                vid_cls_dict[video_name] = video_info['annotations'][0]["label"]
    return video_dict


def Soft_NMS(df, nms_threshold=1e-5, num_prop=100):
    '''
    From BSN code
    :param df:
    :param nms_threshold:
    :return:
    '''
    df = df.sort_values(by="score", ascending=False)

    tstart = list(df.xmin.values[:])
    tend = list(df.xmax.values[:])
    tscore = list(df.score.values[:])

    rstart = []
    rend = []
    rscore = []

    # # frost: I use a trick here, remove the detection
    # # which is longer than 300
    # for idx in range(0, len(tscore)):
    #     if tend[idx] - tstart[idx] >= 300:
    #         tscore[idx] = 0

    while len(tscore) > 1 and len(rscore) < num_prop and max(tscore)>0:
        max_index = tscore.index(max(tscore))
        for idx in range(0, len(tscore)):
            if idx != max_index:
                tmp_iou = IOU(tstart[max_index], tend[max_index], tstart[idx], tend[idx])
                if tmp_iou > 0:
                    tscore[idx] = tscore[idx] * np.exp(-np.square(tmp_iou) / nms_threshold)

        rstart.append(tstart[max_index])
        rend.append(tend[max_index])
        rscore.append(tscore[max_index])
        tstart.pop(max_index)
        tend.pop(max_index)
        tscore.pop(max_index)

    newDf = pd.DataFrame()
    newDf['score'] = rscore
    newDf['xmin'] = rstart
    newDf['xmax'] = rend
    return newDf


def IOU(s1, e1, s2, e2):
    if (s2 > e1) or (s1 > e2):
        return 0
    Aor = max(e1, e2) - min(s1, s2)
    Aand = min(e1, e2) - max(s1, s2)
    return float(Aand) / Aor

# def _gen_detection_video(video_name, video_score, video_cls, video_info, opt, num_prop=200, topk = 2):
    
#     score_1 = np.max(video_score)
#     class_1 = video_cls
#     video_score[np.argmax(video_score)] = -1
#     score_2 = np.max(video_score)
#     class_2 = video_cls
#     proposal_list = []
#     if os.path.exists(os.path.join(opt["output"], 'results/' + video_name + ".csv")):
#         df = pd.read_csv(os.path.join(opt["output"], 'results' + "/" + video_name + ".csv"))
#         df['score'] = df.clr_score.values[:] * df.reg_socre.values[:]
#         if len(df) > 1:
#             df = Soft_NMS(df, opt["nms_thr"])

#         df = df.sort_values(by="score", ascending=False)
#         video_duration=float((video_info["duration_frame"]/16)*16)/video_info["duration_frame"]*video_info["duration_second"]
#         # video_duration = video_info["duration_second"]
#         proposal_list = []

#         for j in range(min(100, len(df))):
#             tmp_proposal = {}
#             tmp_proposal["label"] = str(class_1)
#             tmp_proposal["score"] = float(df.score.values[j] * score_1)
#             tmp_proposal["segment"] = [max(0, df.xmin.values[j]) * video_duration,
#                                     min(1, df.xmax.values[j]) * video_duration]
#             proposal_list.append(tmp_proposal)
#         for j in range(min(100, len(df))):
#             tmp_proposal = {}
#             tmp_proposal["label"] = str(class_2)
#             tmp_proposal["score"] = float(df.score.values[j] * score_2)
#             tmp_proposal["segment"] = [max(0, df.xmin.values[j]) * video_duration,
#                                     min(1, df.xmax.values[j]) * video_duration]
#             proposal_list.append(tmp_proposal)

#     # print('The {}-th video {} is finished'.format(idx, video_name))

#     return {video_name: proposal_list}

# def gen_detection_multicore(opt):
#     # get video duration
#     infer_dict = get_infer_dict(opt)

#     # load class name and video level classification
#     cls_data_ = load_json("data/cuhk_val_simp_share.json")
#     cls_data = load_json("/home/phd/Desktop/sauradip_research/TAL/A2CL-PT/output/result_a2clpt_lbl_4418.json")
#     # print(len(list(cls_data["results"].keys()))) ## 4419
#     # print(len(list(infer_dict.keys()))) ## 4728
#     # cls_data_score, cls_data_action = cls_data["results"], cls_data["class"]
#     cls_data_score, cls_data_cls = {}, {}
#     cnt=0
#     for idx, vid in enumerate(infer_dict.keys()):
#         vid = vid
#         cls_data_score[vid] = np.array(cls_data_["results"][vid[2:]])
#         cls_data_cls[vid] = cls_data[vid] #[np.argmax(cls_data_score[vid])] # find the max class
#     #     else:
#     #         cnt+=1
#     # print("No of Videos not found ", str(cnt))



    # parallel = Parallel(n_jobs=15, prefer="processes")
    # # detection = parallel(delayed(_gen_detection_video)(vid, cls_data_score[vid], video_cls, infer_dict['v_'+vid], opt)
    # #                     for vid, video_cls in cls_data_cls.items())
    # detection = parallel(delayed(_gen_detection_video)(vid, cls_data_score[vid], video_cls, infer_dict[vid], opt)
    #                     for vid, video_cls in cls_data_cls.items())
    # detection_dict = {}
    # [detection_dict.update(d) for d in detection]
    # output_dict = {"version": "ANET v1.3, GTAD", "results": detection_dict, "external_data": {}}

    # with open(opt["output"] + '/detection_result_nms{}.json'.format(opt['nms_thr']), "w") as out:
    #     json.dump(output_dict, out)


def _gen_detection_video(video_name, video_score, video_cls, video_info, opt, num_prop=200, topk = 2):
    # cls_data = load_json("/home/phd/Desktop/sauradip_research/TAL/A2CL-PT/output/result_a2clpt_lbl_4418.json")
    score_1 = np.max(video_score)
    class_1 = video_cls[np.argmax(video_score)]
    video_score[np.argmax(video_score)] = -1
    score_2 = np.max(video_score)
    class_2 = video_cls[np.argmax(video_score)]
    if os.path.exists(os.path.join(opt["output"], 'results2/v_' + video_name + ".csv")):
        # print("----------------")
        # lines = 
        df = pd.read_csv(os.path.join(opt["output"], 'results2/v_' + video_name + ".csv"))
        df['score'] = df.clr_score.values[:]
        if len(df) > 1:
            df = Soft_NMS(df, opt["nms_thr"])

        df = df.sort_values(by="score", ascending=False)
        video_duration=float((video_info["duration_frame"]/16)*16)/video_info["duration_frame"]*video_info["duration_second"]
        # video_duration = video_info["duration_second"]
        proposal_list = []
        df1 = pd.read_csv(opt["video_info"])
        json_data = load_json(opt["video_anno"])
        database = json_data
        video_info = database["v_"+video_name]
        # video_new_info['annotations'] = video_info['annotations']
        labels = video_info['annotations'][0]["label"]
        for j in range(min(100, len(df))):
            tmp_proposal = {}
            tmp_proposal["label"] = labels
            tmp_proposal["score"] = float(df.score.values[j]* score_1)
            tmp_proposal["segment"] = [max(0, df.xmin.values[j]) * video_duration,
                                    min(1, df.xmax.values[j]) * video_duration]
            proposal_list.append(tmp_proposal)
        if not opt["multi_instance"] and opt["shot"] == 5:
            print("do-nothing")
        else:
            for j in range(min(100, len(df))):
                tmp_proposal = {}
                tmp_proposal["label"] = labels
                tmp_proposal["score"] = float(df.score.values[j])
                tmp_proposal["segment"] = [max(0, df.xmin.values[j]) * video_duration,
                                        min(1, df.xmax.values[j]) * video_duration]
                proposal_list.append(tmp_proposal)

        # print('The {}-th video {} is finished'.format(idx, video_name))

    return {video_name: proposal_list}

def gen_detection_multicore(opt):
    # get video duration
    infer_dict = get_infer_dict(opt)

    # load class name and video level classification
    cls_data = load_json("data/cuhk_val_simp_share.json")
    # cls_data_score, cls_data_action = cls_data["results"], cls_data["class"]
    cls_data_score, cls_data_cls = {}, {}
    for idx, vid in enumerate(infer_dict.keys()):
        # print(cls_data["class"])
        # if cls_data["class"] in test_class:
        vid = vid[2:]
        cls_data_score[vid] = np.array(cls_data["results"][vid])
        cls_data_cls[vid] = cls_data["class"] #[np.argmax(cls_data_score[vid])] # find the max class



    parallel = Parallel(n_jobs=15, prefer="processes")
    detection = parallel(delayed(_gen_detection_video)(vid, cls_data_score[vid], video_cls, infer_dict['v_'+vid], opt)
                        for vid, video_cls in cls_data_cls.items())
    detection_dict = {}
    [detection_dict.update(d) for d in detection]
    output_dict = {"version": "ANET v1.3, GTAD", "results": detection_dict, "external_data": {}}

    with open(opt["output"] + '/detection_result_nms{}.json'.format(opt['nms_thr']), "w") as out:
        json.dump(output_dict, out)



if __name__ == "__main__":
    opt = opts.parse_opt()
    opt = vars(opt)

    print("Detection post processing start")
    gen_detection_multicore(opt)
    print("Detection Post processing finished")


    from evaluation.eval_detection import ANETdetection
    anet_detection = ANETdetection(
        ground_truth_filename="./evaluation/activity_net_1_3_new.json",
        prediction_filename=os.path.join(opt['output'], "detection_result_nms{}.json".format(opt['nms_thr'])),
        subset='validation', verbose=True, check_status=False)
    anet_detection.evaluate()

    mAP_at_tIoU = [f'mAP@{t:.2f} {mAP*100:.3f}' for t, mAP in zip(anet_detection.tiou_thresholds, anet_detection.mAP)]
    results = f'Detection: average-mAP {anet_detection.average_mAP*100:.3f} {" ".join(mAP_at_tIoU)}'
    print(results)
    with open(os.path.join(opt['output'], 'results.txt'), 'a') as fobj:
        fobj.write(f'{results}\n')

