
from gtad_lib import opts
import os

opt = opts.parse_opt()
opt = vars(opt)

# print("Detection post processing start")
# gen_detection_multicore(opt)
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

