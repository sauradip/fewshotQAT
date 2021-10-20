import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from matplotlib import cm

# def viusalize(features1,features2,gt):
#     cmap = cm.get_cmap('tab20')
#     # labels = np.zeros
#     # print(features1.size())
#     # print(features2.size())
#     # print(gt.size())
#     features1 = features1.detach().cpu().numpy()
#     features2 = features2.detach().cpu().numpy()
#     gt = gt.detach().cpu().numpy()
#     gt_bkg = 1-gt
#     labels=np.concatenate((gt,gt_bkg))
#     print(features2)
#     features = np.concatenate((features1,features2),axis=1)
#     tsne = TSNE(n_components=2).fit_transform(features)
#     # print(tsne)
#     # extract x and y coordinates representing the positions of the images on T-SNE plot

#     tx = tsne[:, 0]
#     ty = tsne[:, 1]

#     tx = scale_to_01_range(tx)
#     ty = scale_to_01_range(ty)

#     # print(tx)

#     # initialize a matplotlib plot
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
          
#     # # for every class, we'll add a scatter plot separately
#     # for label in colors_per_class:
#     #     # find the samples of the current class in the data
#     #     indices = [i for i, l in enumerate(labels) if l == label]

#     #     # extract the coordinates of the points of this class only
#     #     current_tx = np.take(tx, indices)
#     #     current_ty = np.take(ty, indices)

#     #     # convert the class color to matplotlib format
#     #     color = np.array(colors_per_class[label], dtype=np.float) / 255

#     #     # add a scatter plot with the corresponding color and label
#     #     ax.scatter(current_tx, current_ty, c=color, label=label)
#     # ax.scatter(tx,ty,c=['red','yellow'], label=['prdicted','gt'])
#     # num_categories = 2
#     # for lab in range(num_categories):
#     #     indices = lab==labels[lab]
#     #     current_tx =  np.take(tx, indices)
#     #     current_ty = np.take(ty, indices)
#     #     ax.scatter(current_tx,current_ty, c=np.array(cmap(lab)).reshape(1,4), label = labels ,alpha=0.5)

#     for lab in range(2):
#         indices = gt == lab
#         # print(tx)
#         current_tx = np.take(tx,indices)
#         print(current_tx)
#         current_ty = np.take(ty,indices)
#         ax.scatter(current_tx,current_ty, c=np.array(cmap(lab)).reshape(1,4),alpha=0.5)
#     # indices = 
#     # current_tx = np.take(tx,indices)
#     # ax.scatter(tx,ty, c=np.array(cmap(gt)), cmap=plt.cm.get_cmap("jet", 10) ,alpha=0.5)
#     # build a legend using the labels we set previously
#     ax.legend(loc='best')

#     # finally, show the plot
#     # plt.show()
#     plt.savefig('tSNE.png')

def viusalize(features1,features2,gt,count,modes):
    cmap = cm.get_cmap('tab20')
    # labels = np.zeros
    print(features1.size())
    print(features2.size())
    print(gt.size()[0])
    len_gt = gt.size()
    features1 = features1.detach().cpu().numpy()
    features2 = features2.detach().cpu().numpy()

    gt = gt.detach().cpu().numpy()
    gt_bkg = 1-gt
    labels=np.concatenate((gt,gt_bkg))
    # print(features2)
    features = np.concatenate((features1,features2))
    tsne = TSNE(n_components=2).fit_transform(features)
    # print(tsne)
    # extract x and y coordinates representing the positions of the images on T-SNE plot

    tx = tsne[:, 0]
    ty = tsne[:, 1]
    new_gt = np.ones(len_gt[0]+1).astype(int)                                                                                                         
    new_gt[0] = 0
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    # print(tx)

    # initialize a matplotlib plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
          
    # # for every class, we'll add a scatter plot separately
    # for label in colors_per_class:
    #     # find the samples of the current class in the data
    #     indices = [i for i, l in enumerate(labels) if l == label]

    #     # extract the coordinates of the points of this class only
    #     current_tx = np.take(tx, indices)
    #     current_ty = np.take(ty, indices)

    #     # convert the class color to matplotlib format
    #     color = np.array(colors_per_class[label], dtype=np.float) / 255

    #     # add a scatter plot with the corresponding color and label
    #     ax.scatter(current_tx, current_ty, c=color, label=label)
    # ax.scatter(tx,ty,c=['red','yellow'], label=['prdicted','gt'])
    # num_categories = 2
    # for lab in range(num_categories):
    #     indices = lab==labels[lab]
    #     current_tx =  np.take(tx, indices)
    #     current_ty = np.take(ty, indices)
    #     ax.scatter(current_tx,current_ty, c=np.array(cmap(lab)).reshape(1,4), label = labels ,alpha=0.5)
    label_colors = ['b','g','r','c','m']
    colors = [label_colors[i+1] for i in new_gt]

    for lab in range(2):
        # print(lab)
        indices = gt == lab
        # print(tx)
        current_tx = np.take(tx,indices)
        # print(current_tx)
        current_ty = np.take(ty,indices)

    # ax.scatter(tx,ty, c=np.array(cmap(gt)).reshape(100,4),alpha=0.5)
    # print(np.shape(np.array(cmap(new_gt[0]))))
    
    # ax.scatter(tsne[0, 0],tsne[0, 1], np.array(cmap(new_gt[0])).reshape(1,4) ,alpha=0.5)
    ax.scatter(tx[0],ty[0], c = colors[0] ,alpha=0.5, marker = 'v', label = "classifier_weight")
    ax.scatter(tx[1:],ty[1:], c = colors[1:] ,alpha=0.5, marker = '*', label = "query sample")
    
    # indices = 
    # current_tx = np.take(tx,indices)
    # ax.scatter(tx,ty, c=np.array(cmap(gt)), cmap=plt.cm.get_cmap("jet", 10) ,alpha=0.5)
    # build a legend using the labels we set previously
    ax.legend(loc='best')

    # finally, show the plot
    # plt.show()
    if modes=="before":
        plt.savefig('/home/phd/Desktop/sauradip_research/TAL/gtad/tsne_plots/before/tSNE_'+str(count)+'.png')
    else:
        plt.savefig('/home/phd/Desktop/sauradip_research/TAL/gtad/tsne_plots/after/tSNE_'+str(count)+'.png')

def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)
    # make the distribution fit [0; 1] by dividing by its range

    return starts_from_zero / value_range

def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)
    # make the distribution fit [0; 1] by dividing by its range

    return starts_from_zero / value_range


