import numpy as np
import torch
import cv2
from matplotlib import pyplot as plt


def scale_min_max(x): return (x - x.min())/(np.ptp(x))

def visualize_grad_cam(attr, x, img_size=512):
    def get_threshold_by_median(attr, direction='>'):
        if direction == '<':
            neg_1 = attr[np.where(attr<=0)]
            median_1 = np.median(neg_1)
            neg_2 = neg_1[neg_1<=np.median(neg_1)]
            median_2 = np.median(neg_2)
            return median_2
        neg_1 = attr[np.where(attr>0)]
        median_1 = np.median(neg_1)
        neg_2 = neg_1[neg_1>np.median(neg_1)]
        median_2 = np.median(neg_2)
        return median_2
        attr = attr - attr.mean() # zero centering

    attr = torch.squeeze(attr).detach().cpu().numpy()
    attr = cv2.resize(attr, (img_size,img_size))

    neg = attr <= get_threshold_by_median(attr, '<')
    pos = attr > get_threshold_by_median(attr, '>')

    from skimage.segmentation import mark_boundaries
    neg = mark_boundaries(scale_min_max(x[0]), neg, color=(0.3,0.2,0.8), mode='thick')
    pos = mark_boundaries(scale_min_max(x[0]), pos, color=(0.3,1,0), mode='thick')
    
    fig, axes = plt.subplots(1,2, figsize=(20, 10))

    axes[0].imshow(neg)
    axes[1].imshow(pos)

    axes[0].set_title('Negative Attribution')
    axes[1].set_title('Positive Attribution')
    
    plt.show()


from matplotlib import pyplot as plt
import cv2
import pandas as pd
import numpy as np
import seaborn as sns
import sys
sys.path.append('..')
from TedAI.tedai import report_distribution
def visualize_label_distributions(dist_df, test_df, preds_on_test, train_df, dist_func, topk=None, outlier_removal_func=None, class_weights=None):
    c1,c2,c3,c4,c5 = sns.color_palette(n_colors=5)

    topk = len(dist_df) if topk is None else min(len(dist_df), topk)
    c = dist_df.sort_values(dist_func)[1:topk]

    inliers, outliers = outlier_removal_func(c[dist_func], return_outliers=True)
    c_copy = c[c.index.isin(inliers)]
    
    fig, axes = plt.subplots(1, 4, figsize=(30, 7), gridspec_kw={'width_ratios': [2, 3, 1, 3]})
    n = c.iloc[0].Error_Refference_Images
    if len(c_copy) == 0:
        print(f'Image name: {n}')

        axes[0].set_title('No neighbors found.')
        axes[0].imshow(cv2.imread('/raid/huytd16/vinmec_512/' + n))
        axes[0].grid(False)
        plt.show()
        return 
        
    
    
    ref_label = test_df[test_df['Image_model_error']==n].iloc[0][list(train_df.columns[1:])]
    ref_label = [k for k,v in ref_label.to_dict().items() if v==1]
    ref_label = ','.join(c for c in ref_label) or 'Other Findings'
    plot_title = ref_label
    
    nb_df = train_df[train_df.Images.isin(c_copy['Images'])]
    nd = nb_df['No Disease'].sum()

    if c_copy.iloc[0].ND_labels_of_Refference_Images:
        nb_ref = 0
        ref_label = 'Has Disease'
        otr_label = ref_label
    else:
        otr_label = 'Others'
        if ref_label == 'Other Findings':
            nb_ref = 0
            ref_label = ' '
        else:
            nb_ref = nb_df[ref_label.split(',')].sum(axis=1).astype(bool).sum()
    nb_otr = len(nb_df) - (nd + nb_ref)
    counts_df = pd.DataFrame({ 'No Disease': nd, ref_label: nb_ref, otr_label: nb_otr}, index=['counts'])
    
    # visualization
    print(f'Image name: {n}')

    axes[0].set_title(plot_title)
    axes[0].imshow(cv2.imread('/raid/huytd16/vinmec_512/' + n))
    axes[0].grid(False)
    
    
    axes[1].set_title('Neighbors predictions distribution vs Label')
    
    if c_copy.HD_preds_of_Images.var() > 1e-3:
        sns.kdeplot(data=(1 - c_copy.ND_preds_of_Images).values.astype(float), ax=axes[1], color=c1, fill=True, label='NoDiseaseModel pred')
        sns.kdeplot(data=(c_copy.HD_preds_of_Images).values.astype(float), ax=axes[1], color=c3, fill=True, label='HasDiseaseModel pred')
    else:
        axes[1].bar((1 - c_copy.ND_preds_of_Images).mean(), 3, alpha=0.7, color=c1, width=0.03, label='NoDiseaseModel pred')
        axes[1].bar((c_copy.HD_preds_of_Images).mean(), 3, alpha=0.7, color=c3, width=0.03, label='HasDiseaseModel pred')
    axes[1].bar(1 - c_copy.iloc[0].ND_labels_of_Refference_Images, 5, color=c2,width=0.04, label='Has Disease label')
    axes[1].bar(1 - preds_on_test[preds_on_test['Images']==n].preds, 5, color=c5, width=0.015, label='Has Disease pred')
    axes[1].axes.yaxis.set_visible(False)
    axes[1].axes.set_xlim([-0.02, 1.02])
    axes[1].axes.set_ylim([0, 7])
    axes[1].legend()
    
    axes[2].set_title('Neighbors Label proportion')
    counts_df.plot.bar(stacked=True, width=0.1,color=[c1, c3, c4], xlim=[-0.3, 0.3], ax=axes[2], legend=False)
    old_h = 0
    for i,p in enumerate(axes[2].patches):
        height = p.get_height()/2
        axes[2].annotate(counts_df.columns[i].replace(',', '\n'), (p.get_x() + p.get_width() / 2., height + old_h), 
                         ha='left', va='center', xytext=(20, 0), textcoords='offset points')
        old_h = old_h+height*2
    
    if class_weights is None:
        axes[3].set_title('Neighbors Label distribution')
        report_distribution(nb_df, train_df.columns[1:])[['1.00']].plot.bar(ax=axes[3], color=c4, legend=False)
    else:
        axes[3].set_title('Normalized neighbors Label distribution')
        a = report_distribution(nb_df, train_df.columns[1:])
        (class_weights*a['1.00']).plot.bar(ax=axes[3], color=c4, legend=False)

    plt.show()