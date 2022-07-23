import os

import numpy as np

import paths
import SimpleITK as sitk
import torch
from scipy import ndimage

from utils.metric_utils import connectivity_region_analysis, eval_dice, eval_average_surface_distances


class ComputeMetricsMsm:
    def __init__(self,val_ids,test_ids,logger):
        self.no_mask_labeled_amount = []
        self.mask_labeled_amount = []
        self.val_ids = val_ids
        self.test_ids = test_ids
        self.predict = None
        self.logger = logger

    def compute(self,val_or_test,filter_sparse_tagging):
        if val_or_test == 'val':
            ids = self.val_ids
            self.no_mask_labeled_amount = []
            self.mask_labeled_amount = []
            remove_label_th = 0
        else:
            assert val_or_test == 'test'
            ids = self.test_ids
            remove_label_th = np.quantile(self.no_mask_labeled_amount,0.55) * 0.9 + np.quantile(self.mask_labeled_amount,0.1) * 0.1
        dice = []
        asd = []

        for ind, id1 in enumerate(ids):
            file_image_path = os.path.join(paths.MSM_DATA_PATH,id1[0])
            file_label_path = os.path.join(paths.MSM_DATA_PATH,id1[1])

            itk_image = sitk.ReadImage(
                file_image_path)
            itk_mask = sitk.ReadImage(file_label_path)
            image = sitk.GetArrayFromImage(itk_image)
            mask = sitk.GetArrayFromImage(itk_mask)
            binary_mask = np.ones(mask.shape)
            mean = np.sum(image * binary_mask) / np.sum(binary_mask)
            std = np.sqrt(np.sum(np.square(image - mean) * binary_mask) / np.sum(binary_mask))
            image = (image - mean) / std
            mask[mask == 2] = 1
            image = image.transpose([1, 2, 0])
            mask = mask.transpose([1, 2, 0])
            end_shape = (384, 384)
            preds = np.zeros(mask.shape)
            frame_list = [kk for kk in range(1, image.shape[2] - 1)]
            for ii in range(image.shape[2]):
                vol = np.zeros([1, 3, end_shape[0], end_shape[1]],dtype=np.float32)

                for idx, jj in enumerate(frame_list[ii: ii + 1]):
                    vol[0, :, :, :] = image[..., jj - 1: jj + 2].transpose([2, 0, 1])
                torch.cuda.empty_cache()
                images = torch.from_numpy(vol)
                outputs = self.predict(images)
                for idx, jj in enumerate(frame_list[ii: (ii + 1)]):
                    current_pred = (outputs > 0.5).squeeze().copy()
                    label_im, nb_labels = ndimage.label(current_pred)
                    most_popular_label_amount = np.max(ndimage.sum(current_pred, label_im, range(nb_labels + 1)))
                    if val_or_test == 'val':
                        if len(np.unique(mask[:,:,jj])) == 1:
                            self.no_mask_labeled_amount.append(most_popular_label_amount)
                        else:
                            self.mask_labeled_amount.append(most_popular_label_amount)
                        if self.mask_labeled_amount and self.no_mask_labeled_amount:
                            remove_label_th = np.quantile(self.no_mask_labeled_amount,0.55) * 0.9 + np.quantile(self.mask_labeled_amount,0.1) * 0.1

                    if filter_sparse_tagging and most_popular_label_amount < remove_label_th:
                        current_pred = np.zeros(current_pred.shape)
                    preds[..., jj] = current_pred
            processed_preds = connectivity_region_analysis(preds)
            dice_subject = eval_dice(mask, processed_preds)
            asd_subject = eval_average_surface_distances(mask, processed_preds)

            dice.append(dice_subject)
            asd.append(asd_subject)

        dice_avg = np.mean(dice, axis=0).tolist()[0]
        asd_avg = np.mean(asd)
        remove_label_th = np.quantile(self.no_mask_labeled_amount,0.55) * 0.5 + np.quantile(self.mask_labeled_amount,0.2) * 0.5

        print("dice_avg %.4f" % (dice_avg))


        return {'dice':float(dice_avg),'asd':float(asd_avg),'remove_label_th':float(remove_label_th)}