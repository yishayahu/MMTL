from dpipe.dataset.segmentation import SegmentationFromCSV
import numpy as np

class CC359(SegmentationFromCSV):
    def __init__(self, data_path, modalities=('MRI',), target='brain_mask', metadata_rpath='meta.csv'):
        super().__init__(data_path=data_path,
                         modalities=modalities,
                         target=target,
                         metadata_rpath=metadata_rpath)
        self.n_domains = len(self.df['fold'].unique())

    def load_image(self, i):
        return np.float32(super().load_image(i)[0])  # 4D -> 3D

    def load_segm(self, i):
        return np.float32(super().load_segm(i))  # already 3D

    def load_shape(self, i):
        return np.int32(np.shape(self.load_segm(i)))

    def load_spacing(self, i):
        voxel_spacing = np.array([self.df['x'].loc[i], self.df['y'].loc[i], self.df['z'].loc[i]])
        return voxel_spacing

    def load_domain_label(self, i):
        domain_id = self.df['fold'].loc[i]
        return np.eye(self.n_domains)[domain_id]  # one-hot-encoded domain

    def load_domain_label_number(self, i):
        return self.df['fold'].loc[i]

    def load_domain_label_number_binary_setup(self, i, domains):
        """Assigns '1' to the domain of the largest index; '0' to another one
        Domains may be either (index1, index2) or (sample_scan1_id, sample_scan2_id) """

        if type(domains[0]) != int:
            # the fold numbers of the corresponding 2 samples
            doms = (self.load_domain_label_number (domains[0]), self.load_domain_label_number (domains[1]))
        else:
            doms = domains
        largest_domain = max(doms)
        domain_id = self.df['fold'].loc[i]
        if domain_id == largest_domain:
            return 1
        elif domain_id in doms:  # error otherwise
            return 0