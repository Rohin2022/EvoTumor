import SimpleITK as sitk
import numpy as np
from scipy.ndimage import label
from radiomics import featureextractor
import logging

logging.getLogger("radiomics").setLevel(logging.ERROR)


class RadiomicsMetricsEvaluator:
    def __init__(self, mask_columns, tumor_columns, spacing):
        shape_settings = {'geometryTolerance': 1e-4, 'label': 1}
        self.extractor = featureextractor.RadiomicsFeatureExtractor(**shape_settings)

        self.extractor.enableAllFeatures()


        # binWidth must match the offline extraction (tumor_metrics_full.py: binWidth=25)
        tumor_settings = {'geometryTolerance': 1e-4, 'label': 1, 'binWidth': 25}
        self.tumor_extractor = featureextractor.RadiomicsFeatureExtractor(**tumor_settings)

        self.tumor_extractor.enableAllFeatures()


        self.mask_columns = mask_columns          # e.g. ['original_shape_Elongation', ...]
        self.tumor_columns = tumor_columns    # includes non-pyradiomics fields too

        self.spacing = spacing

    def compute_mask(self, tumor_mask):
        """
        tumor_mask: already-binarized 3D numpy array or torch tensor.
        """

        bbox_columns = ["diameter_x_mm","diameter_y_mm","diameter_z_mm"]

        shape_radiomics = [x for x in self.mask_columns if x not in bbox_columns]

        #self.extractor.enableFeaturesByName(shape=shape_radiomics)


        mask = tumor_mask
        if hasattr(mask, "numpy"):
            mask = mask.cpu().numpy()
        mask = np.squeeze(mask).astype(np.uint8)

        metrics = {col: 0.0 for col in self.mask_columns}

        if not mask.any():
            return metrics

        mask_sitk = sitk.GetImageFromArray(mask)
        mask_sitk.SetSpacing([float(s) for s in self.spacing])

        try:
            features = self.extractor.execute(mask_sitk, mask_sitk)
            for key, value in features.items():
                if key in metrics:
                    try:
                        metrics[key] = float(value)
                    except (TypeError, ValueError):
                        pass
        except Exception as e:
            print(f"PyRadiomics shape extraction failed: {e}")

        return metrics

    def compute_tumor(self, ct, tumor_mask, organ_mask=None):
        """
        ct: raw HU 3D numpy array or torch tensor, no windowing/normalization.
        tumor_mask: already-binarized 3D array, same shape as ct.
        organ_mask: optional binarized healthy-organ mask, needed for attenuation_delta.
        """

        #self.tumor_extractor.enableFeaturesByName(shape=self.tumor_columns)



        if hasattr(ct, "numpy"):
            ct = ct.cpu().numpy()
        if hasattr(tumor_mask, "numpy"):
            tumor_mask = tumor_mask.cpu().numpy()

        ct = np.squeeze(ct).astype(np.float32)
        mask = np.squeeze(tumor_mask).astype(np.uint8)

        metrics = {col: 0.0 for col in self.tumor_columns}

        if not mask.any():
            return metrics

        ct_sitk = sitk.GetImageFromArray(ct)
        ct_sitk.SetSpacing([float(s) for s in self.spacing])

        mask_sitk = sitk.GetImageFromArray(mask)
        mask_sitk.CopyInformation(ct_sitk)

        structure = np.ones((3, 3, 3), dtype=bool)
        _, num_components = label(mask, structure=structure)
        metrics["num_components"] = int(num_components)

        nz = np.argwhere(mask)
        extent = nz.max(axis=0) - nz.min(axis=0) + 1
        metrics["diameter_x_mm"] = float(extent[2] * self.spacing[0])
        metrics["diameter_y_mm"] = float(extent[1] * self.spacing[1])
        metrics["diameter_z_mm"] = float(extent[0] * self.spacing[2])

        try:
            features = self.tumor_extractor.execute(ct_sitk, mask_sitk)
            for key, value in features.items():
                if key in metrics:
                    try:
                        metrics[key] = float(value)
                    except (TypeError, ValueError):
                        pass

            if organ_mask is not None:
                if hasattr(organ_mask, "numpy"):
                    organ_mask = organ_mask.cpu().numpy()
                bin_organ_mask = np.squeeze(organ_mask).astype(bool)
                healthy_voxels = ct[bin_organ_mask & (~mask.astype(bool))]

                if len(healthy_voxels) > 0:
                    mean_organ = np.mean(healthy_voxels)
                    std_organ = np.std(healthy_voxels)
                    std_organ = std_organ if std_organ != 0 else 1e-5
                    mean_tumor = float(features.get('original_firstorder_Mean', 0.0))
                    metrics["attenuation_delta"] = (mean_tumor - mean_organ) / std_organ
            else:
                logging.getLogger("radiomics").warning(
                    "compute_tumor: no organ_mask provided — attenuation_delta left at 0.0"
                )

        except Exception as e:
            print(f"PyRadiomics tumor extraction failed: {e}")

        return metrics