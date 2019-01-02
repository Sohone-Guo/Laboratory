import numpy as np
import tensorflow as tf

class BBoxUtility(object):
    """ Utility class to do some stuff with bounding boxes and priors.

    - Arguments
        num_classes: Number of classes including background.
        priors: Priors and variances, numpy tensor of shape (num_priors,8),
            priors[i] = [xmin,ymin,xmax,ymax,varxc,varyc, varw, varh].
        overlap_threshold: Threshold to assign box to a prior.
        nms_thresh: Nms threshold.
        top_k: Number of total bboxes to be kept per image after nms step.
    """