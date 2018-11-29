import tensorflow as tf

class MultiboxLoss(object):
    """ Multibox loss with some helper functions.

    - Arguments
        num_classes: Number of classes including backgroudn.
        alpha: Weight of L1-smooth loss.
        neg_pos_ratio: Max ratio of negative to positive boxes in loss
        background_label_id: Id of background label.
        negatives_for_hard: Number of negative boxes to consider
            it there is no positive boxes in batch.

    - ToDo
        Add possibility for background label id be not zero.
    """

    def __init__(self, num_classes, alpha=1.0, 
                neg_pos_ratio=3.0,background_label_id=0,
                negatives_for_hard=100.0):

        self.num_classes = num_classes
        self.alpha = alpha
        self.neg_pos_ratio = neg_pos_ratio
        if background_label_id != 0:
            raise Exception("Only 0 as background label id is supported.")
        self.background_label_id = background_label_id
        self.negatives_for_hard = negatives_for_hard


    
    def compute_loss(self,y_true,y_pred):
        """ Compute mulibox loss

        - Arguments
            y_true: Ground truth targets,
                tensor of shape (?,number_boxs)

        """

