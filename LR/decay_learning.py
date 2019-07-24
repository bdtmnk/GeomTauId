import keras
import keras.backend as K


class DecayLR(keras.callbacks.Callback):
    """

    """

    def __init__(self, monitor, lr_min=0.000001, factor=0.2, patience=3):
        """

        :param monitor:
        :param lr_min:
        :param factor:
        :param patience:
        :return:
        """
        super(DecayLR, self).__init__()
        self.monitor = monitor
        self.lr_min = lr_min
        self.factor = factor
        self.patience = patience
        self.wait = 0
        return

    def on_epoch_end(self, epoch, logs=None):
        """

        :return:
        """
        if epoch == 1:
            return
        else:
            lr = K.get_value(self.model.optimizer.lr)
            if lr > self.lr_min:
                if self.monitor[-1] < self.monitor[-2]:
                    self.wait += 1
                else:
                    self.wait = 0
                if self.wait > self.patience:
                    new_lr = max(lr*self.factor, self.lr_min)
                    K.set_value(self.model.optimizer.lr, new_lr)
                    self.wait = 0
            return
