import numpy as np
import keras
from keras.callbacks import Callback

class LearningRateScheduler(Callback):

    def __init__(self, init_lr=0.01, schedule=[(25, 1e-2), (50, 1e-3), (100, 1e-4)]):
        super(Callback, self).__init__()
        self.init_lr = init_lr
        self.schedule = schedule

    def on_epoch_end(self, epoch, logs={}):
        lr = self.init_lr
        for i in range(0, len(self.schedule) - 1):
            if epoch >= self.schedule[i][0] and epoch < self.schedule[i + 1][0]:
                lr = self.schedule[i][1]

        if epoch >= self.schedule[-1][0]:
            lr = self.schedule[-1][1]

        print('Learning rate:{}'.format(lr))
        keras.backend.set_value(self.model.optimizer.lr, lr)
