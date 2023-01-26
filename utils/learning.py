from tensorflow.keras.callbacks import *
from tensorflow.keras import backend as K
import numpy as np

class CyclicLR(Callback):
    """This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or 
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each 
        cycle iteration.
    For more detail, please see paper.
    
    # Example
        ```python
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    
    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```    
    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore 
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where 
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored 
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on 
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
    """

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1/(2.**(x-1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma**(x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.
        
    def clr(self):
        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))
        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(self.clr_iterations)
        
    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())        
            
    def on_batch_end(self, epoch, logs=None):
        
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        
        K.set_value(self.model.optimizer.lr, self.clr())
        
        
class LearningRateCallback(Callback):
    def __init__(self, lr_max, lr_min, lr_max_compression=5, t0=10, tmult=1, trigger_val_acc=0.0, show_lr=True):
        # Global learning rate max/min
        self.lr_max = lr_max
        self.lr_min = lr_min
        # Max learning rate compression
        self.lr_max_compression = lr_max_compression
        # Warm restarts params
        self.t0 = t0
        self.tmult = tmult
        # Learning rate decay trigger (早い段階で減衰させても訓練が遅くなるだけなので)
        self.trigger_val_acc = trigger_val_acc
        # init parameters
        self.show_lr = show_lr
        self._init_params()        
        
    def _init_params(self):
        # Decay triggered
        self.triggered = False
        # Learning rate of next warm up
        self.lr_warmup_next = self.lr_max
        self.lr_warmup_current = self.lr_max
        # Current learning rate
        self.lr = self.lr_max
        # Current warm restart interval
        self.ti = self.t0
        # Warm restart count
        self.tcur = 1
        # Best validation accuracy
        self.best_val_acc = 0

    def on_train_begin(self, logs):
        self._init_params()

    def on_epoch_end(self, epoch, logs):
        if not self.triggered and logs["val_acc"] >= self.trigger_val_acc:
            self.triggered = True

        if self.triggered:
            # Update next warmup lr when validation acc surpassed
            if logs["val_acc"] > self.best_val_acc:
                self.best_val_acc = logs["val_acc"]
                # Avoid lr_warmup_next too small
                if self.lr_max_compression > 0:
                    self.lr_warmup_next = max(self.lr_warmup_current / self.lr_max_compression, self.lr)
                else:
                    self.lr_warmup_next = self.lr
        if self.show_lr:
            print(f"epoch = {epoch+1}, sgdr_triggered = {self.triggered}, best_val_acc = {self.best_val_acc}, " + 
                  f"current_lr = {self.lr:f}, next_warmup_lr = {self.lr_warmup_next:f}, next_warmup = {self.ti-self.tcur}")

    # SGDR
    def lr_scheduler(self, epoch):
        if not self.triggered: return self.lr
        # SGDR
        self.tcur += 1
        if self.tcur > self.ti:
            self.ti = int(self.tmult * self.ti)
            self.tcur = 1
            self.lr_warmup_current = self.lr_warmup_next
        self.lr = float(self.lr_min + (self.lr_warmup_current - self.lr_min) * (1 + np.cos(self.tcur/self.ti*np.pi)) / 2.0)
        return self.lr
    
def lr_scheduler_warmup(epoch, lr, warmup_epochs=15, decay_epochs=100, initial_lr=1e-6, base_lr=1e-3, min_lr=5e-5):
    if epoch <= warmup_epochs:
        pct = epoch / warmup_epochs
        return ((base_lr - initial_lr) * pct) + initial_lr

    if epoch > warmup_epochs and epoch < warmup_epochs+decay_epochs:
        pct = 1 - ((epoch - warmup_epochs) / decay_epochs)
        return ((base_lr - min_lr) * pct) + min_lr

    return min_lr

