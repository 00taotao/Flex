import torch.nn as nn
# Knowledge distillation
class Distill:
    def __init__(self, args) -> None:
        self.T = args.T
        pass
    def distillation_loss_zero(self, outputs, labels = None, teacher_outputs = None):
        criterion2 = nn.KLDivLoss(reduction='batchmean')

        Loss = criterion2(nn.functional.log_softmax(outputs / self.T, dim=1),
                           nn.functional.softmax(teacher_outputs / self.T, dim=1)) * (self.T * self.T)
        return  Loss
