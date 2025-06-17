import torch    
import torch.nn.functional as F

def Make_Optimizer(model):
    if model.num_classes == 1:
        return torch.optim.AdamW(model.parameters(), lr=0.018, weight_decay=1e-4)
    else:
        return torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)

def Make_LR_Scheduler(optimizer):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 50, eta_min = 1e-6)

def Make_Loss_Function(number_of_classes):
    class DiceCELoss:
        def __init__(self, weight=0.6, epsilon=1e-6, mode='multiclass', ignore_index=255):
            self.weight = weight
            self.epsilon = epsilon
            self.mode = mode
            self.ignore_index = ignore_index
        
        def __call__(self, pred, target):
            if self.mode == 'binary':
                pred = pred.squeeze(1)
                target = target.squeeze(1).float()
                intersection = torch.sum(pred * target, dim=(1, 2))
                union = torch.sum(pred, dim=(1, 2)) + torch.sum(target, dim=(1, 2))
                dice = (2 * intersection + self.epsilon) / (union + self.epsilon)
                dice_loss = 1 - dice.mean()
                
                ce_loss = F.binary_cross_entropy(pred, target)
            
            elif self.mode == 'multiclass':
                device = "cuda" if torch.cuda.is_available() else "cpu"
                weight = torch.tensor([0.0004, 0.0671, 0.1366, 0.0526, 0.0796, 0.0603, 0.0305, 0.0310, 0.0180,
                                       0.0551, 0.0495, 0.0458, 0.0250, 0.0411, 0.0366, 0.0090, 0.0717, 0.0578,
                                       0.0341, 0.0323, 0.0661], dtype=torch.float32, device=device)
                batchsize, num_classes, H, W = pred.shape
                target = target.squeeze(1)
                pred_softmax = F.softmax(pred, dim=1)
                target_one_hot = F.one_hot(target, num_classes=num_classes).squeeze(1).permute(0, 3, 1, 2).float()
                intersection = torch.sum(pred_softmax * target_one_hot, dim=(2, 3))
                union = torch.sum(pred_softmax, dim=(2, 3)) + torch.sum(target_one_hot, dim=(2, 3))
                dice = (2 * intersection + self.epsilon) / (union + self.epsilon)
                dice_loss = 1 - dice.mean()
                ce_loss = DiceCELoss.ce2d_deterministic(pred, target, weight=weight)
            else:
                raise ValueError("mode should be 'binary' or 'multiclass'")

            combined_loss = self.weight * dice_loss + (1 - self.weight) * ce_loss
            return combined_loss

        def ce2d_deterministic(logits, target, weight=None, ignore_index=-100, eps=1e-7):
            log_p = torch.log_softmax(logits, dim=1)
            log_p = log_p.gather(1, target.unsqueeze(1))
            log_p = log_p.squeeze(1)
            mask = (target != ignore_index)
            if weight is not None:
                w = weight[target]
                loss = -(w * log_p * mask).sum() / (w * mask).sum()
            else:
                loss = -(log_p * mask).sum() / mask.sum()
            return loss
    
    BINARY_SEG = True if number_of_classes==2 else False
    return DiceCELoss(mode='binary') if BINARY_SEG else DiceCELoss(mode='multiclass') 