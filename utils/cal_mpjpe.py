import torch

def apply_umeyama(batch_gt, batch_pred, rotation=True, scaling=True):

    # pred and gt size (batch_size, num_joints, 3)
    pred_centered = batch_pred - batch_pred.mean(axis=1, keepdim=True)
    gt_centered = batch_gt - batch_gt.mean(axis=1, keepdim=True)

    H = pred_centered.transpose(1, 2).matmul(gt_centered)
    u, _, v = torch.svd(H)  # Kabsch algorithm
    R = v.matmul(u.transpose(1, 2))

    if scaling:
        c = (gt_centered.norm(p='fro', dim=2) / pred_centered.norm(p='fro', dim=2)).mean(axis=1, keepdim=True).unsqueeze(-1)
    else:
        c = 1.0

    if rotation:
        return batch_pred.matmul(R.transpose(1, 2)) * c
    else:
        return batch_pred * c

def mpjpe(pred, gt):
    """
    Calculate the mean per joint position error (MPJPE) between predicted and ground truth 3D human pose.

    Args:
        pred (torch.Tensor): Predicted 3D human pose tensor of shape (N, J, 3).
        gt (torch.Tensor): Ground truth 3D human pose tensor of shape (N, J, 3).

    Returns:
        float: Mean per joint position error (MPJPE).
    """
    N, J = pred.shape[:2]
    return torch.sqrt(((pred - gt) ** 2).sum(dim=2)).mean()


if __name__ == '__main__':
    pred = torch.randn((1024, 17, 3))
    target = pred * 5
    pred_scaled = apply_umeyama(target, pred)
    print((pred_scaled - target).sum()) # the result should be zero
