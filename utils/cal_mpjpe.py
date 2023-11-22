import torch


def apply_umeyama(batch_gt, batch_pred, rotation=True, scaling=True):
    # pred and gt size (batch_size, num_joints, 3)
    pred_centered = batch_pred - batch_pred.mean(axis=1, keepdim=True)
    gt_centered = batch_gt - batch_gt.mean(axis=1, keepdim=True)

    H = pred_centered.transpose(1, 2).matmul(gt_centered)
    u, _, v = torch.svd(H)  # Kabsch algorithm
    R = v.matmul(u.transpose(1, 2))

    if scaling:
        c = (gt_centered.norm(p='fro', dim=2) / pred_centered.norm(p='fro', dim=2)).mean(axis=1,
                                                                                         keepdim=True).unsqueeze(-1)
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
        pred (torch.Tensor): Predicted 3D human pose tensor of shape (N, C, J, 3).
        gt (torch.Tensor): Ground truth 3D human pose tensor of shape (N, C, J, 3).

    Returns:
        float: Mean per joint position error (MPJPE).
    """
    return torch.sqrt(((pred - gt) ** 2).sum(dim=2)).mean()


def test_mse(pred, gt):
    lossf = torch.nn.MSELoss()
    loss = lossf(pred, gt)
    return loss.item()


if __name__ == '__main__':
    # pred1 = torch.tensor([[[0, 0, 0], [1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4], [5, 5, 5]]], dtype=torch.float32)
    # truth1 = torch.tensor([[[1, 0, 0], [1, 2, 2], [2, 4, 2]], [[4, 3, 3], [4, 5, 5], [5, 6, 7]]], dtype=torch.float32)
    # print(pred1.shape, truth1.shape)
    # pred1 = torch.unsqueeze(pred1, dim=1)
    # truth1 = torch.unsqueeze(truth1, dim=1)
    # print(pred1.shape, truth1.shape)
    # print(mpjpe(pred1, truth1))

    pred1 = torch.tensor([[[1, 1, 3], [3, 4, 5]], [[2, 3, 3], [4, 4, 5]]], dtype=torch.float32)
    truth1 = torch.tensor([[[2, 2, 2], [3, 3, 3]], [[3, 3, 3], [4, 4, 4]]], dtype=torch.float32)
    print(test_mse(pred1, truth1))

    print(mpjpe(pred1, truth1))