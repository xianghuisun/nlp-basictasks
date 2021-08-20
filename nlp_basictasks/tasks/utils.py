import torch

def batch_to_device(batch,target_device):
    """
    send a pytorch batch to a device (CPU/GPU)
    """
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(target_device)
    return batch

def eval_during_training(model, evaluator, output_path,  epoch, steps, callback):
    if evaluator is not None:
        score_and_auc = evaluator(model, output_path=output_path, epoch=epoch, steps=steps)
        if callback is not None:
            callback(score_and_auc, epoch, steps)
        return score_and_auc
    return None
