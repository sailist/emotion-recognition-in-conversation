from torch.optim.optimizer import Optimizer


def make_optim(model, optim_builder, split=False) -> Optimizer:
    if split:
        wd_params, non_wd_params = [], []
        for name, param in model.named_parameters():
            if 'bn' in name:
                non_wd_params.append(param)
            else:
                wd_params.append(param)
        param_list = [
            {'params': wd_params}, {'params': non_wd_params, 'weight_decay': 0}]
        optim = optim_builder.build(param_list)
    else:
        optim = optim_builder.build(model.parameters())
    return optim
