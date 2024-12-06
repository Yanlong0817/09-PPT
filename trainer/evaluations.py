import torch
from models.model import Final_Model

def evaluate_trajectory_sdd(dataset, model: Final_Model, config, device):
    # for the fulfillment stage or trajectory stage, we should have a fixed past/intention memory bank.
    ade_48s = fde_48s = 0
    samples = 0
    dict_metrics = {}

    y_true = []
    y_pred = []

    with torch.no_grad():
        for _, (ped, neis, mask, initial_pos, scene) in enumerate(dataset):
            ped, neis, mask, initial_pos = (
                ped.to(device),
                neis.to(device),
                mask.to(device),
                initial_pos.to(device),
            )
            if config.dataset_name == "eth":
                ped[:, :, 0] = ped[:, :, 0] * config.data_scaling[0]
                ped[:, :, 1] = ped[:, :, 1] * config.data_scaling[1]

            traj_norm = ped
            output = model.get_trajectory(traj_norm, neis, mask, scene)
            output = output.data
            # print(output.shape)

            y_true.append(traj_norm)
            y_pred.append(output)
            samples += output.shape[0]

    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)

    convert = torch.load("convert.ckpt")

    y_true = y_true * torch.tensor(convert).to(device).unsqueeze(1).unsqueeze(1)
    y_pred = y_pred * torch.tensor(convert).to(device).unsqueeze(1).unsqueeze(1).unsqueeze(1)

    future_rep = y_true[:, 8:-1, :].unsqueeze(1).repeat(1, config.goal_num, 1, 1)
    future_goal = y_true[:, -1:, :].unsqueeze(1).repeat(1, config.goal_num, 1, 1)
    future = torch.cat((future_rep, future_goal), dim=2)
    distances = torch.norm(y_pred - future, dim=3)

    fde_mean_distances = torch.mean(distances[:, :, -1:], dim=2) # find the tarjectory according to the last frame's distance
    fde_index_min = torch.argmin(fde_mean_distances, dim=1)
    fde_min_distances = distances[torch.arange(0, len(fde_index_min)), fde_index_min]
    fde_48s += torch.sum(fde_min_distances[:, -1])

    ade_mean_distances = torch.mean(distances[:, :, :], dim=2) # find the tarjectory according to the last frame's distance
    ade_index_min = torch.argmin(ade_mean_distances, dim=1)
    ade_min_distances = distances[torch.arange(0, len(ade_index_min)), ade_index_min]
    ade_48s += torch.sum(torch.mean(ade_min_distances, dim=1))

    # future_rep = traj_norm[:, 8:, :].unsqueeze(1).repeat(1, 20, 1, 1)
    # distances = torch.norm(output - future_rep, dim=3)
    # mean_distances = torch.mean(distances[:, :, -1:], dim=2)  # find the tarjectory according to the last frame's distance
    # index_min = torch.argmin(mean_distances, dim=1)
    # min_distances = distances[torch.arange(0, len(index_min)), index_min]
    #
    # fde_48s += torch.sum(min_distances[:, -1])
    # ade_48s += torch.sum(torch.mean(min_distances, dim=1))

    dict_metrics['fde_48s'] = fde_48s / samples
    dict_metrics['ade_48s'] = ade_48s / samples

    return dict_metrics['fde_48s'], dict_metrics['ade_48s']


def evaluate_trajectory(dataset, model: Final_Model, config, device):
    # for the fulfillment stage or trajectory stage, we should have a fixed past/intention memory bank.
    ade_48s = fde_48s = 0
    samples = 0
    dict_metrics = {}

    with torch.no_grad():
        for _, (ped, neis, mask, initial_pos, scene) in enumerate(dataset):
            ped, neis, mask, initial_pos = (
                ped.to(device),
                neis.to(device),
                mask.to(device),
                initial_pos.to(device),
            )
            if config.dataset_name == "eth":
                ped[:, :, 0] = ped[:, :, 0] * config.data_scaling[0]
                ped[:, :, 1] = ped[:, :, 1] * config.data_scaling[1]

            traj_norm = ped
            output = model.get_trajectory(traj_norm, neis, mask, scene)
            output = output.data
            # print(output.shape)

            future_rep = traj_norm[:, 8:-1, :].unsqueeze(1).repeat(1, config.goal_num, 1, 1)
            future_goal = traj_norm[:, -1:, :].unsqueeze(1).repeat(1, config.goal_num, 1, 1)
            future = torch.cat((future_rep, future_goal), dim=2)
            distances = torch.norm(output - future, dim=3)

            fde_mean_distances = torch.mean(distances[:, :, -1:], dim=2) # find the tarjectory according to the last frame's distance
            fde_index_min = torch.argmin(fde_mean_distances, dim=1)
            fde_min_distances = distances[torch.arange(0, len(fde_index_min)), fde_index_min]
            fde_48s += torch.sum(fde_min_distances[:, -1])

            ade_mean_distances = torch.mean(distances[:, :, :], dim=2) # find the tarjectory according to the last frame's distance
            ade_index_min = torch.argmin(ade_mean_distances, dim=1)
            ade_min_distances = distances[torch.arange(0, len(ade_index_min)), ade_index_min]
            ade_48s += torch.sum(torch.mean(ade_min_distances, dim=1))

            # future_rep = traj_norm[:, 8:, :].unsqueeze(1).repeat(1, 20, 1, 1)
            # distances = torch.norm(output - future_rep, dim=3)
            # mean_distances = torch.mean(distances[:, :, -1:], dim=2)  # find the tarjectory according to the last frame's distance
            # index_min = torch.argmin(mean_distances, dim=1)
            # min_distances = distances[torch.arange(0, len(index_min)), index_min]
            #
            # fde_48s += torch.sum(min_distances[:, -1])
            # ade_48s += torch.sum(torch.mean(min_distances, dim=1))
            samples += distances.shape[0]


        dict_metrics['fde_48s'] = fde_48s / samples
        dict_metrics['ade_48s'] = ade_48s / samples

    return dict_metrics['fde_48s'], dict_metrics['ade_48s']
