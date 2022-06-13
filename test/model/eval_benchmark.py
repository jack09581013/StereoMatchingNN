from utils.dataset import *
from torch.utils.data import DataLoader, Subset
import numpy as np
from test.model.profile import *
from colorama import Fore, Style
import test.model.profile

max_disparity = 144
version = 500
seed = 0
lr_check = False
max_disparity_diff = 1.5
merge_cost = True
candidate = False

profile = test.model.profile.GDFNet_mdc6f()
dataset = 'KITTI_2015_benchmark'

model = profile.load_model(max_disparity, version)[1]
version, loss_history = profile.load_history(version)

print('Using model:', profile)
print('Using dataset:', dataset)
print('Network image size:', (KITTI_2015_benchmark.HEIGHT, KITTI_2015_benchmark.WIDTH))
print('Number of parameters: {:,}'.format(sum(p.numel() for p in model.parameters())))

losses = []
error = []
confidence_error = []
total_eval = []

test_dataset = KITTI_2015_benchmark()
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
print('Number of testing data:', len(test_dataset))

model.eval()
for batch_index, (X, Y, origin_height, origin_width) in enumerate(test_loader):
    with torch.no_grad():
        tools.tic()

        eval_dict = profile.eval(X, Y, dataset, merge_cost=merge_cost, lr_check=False, candidate=candidate,
                                 regression=True,
                                 penalize=False, slope=1, max_disparity_diff=1.5)

        time = utils.timespan_str(tools.toc(True))
        loss_str = f'loss = {utils.threshold_color(eval_dict["epe_loss"])}{eval_dict["epe_loss"]:.3f}{Style.RESET_ALL}'
        error_rate_str = f'{eval_dict["error_sum"] / eval_dict["total_eval"]:.2%}'
        print(f'[{batch_index + 1}/{len(test_loader)} {time}] {loss_str}, error rate = {error_rate_str}')

        losses.append(float(eval_dict["epe_loss"]))
        error.append(float(eval_dict["error_sum"]))
        total_eval.append(float(eval_dict["total_eval"]))

        if merge_cost:
            confidence_error.append(float(eval_dict["CE_avg"]))

        if torch.isnan(eval_dict["epe_loss"]):
            print('detect loss nan in testing')
            exit(1)

        plotter = utils.CostPlotter()

        origin_width = int(origin_width)
        origin_height = int(origin_height)

        # dimension X: batch, channel*2, height, width = 1, 6, 352, 1216
        plotter.plot_image_disparity(X[0], Y[0, 0], dataset, eval_dict,
                                     max_disparity=max_disparity,
                                     save_result_file=(f'{profile}_benchmark/{dataset}', batch_index, False,
                                                       error_rate_str),
                                     resize=(origin_width, origin_height))
        # exit(0)
        # os.system('nvidia-smi')

print(f'avg loss = {np.array(losses).mean():.3f}')
print(f'std loss = {np.array(losses).std():.3f}')
print(f'avg error rates = {np.array(error).sum() / np.array(total_eval).sum():.2%}')
if merge_cost:
    print(f'avg confidence error = {np.array(confidence_error).mean():.3f}')
print('Number of test case:', len(losses))
