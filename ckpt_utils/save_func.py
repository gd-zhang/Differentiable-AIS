import torch, sys, os, time, glob

"""
Save 3 latest checkpoints. Return the checkpoint path

- Checkpoints will be named "checkpoint_0.pth", "checkpoint_1.pth", "checkpoint_2.pth".

- Assumes no other functions will save checkpoints to save_dir

- Writes information about checkpoint taken into the file <checkpoint_name>.stats
  for statistics gathering purposes.
"""

def checkpoint_save(state_dict, save_dir):
    # determine the file where to save the checkpoint
    checkpoints = glob.glob(os.path.join(save_dir, '*.pth'))
    flag = 0
    if len(checkpoints) == 0:
        flag = 0
        checkpoint_location = 'checkpoint_0.pth'
        checkpoint_path = os.path.join(save_dir, checkpoint_location)
    elif len(checkpoints) == 1:
        flag = 1
        checkpoint_location = 'checkpoint_1.pth'
        checkpoint_path = os.path.join(save_dir, checkpoint_location)
    else:
        #there are 3 checkpoints in the directory, we overwrite the oldest
        last_modified_dates = [os.path.getmtime(x) for x in checkpoints]
        last_modified_index = last_modified_dates.index(min(last_modified_dates))
        checkpoint_path = checkpoints[last_modified_index]

    # save the checkpoint to the location and log the time and size
    t0 = time.perf_counter()
    checkpoint = torch.save(state_dict, checkpoint_path)
    t1 = time.perf_counter()
    save_time = t1 - t0
    statinfo = os.stat(checkpoint_path)
    cmd = "echo \"{}\n{}\" > {}.stats".format(statinfo.st_size, save_time, checkpoint_path)
    os.system(cmd)
    return checkpoint_path