import torch, sys, os, time, glob

"""
Loop through all checkpoints in save_dir and attempt to load the latest one.
"""

def latest_checkpoint_load(save_dir):
    checkpoints = glob.glob(os.path.join(save_dir, '*.pth'))
    while checkpoints != []:
        # find latest checkpoint
        last_modified_dates = [os.path.getmtime(x) for x in checkpoints]
        last_modified_index = last_modified_dates.index(max(last_modified_dates))
        # checkpoint_location is the latest checkpoint
        checkpoint_location = checkpoints[last_modified_index]
        try:
            # try to load the latest checkpoint and return it
            checkpoint = torch.load(checkpoint_location)
            message = "\'Loaded checkpoint {}\'".format(checkpoint_location)
            os.system("echo " + message + " >> " + save_dir + "/output")
            return (checkpoint, checkpoint_location)
        except Exception:
            message = "Could not load checkpoint file {}. Skipping it and trying the next one.".format(checkpoint_location)
            os.system("echo " + message + " >> " + save_dir + "/output")
            del checkpoints[last_modified_index]
    message = "Could not find a working checkpoint file in {}".format(save_dir)
    os.system("echo " + message + " >> " + save_dir + "/output")
    return None