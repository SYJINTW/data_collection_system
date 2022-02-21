import math
import os
import subprocess
import numpy as np
import pandas as pd
import json



def change_json_cameras(pose_trace):
    pose_trace_split = pose_trace.split("_")
    filename = 'test.json'
    with open(filename, 'r') as f:
        data = json.load(f)
        data['sourceCameraNames'] = [f"v{pose_trace_split[1]}"]
    os.remove(filename)
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

    # "sourceCameraNames"

def main():
    pose_trace = 'pose_0_miv'
    change_json_cameras(pose_trace)

if __name__ == '__main__':
    main()
