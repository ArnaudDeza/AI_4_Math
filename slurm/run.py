# run_cem_jobs.py

import subprocess
import time
import os
from parameter_grid import generate_parameter_grid
from datetime import datetime, timedelta

def run_cem_jobs():
    """
    Manages the execution of CEM training jobs in parallel.
    Each job uses one GPU and runs a unique set of parameters.
    """
    parameter_sets = generate_parameter_grid()
    total_params = len(parameter_sets)
    print(f"Total parameter sets generated: {total_params}")

    # Write all parameters to a file
    with open('parameters_tested.txt', 'w') as f:
        f.write("Parameters being tested:\n")
        for idx, params in enumerate(parameter_sets):
            f.write(f"Set {idx+1}: {params}\n")

    num_gpus = 8
    gpu_list = list(range(num_gpus))
    processes = []
    current_idx = 0
    start_time = datetime.now()
    end_time = start_time + timedelta(hours=4) - timedelta(minutes=5)  # 5-minute buffer

    while datetime.now() < end_time and (current_idx < total_params or processes):
        # Clean up finished processes
        for p in processes[:]:
            if p['process'].poll() is not None:
                gpu_list.append(p['gpu'])
                processes.remove(p)

        # Start new processes if GPUs are available
        while gpu_list and current_idx < total_params and datetime.now() < end_time:
            params = parameter_sets[current_idx]
            gpu = gpu_list.pop(0)
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(gpu)
            cmd = ['python', 'cem_train.py']
            for key, value in params.items():
                if isinstance(value, list):
                    cmd.append(f'--{key}')
                    cmd.extend(map(str, value))
                else:
                    cmd.extend([f'--{key}', str(value)])
                
            # ALSO ADD current idx to cmd
            cmd.extend(['--current_idx', str(current_idx)])
            print(f"Launching process {current_idx+1} on GPU {gpu} with params: {params}")
            process = subprocess.Popen(cmd, env=env)
            processes.append({'process': process, 'gpu': gpu})
            current_idx += 1

        time.sleep(5)  # Wait before checking again

    # Wait for all processes to finish
    for p in processes:
        p['process'].wait()

if __name__ == "__main__":
    run_cem_jobs()
