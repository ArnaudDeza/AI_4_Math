# test_cem.py

import subprocess

def test_cem():
    """
    Runs CEM training with various parameters for 5 iterations each.
    """
    test_params = [
        {'model': 'MLP', 'learning_rate': 0.001, 'batch_size': 64, 'iterations': 5, 'hidden_sizes': [128, 64]},
         ]
    for idx, params in enumerate(test_params):
        cmd = ['python', 'CEM_train.py']
        for key, value in params.items():
            if isinstance(value, list):
                value = ' '.join(map(str, value))
            cmd.extend([f'--{key}', str(value)])
        print(f"Running test {idx+1} with params: {params}")
        subprocess.run(cmd)

if __name__ == '__main__':
    test_cem()
