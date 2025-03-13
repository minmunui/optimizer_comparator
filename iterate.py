import os
import subprocess
import multiprocessing
from itertools import product
import time
import argparse


def run_with_params(params: dict) -> bool:
    """
    Function to run Python script with given parameters

    Args:
        params: Dictionary containing parameter values
    """

    output_file = f"results!{parameter_status(list(params.values()), list(params.keys()))}.txt"

    # Construct command to execute
    command = f"python main.py {command_parameters(params)}"

    # Initialize result file
    with open(output_file, 'w') as f:
        f.write(f"Running command: {command}\n")
        f.write("-" * 50 + "\n\n")

    print(f"Starting: {command}")

    # Activate virtual environment and execute command
    try:
        # Virtual environment path
        venv_activate = os.path.abspath(".venv\\Scripts\\activate.bat")

        # Full command (activate venv + run python script)
        full_command = f"cmd /c \"{venv_activate} && {command} && deactivate\""

        # Execute subprocess and capture output
        result = subprocess.run(
            full_command,
            shell=True,
            capture_output=True,
            text=True
        )

        # Save results to file
        with open(output_file, 'a') as f:
            f.write(result.stdout)
            if result.stderr:
                f.write("\nErrors:\n")
                f.write(result.stderr)

        print(f"Completed: {parameter_status(params, list(params.keys()))}")
        return True

    except Exception as e:
        print(f"Error with parameters: {params}, Error: {str(e)}")
        with open(output_file, 'a') as f:
            f.write(f"\nError occurred: {str(e)}\n")
        return False


def parameter_status(param_values: list[str], param_keys: list[str]) -> str:
    """
    Function to generate a string representation of parameter values

    Args:
        param_values: Dictionary containing parameter values
        param_keys: List of parameter names

    Returns:
        String representation of parameter values
    """
    print(f"param_values: {param_values}")
    return "!".join([f"{key}={value}" for key, value in zip(param_keys, param_values)])


def command_parameters(params: dict) -> str:
    return " ".join([f"--{key} {value}" for key, value in params.items()])


def main():
    parser = argparse.ArgumentParser(description='Run parameter combinations in parallel')
    parser.add_argument('--processes', type=int, default=multiprocessing.cpu_count(),
                        help='Number of processes to use (default: number of CPU cores)')
    args = parser.parse_args()

    # Set parameter values
    # mutation_rate_values = [0.001, 0.005, 0.02, 0.04]
    # crossover_rate_values = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    # seed_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    candidate_dict = {
        # "da_initial_temp": [2615, 5230, 10460, 20920],
        # "da_maxiter": [1000, 5000, 10000, 15000],
        # "random_seed": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        # "fraction": [4],
        # "solver": ["da"]
        "ga_crossover_rate": [1.0],
        "ga_mutation_rate": [0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3],
        "random_seed": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "ga_immigration_size": [100, 200, 300, 400, 500, 600],
        "solver": ["ga"],
        "fraction": [3, 4]
    }

    # Generate all parameter combinations
    parameter_combinations = list(product(*candidate_dict.values()))
    total_combinations = len(parameter_combinations)

    print(f"Running {total_combinations} parameter combinations")
    print(f"Number of parallel processes: {args.processes}")

    # Create summary file
    with open("summary.txt", "w") as f:
        f.write(f"Summary of {total_combinations} parameter combinations\n")
        f.write("Start time: " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
        f.write("-" * 50 + "\n\n")
        f.write("List of parameter combinations:\n")
        for i, param_value in enumerate(parameter_combinations, 1):
            param_status = parameter_status(param_value, list(candidate_dict.keys()))
            f.write(f"{i}. {param_status}\n")

    # Create process pool and execute tasks in parallel
    start_time = time.time()

    parameter_dicts = [dict(zip(candidate_dict.keys(), values)) for values in parameter_combinations]

    with multiprocessing.Pool(processes=args.processes) as pool:
        results = pool.map(run_with_params, parameter_dicts)

    # Update summary after execution
    end_time = time.time()
    execution_time = end_time - start_time

    successful = sum(1 for r in results if r)

    with open("summary.txt", "a") as f:
        f.write("\n" + "-" * 50 + "\n")
        f.write(f"Completion time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total execution time: {execution_time:.2f} seconds\n")
        f.write(f"Success: {successful}/{total_combinations}\n")

    print(f"\nAll executions completed. {successful}/{total_combinations} successful")
    print(f"Total execution time: {execution_time:.2f} seconds")
    print("Results for each parameter combination are saved in individual files")
    print("Execution summary can be found in summary.txt")


if __name__ == "__main__":
    main()
