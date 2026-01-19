import time
import subprocess
import csv
import datetime
import re
import shutil

import pandas as pd

from pathlib import Path
from multiprocessing import cpu_count, Manager
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map

def log_completion(rundata_path, lock):
    COMPLETED_LOG = Path("completed.csv")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with lock:
        with open(COMPLETED_LOG, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([str(rundata_path), timestamp])

def log_error(rundata_path, error_reason, lock):
    ERROR_LOG = Path("errors.csv")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with lock:
        with open(ERROR_LOG, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([str(rundata_path), error_reason, timestamp])

def get_processed_runs():
    COMPLETED_LOG = Path("completed.csv")
    ERROR_LOG = Path("errors.csv")
    
    processed = set()
    
    if COMPLETED_LOG.exists():
        with open(COMPLETED_LOG, "r") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row:
                    processed.add(row[0])

    if ERROR_LOG.exists():
        with open(ERROR_LOG, "r") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row:
                    processed.add(row[0])

    return processed

def run_shetran(exe_path: Path, rundata_path: Path, lock):
    """
    Executes a SHETRAN simulation.
    """
    if not exe_path.exists():
        msg = f"Error: Executable not found at {exe_path}"
        print(msg)
        log_error(rundata_path, msg, lock)
        return
    if not rundata_path.exists():
        msg = f"Error: Rundata file not found at {rundata_path}"
        print(msg)
        log_error(rundata_path, msg, lock)
        return

    working_dir = rundata_path.parent
    pri_path = working_dir / f"output_{rundata_path.name[8:-4]}_pri.txt"
    terminal_log_path = working_dir / "terminal.txt"

    command = [str(exe_path), "-f", str(rundata_path)]
    run_id_str = f"{working_dir.parent.name}/{working_dir.name}"
    
    print(f"Starting SHETRAN run for: {run_id_str}")

    try:
        start_time = time.time()
        
        with open(terminal_log_path, "w") as log_file:
            process = subprocess.Popen(
                command,
                cwd=working_dir,
                stdout=log_file,
                stderr=log_file,
                text=True
            )

            while process.poll() is None:
                # 1. Check for Timeout
                if time.time() - start_time > 6000:
                    msg = "Timeout (>6000s)"
                    print(f"Run ID {run_id_str} {msg}.")
                    process.kill()
                    log_error(rundata_path, msg, lock)
                    return

                if pri_path.exists():
                    try:
                        with open(pri_path, "r") as f:
                            output = f.read()
                            if "FATAL ERROR" in output:
                                msg = "FATAL ERROR in pri file"
                                print(f"Run ID {run_id_str} FAILED: {msg}")
                                process.kill()
                                log_error(rundata_path, msg, lock)
                                return
                            if "### Error asummary and Advice ###" in output:
                                msg = "Error Summary in pri file"
                                print(f"Run ID {run_id_str} FAILED: {msg}")
                                process.kill()
                                log_error(rundata_path, msg, lock)
                                return
                    except Exception:
                        pass

                time.sleep(5)

        if process.returncode == 0:
            print(f"SHETRAN run completed successfully for: {run_id_str}")
            log_completion(rundata_path, lock)
        else:
            msg = f"Process finished with return code: {process.returncode}"
            print(f"SHETRAN run {run_id_str} {msg}")
            log_error(rundata_path, msg, lock)

    except Exception as e:
        msg = f"Unexpected Exception: {e}"
        print(f"An unexpected error occurred in {run_id_str}: {msg}")
        log_error(rundata_path, msg, lock)
        try:
            process.kill()
        except:
            pass

def run_preprocessor(prep_exe_path: Path, xml_file_path: Path):
    """
    Executes the SHETRAN Pre-processor.
    """
    if not prep_exe_path.exists():
        print(f"Error: Pre-processor executable not found at {prep_exe_path}")
        return
    if not xml_file_path.exists():
        print(f"Error: Input file not found at {xml_file_path}")
        return

    working_dir = xml_file_path.parent
    command = [str(prep_exe_path), str(xml_file_path)]
    print(f"Starting Pre-processor run for: {working_dir.parent.name}/{working_dir.name}")

    try:
        result = subprocess.run(
            command, cwd=working_dir, capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            print(f"Pre-processing completed successfully for: {working_dir.parent.name}/{working_dir.name}")
        else:
            print(f"Pre-processing of {working_dir.parent.name}/{working_dir.name} failed with error: {result.stderr}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def create_pet_file(run_dir: Path, scenario: str):
    df = pd.read_csv(Path("outputs/shetran_inputs/pet.csv"))
    pet = df[scenario]
    output_path = run_dir / "28001_PET.csv"
    pet.to_csv(output_path, header=["Average PET current climate (mm/day)"], index=False)

def create_temp_file(run_dir: Path, scenario: str):
    df = pd.read_csv(Path("outputs/shetran_inputs/temps.csv"))
    temps = df[scenario]
    output_path = run_dir / "28001_Temp.csv"
    temps.to_csv(output_path, header=["average daily temperature "], index=False)        

def run_sort_helper(path):
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', path.name)]

def shetran_runner():
    COMPLETED_LOG = Path("completed.csv")
    ERROR_LOG = Path("errors.csv")

    shetran_model_dir = Path("data/shetran")
    shetran_ensemble_dir = Path("outputs/shetran").resolve()
    rainfall_ensemble_dir = Path("outputs/rainfall_ensemble")

    scenarios = [
                "Baseline",
                "RCP2.6_10th",
                "RCP2.6_50th",
                "RCP2.6_90th",
                "RCP4.5_10th",
                "RCP4.5_50th",
                "RCP4.5_90th",
                "RCP6.0_10th",
                "RCP6.0_50th",
                "RCP6.0_90th",
                "RCP8.5_10th",
                "RCP8.5_50th",
                "RCP8.5_90th"
    ]

    for item in rainfall_ensemble_dir.iterdir():
        rainfall = pd.read_csv(item)
        dir_name = item.stem
        outer_directory =shetran_ensemble_dir / dir_name
        outer_directory.mkdir(exist_ok=True)
        for scenario in scenarios:
            inner_directory = outer_directory / scenario
            inner_directory.mkdir(exist_ok=True)
            shutil.copytree(shetran_model_dir, inner_directory, dirs_exist_ok=True)
            create_pet_file(inner_directory, scenario)
            create_temp_file(inner_directory, scenario)
            rain = rainfall[scenario]
            rain.to_csv(inner_directory / "28001_Rainfall_sim.csv", header=["1"], index=False)
    
    if not COMPLETED_LOG.exists():
        with open(COMPLETED_LOG, "w", newline='') as f:
            csv.writer(f).writerow(["Rundata Path", "Timestamp"])
            
    if not ERROR_LOG.exists():
        with open(ERROR_LOG, "w", newline='') as f:
            csv.writer(f).writerow(["Rundata Path", "Error Reason", "Timestamp"])

    print("Loading previous run history...")
    processed_runs = get_processed_runs()
    print(f"Found {len(processed_runs)} previously processed runs (completed or failed).")

    scenarios = [
            "Baseline", "RCP2.6_10th", "RCP2.6_50th", "RCP2.6_90th",
            "RCP4.5_10th", "RCP4.5_50th", "RCP4.5_90th",
            "RCP6.0_10th", "RCP6.0_50th", "RCP6.0_90th",
            "RCP8.5_10th", "RCP8.5_50th", "RCP8.5_90th"
        ]
    
    pre_processor_args = []
    shetran_args = []

    manager = Manager()
    csv_lock = manager.Lock()

    print("Scanning directories for new runs...")

    all_directories = [d for d in shetran_ensemble_dir.iterdir() if d.is_dir()]
    sorted_directories = sorted(all_directories, key=run_sort_helper)
    
    for directory in sorted_directories:
        for subdirectory in scenarios:
            full_path = directory / subdirectory
            
            rundata_file = full_path / "rundata_28001.txt"

            if str(rundata_file) in processed_runs:
                continue

            pre_processor_args.append((Path("C:/Users/ellis/Documents/Shetran2/shetran-prepare-2.2.9-snow.exe"), full_path / "LibraryFile460.xml"))
            
            shetran_args.append((
                Path("C:/Users/ellis/Documents/Shetran2/sv4.4.5x64.exe"), 
                rundata_file,
                csv_lock
            ))

    print(f"Queueing {len(shetran_args)} runs.")

    # Phase 1: Pre-processor
    print("--- Starting Pre-processor Phase ---")
    for args in tqdm(pre_processor_args, desc="Pre-processing"):
        run_preprocessor(*args)
    # Phase 2: Main Simulation
    if shetran_args:
        print("--- Starting SHETRAN Simulation Phase (Parallel) ---")
        thread_map(
            lambda p: run_shetran(*p), 
            shetran_args, 
            max_workers=cpu_count(), 
            desc="Simulations"
        )
        print("--- All runs finished ---")
    else:
        print("No new runs to process.")