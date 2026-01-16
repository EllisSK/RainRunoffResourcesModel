import time
import subprocess
import csv
import datetime

from pathlib import Path
from multiprocessing import cpu_count, Manager
from multiprocessing.pool import ThreadPool

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

def shetran_runner():
    shetran_model_dir = Path("data/shetran")
    shetran_ensemble_dir = Path("outputs/shetran").resolve()
    
    # --- Setup CSV files if they don't exist ---
    if not COMPLETED_LOG.exists():
        with open(COMPLETED_LOG, "w", newline='') as f:
            csv.writer(f).writerow(["Rundata Path", "Timestamp"])
            
    if not ERROR_LOG.exists():
        with open(ERROR_LOG, "w", newline='') as f:
            csv.writer(f).writerow(["Rundata Path", "Error Reason", "Timestamp"])

    # --- Load History for Resuming ---
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

    # Use a Manager logic to handle the lock across threads/processes safely
    manager = Manager()
    csv_lock = manager.Lock()

    print("Scanning directories for new runs...")
    
    for directory in shetran_ensemble_dir.iterdir():
        if directory.is_dir():
            for subdirectory in scenarios:
                full_path = directory / subdirectory
                
                rundata_file = full_path / "rundata_28001.txt"
                
                # --- RESUME CHECK ---
                # We convert to string to match how they are stored in the CSV
                if str(rundata_file) in processed_runs:
                    continue # Skip this run as it is already in completed.csv or errors.csv

                # Uncomment and adjust paths as needed
                pre_processor_args.append((Path("C:/Users/ellis/Documents/Shetran2/shetran-prepare-2.2.9-snow.exe"), full_path / "LibraryFile460.xml"))
                
                shetran_args.append((
                    Path("C:/Users/ellis/Documents/Shetran2/sv4.4.5x64.exe"), 
                    rundata_file,
                    csv_lock  # Pass the lock to the function
                ))

    print(f"Queueing {len(shetran_args)} runs.")

    # Phase 1: Pre-processor
    print("--- Starting Pre-processor Phase ---")
    for args in pre_processor_args:
        run_preprocessor(*args)

    # Phase 2: Main Simulation
    if shetran_args:
        print("--- Starting SHETRAN Simulation Phase (Parallel) ---")
        pool_sim = ThreadPool(processes=cpu_count())
        try:
            pool_sim.starmap(run_shetran, shetran_args)
        finally:
            pool_sim.close()
            pool_sim.join()
            print("--- All runs finished ---")
    else:
        print("No new runs to process.")