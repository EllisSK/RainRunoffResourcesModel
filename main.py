from args_parser import CustomParser

from rainfall_model import *
from temp_pet_model import *
from shetran_runner import *
from resource_model import *

def main():
    parser = CustomParser()
    args = parser.parse_args()

    if args.rainfall:
        measured_rainfall_path = Path("data/nfra/28001_cdr.csv")
        measured_rainfall_data = pd.read_csv(measured_rainfall_path, skiprows=20, names=["Date", "Rainfall (mm)"])
        measured_rainfall_data.index = pd.to_datetime(measured_rainfall_data["Date"])
        measured_rainfall_data = measured_rainfall_data.drop(columns=["Date"])

        full_rainfall_data = measured_rainfall_data
        cropped_rainfall_data = measured_rainfall_data["1981":"2000"]

        transition_matricies = calculate_transition_matricies(full_rainfall_data)
        print("Transition Matricies:\n", transition_matricies)

        parameter_matricies = fit_gamma_function(cropped_rainfall_data)
        print("Gamma Function Parameters:\n", parameter_matricies)

        outputs = Path("outputs")
        ensemble_directory = outputs / "rainfall_ensemble"
        produce_timeseries_ensemble(transition_matricies, parameter_matricies, ensemble_directory, 150)
    elif args.climate:
        baseline_temps = calculate_baseline_temps()
        baseline_pet = calculate_baseline_pet()

        synthetic_temps = create_synthetic_temps(baseline_temps)

        _, pet_model = fit_pet_model()
        _ = create_synthetic_pet(baseline_pet, synthetic_temps, pet_model)
    elif args.shetran:
        shetran_runner()
    elif args.resource:
        shetran_ensemble_dir = Path("outputs/shetran")
        run_resource_model_on_shetran_ensemble(shetran_ensemble_dir)
    else:
        "Please provide some arguments."

    return 0


if __name__ == "__main__":
    main()