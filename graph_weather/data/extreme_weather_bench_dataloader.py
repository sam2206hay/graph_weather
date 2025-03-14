import torch
import xarray as xr
import dacite
from extremeweatherbench import events, utils, config
from extremeweatherbench.utils import ERA5_MAPPING
from torch.utils.data import Dataset

ARCO_ERA5_FULL_URI = (
    "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
)
DEFAULT_FORECAST_SCHEMA_CONFIG = config.ForecastSchemaConfig()

class ExtremeWeatherBenchDataset(Dataset):
    def __init__(self, event):

        gridded_obs = xr.open_zarr(
            ARCO_ERA5_FULL_URI,
            chunks=None,
            storage_options=dict(token="anon"),
        )
        for variable in ERA5_MAPPING:
                gridded_obs = gridded_obs.rename({ERA5_MAPPING[variable]: variable})

        self.gridded_obs = gridded_obs[ERA5_MAPPING]

        yaml_event_case = utils.load_events_yaml()

        for k, v in yaml_event_case.items():
            if k == "cases":
                for individual_case in v:
                    if "location" in individual_case:
                        individual_case["location"]["longitude"] = (
                            utils.convert_longitude_to_360(
                                individual_case["location"]["longitude"]
                            )
                        )
                        individual_case["location"] = utils.Location(
                            **individual_case["location"]
                        )

        self.cases = dacite.from_dict(
            data_class=event,
            data=yaml_event_case,
        )
    

    def __getitem__(self, idx):

        individual_case = self.cases.cases[idx]

        variable_subset_gridded_obs = individual_case._subset_data_vars(self.gridded_obs)
        
        time_subset_gridded_obs_ds = variable_subset_gridded_obs.sel(
            time=slice(individual_case.start_date, individual_case.end_date)
        )
        
        time_subset_gridded_obs_ds = individual_case.perform_subsetting_procedure(
            time_subset_gridded_obs_ds
        )

        print(time_subset_gridded_obs_ds)



if __name__ == '__main__':
    dataset = ExtremeWeatherBenchDataset(events.HeatWave)
    dataset[0]
    