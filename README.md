# energy-consumption-forecasting
Forecasting household energy project with statistical and ML models
## Files
- `energy_consumption_module.py` – main script
- `weather.json` – weather features
- `household_power_consumption.txt` – original data
- `requirements.txt` – dependencies

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run the script: `python energy_consumption_module.py`

## download dataset file
1. household_power_consumption.txt
     → Download from: https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

2.weather.json
     → Provided in the project (size: 39 KB)
      it was downloaded using this link: https://archive-api.open-meteo.com/v1/era5?latitude=48.7761&longitude=2.2901&start_date=2006-12-16&end_date=2010-11-28&daily=temperature_2m_max,temperature_2m_min,temperature_2m_mean&timezone=Europe%2FParis

## paths to the files
--> Inside the main function, insert the paths to the folloiwng location:
              1.  raw_path = r".......household_power_consumption.txt"
              2.  weather_path =r"...weather.json"


