# Data Modeling

Predict the **six-hour occurrence band** of a reported auto theft. The label is built from `OCC_HOUR`. The features are premises type, police division, occurrence weekday, season, and occurrence year.

This extract has no outcome such as recovery, arrest, or loss. Every row is already “Theft Of Motor Vehicle,” so the modeling target has to be one attribute of the report. Time of day is that attribute: `OCC_HOUR` is filled on every row, and four bands stay easy to explain. Place is reported beside the model as a count of premises type within each band. A place classifier would have a heavy majority class (Outside is 45,439 of 82,746 rows) and, for neighbourhood, 158 classes.

Season is a frequency table, not a second model. Month, day-of-year, and the raw occurrence date would rebuild season exactly, so those columns stay out of the feature matrix.

The working notes and calculations are in `Data_Modeling.ipynb`.

## Time bands

| Band | Hours (`OCC_HOUR`) |
|---|---|
| Night | 0–5 |
| Morning | 6–11 |
| Afternoon | 12–17 |
| Evening | 18–23 |

Occurrence hour already peaks at 22 (7,288 rows). Evening is the band that contains that peak. The median hour is 16 and the standard deviation is 7.61, so the bands are broad on purpose.

## Features

| Column | Role |
|---|---|
| `PREMISES_TYPE` | Feature. Seven place groups. |
| `DIVISION` | Feature. Police division, including `NSA`. |
| `OCC_DOW` | Feature. Weekday of the occurrence, with the padding spaces removed. |
| `season` | Feature, mapped from `OCC_MONTH`. Also the column for the season count. |
| `OCC_YEAR` | Feature. Standardized only if the classifier is linear or distance-based. |
| `time_band` | Label, mapped from `OCC_HOUR`. |

| Left out | Reason |
|---|---|
| `OBJECTID` | Row identifier. |
| `EVENT_UNIQUE_ID` | Police occurrence identifier. It repeats, so it is not the modeling unit. |
| `UCR_CODE`, `UCR_EXT`, `OFFENCE`, `CSI_CATEGORY` | Constant on every row. |
| `OCC_HOUR` | Source of the label. |
| `OCC_MONTH`, `OCC_DOY`, `OCC_DATE` | Finer copies of the season feature. |
| `REPORT_HOUR` | Hour the report was taken. Correlation with occurrence hour is 0.13. |
| Report date and its parts | Timing of the report, not the occurrence band. |
| `LOCATION_TYPE` | Detail underneath premises type. Held out so place is not counted twice. |
| Neighbourhood code and name, both maps | The code and the name are the same place. The 158-area map and the 140-area map are different geographies, and the 158-area name has 158 levels. |
| `LONG_WGS84`, `LAT_WGS84`, `x`, `y` | The same intersection-offset point, stored twice. A value of 0 is a placeholder. |

## Process

1. **Transformations.** Drop the six rows whose occurrence year, month, and weekday are blank. Strip weekday padding. Map the hour to a band and the month to a season. Keep `NSA` as a category. One-hot encode the categories. Standardize `OCC_YEAR` because the classifier used here is logistic regression.
2. **Feature selection.** Pandas drops identifiers, constants, and columns that copy the label or the season. Scikit-learn ranks the five remaining columns with `mutual_info_classif` on the training rows.
3. **Split.** NumPy index array, then `train_test_split` with an 80/20 stratified split on the time band (`random_state=42`). The encoder, the scaler, and the mutual-information scores are fit on the training rows.
4. **Imbalanced classes.** Compare training-band shares before choosing a remedy. A band is treated as rare only when it is under 5 percent of the training rows. Morning is the smallest band, at 13.9 percent, so the reported model leaves the weights alone. A `class_weight="balanced"` fit is reported beside it. Resampling is not used: the features are categories, and a synthetic row would mix divisions and weekdays.

## Assumptions

1. The supervised question is: for a reported auto theft, which six-hour occurrence band does it fall in?
2. The unit of the model is the offence row, identified by `OBJECTID`. `EVENT_UNIQUE_ID` repeats on 7,432 occurrences, and one occurrence contributes as many as 20 rows.
3. Features are premises type, police division, occurrence weekday, season, and occurrence year. The label comes only from `OCC_HOUR`.
4. Report hour stays out. It records when the report was taken, and its correlation with occurrence hour is 0.13.
5. Season is a feature in the hour model and a separate frequency table. Month, day-of-year, and the raw occurrence date stay out so the model cannot rebuild the season from a finer copy of the same date.
6. `NSA` means not assigned and stays as a category. Coordinate 0 means an unknown location, and those points are not used.
7. A stratified 80/20 split estimates how well the pattern holds on other reports in this same extract. It does not estimate a future year.
8. The encoder, the scaler, and feature selection are fit on the training rows only.
9. Class weights or resampling are used only if the training bands show a rare class. The expectation is that all four bands are large enough to leave as they are.
10. Counts are counts of reports. They are not rates per person or per vehicle.
11. The 2023 peak and the later decline describe reported volume. The file cannot separate a change in theft from a change in reporting.

## Constraints

- The extract contains one offence only: UCR 2135 / 210, “Theft Of Motor Vehicle,” from 1 January 2014 through 30 September 2026.
- 2026 has 5,699 rows and stops at 30 September. It is out of full-year and full-season comparisons.
- Six rows have blank occurrence year, month, and weekday. Their occurrence dates are 1969, 1978, 1990, and 1996. Those rows are out of the model and the season table.
- Weekday names are padded with spaces to 10 characters.
- The clock time on `REPORT_DATE` and `OCC_DATE` is midnight Eastern stored in UTC. The real hours are `REPORT_HOUR` and `OCC_HOUR`.
- Published coordinates are moved to the nearest intersection. 891 rows are the placeholder 0. 931 neighbourhood rows and 414 division rows are `NSA`. 18 location types are blank.
- Neighbourhood code and neighbourhood name are the same place. The 158-area map and the 140-area map are different geographies.
- Outside, House, Commercial, and Apartment rise and fall together across 2014–2025 (correlations 0.78 to 0.95). The yearly spike is citywide.
- Twelve complete years are enough to see the volume trend. They are a thin sample for forecasting the next year’s total. This model does not make that forecast.

## Key findings

These figures come from the exploration already saved in `results/`.

- Volume on complete report years averages about 6,421 records. It stays near 3,500–3,800 through 2017, peaks at **12,652 in 2023**, then falls to 9,693 in 2024 and 7,427 in 2025.
- Thefts are recorded most often at **hour 22**. Reports are taken most often at **hour 7**.
- Outside accounts for 45,439 rows. Parking lots are the largest location type (30,164), then single homes (24,896) and streets (15,064).
- D23 has the most division rows (11,105). West Humber-Clairville has the most neighbourhood rows (6,279). Both are raw totals.
- September has the most occurrence and report rows. February has the fewest report rows. Thursday has the most report rows and Sunday the fewest.

These figures are calculated in `Data_Modeling.ipynb`.

**Season, occurrence years 2014–2025.** Autumn has the most records, and the four seasons sit close together.

| Season | Records | Share |
|---|---:|---:|
| Winter | 17,846 | 23.2% |
| Spring | 18,767 | 24.4% |
| Summer | 19,494 | 25.3% |
| Autumn | 20,953 | 27.2% |

**Time band, all modeling rows.** Evening is the largest band. Morning is the smallest and still has 11,473 rows.

| Band | Records | Share |
|---|---:|---:|
| Night (0–5) | 17,673 | 21.4% |
| Morning (6–11) | 11,473 | 13.9% |
| Afternoon (12–17) | 17,225 | 20.8% |
| Evening (18–23) | 36,369 | 44.0% |

**Premises mix inside each band.** Outside is the largest share in the morning (63.5%), afternoon (68.9%), and evening (53.2%). House is the largest share at night (44.5%) and rises again in the evening (35.2%). That is the place pattern: daytime reports concentrate outside, and night reports shift toward houses.

**Feature scores, training rows only.** Mutual information with the time band:

| Feature | Score |
|---|---:|
| Premises type | 0.0444 |
| Division | 0.0059 |
| Occurrence year | 0.0054 |
| Weekday | 0.0024 |
| Season | 0.0004 |

Season is left out of the classifier. The other four features stay in.

**Test result.** Logistic regression accuracy is **0.442**. A dummy that always predicts Evening scores **0.440**. The gap is 0.002. Evening recall is 0.948, and the other three bands are rarely predicted. Reweighting the classes drops accuracy to 0.292, which is why the reported model does not use class weights. Of 16,548 test rows, 2,456 share an occurrence id with the training set.

Place, division, weekday, and year do not pin down the hour. The useful time result is the band count itself: Evening holds 44 percent of the records, and the peak hour inside it is 22.

## Technologies

| Technology | Use in this section |
|---|---|
| Python | Notebook for the modeling steps. |
| Pandas | Cleaning, the season and time-band columns, and the season and premises-by-band counts. |
| NumPy | Index array for the split, and the numeric summaries already produced in exploration. |
| scikit-learn | `train_test_split`, `OneHotEncoder`, `StandardScaler`, `mutual_info_classif`, logistic regression, `DummyClassifier`, and the confusion matrix. |
| Matplotlib and Seaborn | Season counts, the premises-by-band chart, and the confusion matrix. |
