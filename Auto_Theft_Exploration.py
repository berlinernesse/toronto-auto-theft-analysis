# -*- coding: utf-8 -*-
"""Toronto Police Auto Theft Open Data — complete data exploration.

Loads the Auto Theft CSV and writes column descriptions, statistical
summaries, missing-data checks, and charts. Pandas profiles the table.
NumPy calculates means, numeric missing counts, and Pearson correlations.
Matplotlib and seaborn draw the figures.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Locate the CSV beside this script or in its datasets/ subfolder.
project_folder = Path(__file__).resolve().parent
candidates = sorted((project_folder / 'datasets').glob('Auto_Theft_Open_Data_*.csv')) if (project_folder / 'datasets').exists() else []
candidates += sorted(project_folder.glob('Auto_Theft_Open_Data_*.csv'))
if not candidates:
    raise FileNotFoundError('Put the Auto_Theft_Open_Data CSV in datasets/ beside this script.')
csv_path = candidates[0]

charts_folder = project_folder / 'charts'
results_folder = project_folder / 'results'
charts_folder.mkdir(exist_ok=True)
results_folder.mkdir(exist_ok=True)
sns.set_theme(style='whitegrid')


def save_chart(filename):
    """Save the current figure and close it so the script can finish unattended."""
    plt.tight_layout()
    plt.savefig(charts_folder / filename, dpi=200)
    plt.close()


def comma(value):
    return f'{int(value):,}'


# Business meaning of each column. Pandas dtypes are recorded separately because
# several of them differ from the type a warehouse should store.
DESCRIPTIONS = {
    'OBJECTID': 'Unique row identifier in this extract. It identifies a row, not a separate statistical measure of crime.',
    'EVENT_UNIQUE_ID': 'Police occurrence identifier (GO number). The file is published at offence level, so one identifier can appear on more than one row.',
    'REPORT_DATE': 'Date the offence was reported. The publisher stores this timestamp in UTC. The clock time is midnight Eastern, not the hour the report was taken.',
    'OCC_DATE': 'Date the offence occurred, also stored in UTC with the same midnight convention. A few rows contain a date even though the occurrence year, month, and day fields are blank.',
    'REPORT_YEAR': 'Four-digit year of the report date.',
    'REPORT_MONTH': 'Month name of the report date.',
    'REPORT_DAY': 'Day of the month of the report date.',
    'REPORT_DOY': 'Day of the year of the report date. Leap years use 366.',
    'REPORT_DOW': 'Weekday name of the report date. Each name is padded with spaces to a width of 10 characters.',
    'REPORT_HOUR': 'Hour the offence was reported, on a 0–23 clock. This is the reporting hour; it is separate from the clock time stored on REPORT_DATE.',
    'OCC_YEAR': 'Year the offence occurred. Pandas stores it as a float because six rows are blank. The business type is a year.',
    'OCC_MONTH': 'Month name of the occurrence date. Blank on the same six rows as OCC_YEAR.',
    'OCC_DAY': 'Day of the month of the occurrence. Stored as a float because those six rows are blank.',
    'OCC_DOY': 'Day of the year of the occurrence. Stored as a float because those six rows are blank.',
    'OCC_DOW': 'Weekday name of the occurrence. Names are space-padded to 10 characters, and six rows are blank.',
    'OCC_HOUR': 'Hour the offence occurred, on a 0–23 clock. This field is filled even on rows whose occurrence date components are blank.',
    'DIVISION': 'Toronto Police division recorded for the occurrence. The value NSA means the row was not assigned to a division.',
    'LOCATION_TYPE': 'Detailed place where the offence occurred, such as a parking lot, a house, or a street.',
    'PREMISES_TYPE': 'Grouped premises category for the location (Outside, House, Commercial, Apartment, Other, Transit, or Educational).',
    'UCR_CODE': 'Statistics Canada Uniform Crime Reporting code. Every row in this extract is theft of a motor vehicle.',
    'UCR_EXT': 'Extension of the UCR code. It is constant in this extract.',
    'OFFENCE': 'Offence name published with the UCR code.',
    'CSI_CATEGORY': 'Community Safety Indicator category used by Toronto Police.',
    'HOOD_158': 'Neighbourhood code from the City of Toronto 158-area geography. Codes are stored as text and include NSA.',
    'NEIGHBOURHOOD_158': 'Neighbourhood name under the 158-area geography, with the area number in parentheses. NSA means the row was not assigned to a neighbourhood.',
    'HOOD_140': 'Neighbourhood code from the older 140-area geography. Some codes are zero-padded, and NSA is included.',
    'NEIGHBOURHOOD_140': 'Neighbourhood name under the 140-area geography, including NSA.',
    'LONG_WGS84': 'Longitude in WGS84 decimal degrees. Published points are offset to the nearest intersection. A value of 0 is a placeholder, not a Toronto location.',
    'LAT_WGS84': 'Latitude in WGS84 decimal degrees. The same intersection offset and zero-value placeholder apply.',
    'x': 'Web Mercator easting in metres, corresponding to LONG_WGS84. Placeholder rows are 0.',
    'y': 'Web Mercator northing in metres, corresponding to LAT_WGS84. Placeholder rows are 0.',
}


# 1. LOAD AND DESCRIBE DATA ELEMENTS
print('Loading:', csv_path.name)
df = pd.read_csv(csv_path, low_memory=False)
print('\n=== 1. DATASET OVERVIEW ===')
print('Rows, columns:', df.shape)
print('Pandas dtypes:\n', df.dtypes.to_string())
print('\nFirst five rows:\n', df.head().to_string(index=False))

report_dt = pd.to_datetime(df['REPORT_DATE'], format='mixed')
occ_dt = pd.to_datetime(df['OCC_DATE'], format='mixed')
print('\nReport dates:', report_dt.min().date(), 'to', report_dt.max().date())
print('Occurrence dates:', occ_dt.min().date(), 'to', occ_dt.max().date())
print('Crime category:\n', df['CSI_CATEGORY'].value_counts(dropna=False).to_string())
print('Offence:\n', df['OFFENCE'].value_counts(dropna=False).to_string())

# Numeric ranges use NumPy. Categorical values use pandas value counts.
report_year_values = df['REPORT_YEAR'].to_numpy(dtype=float)
occ_year_values = df['OCC_YEAR'].to_numpy(dtype=float)
report_hour_values = df['REPORT_HOUR'].to_numpy(dtype=float)
occ_hour_values = df['OCC_HOUR'].to_numpy(dtype=float)

profile = pd.DataFrame({
    'data_type': df.dtypes.astype(str),
    'non_null': df.notna().sum(),
    'missing': df.isna().sum(),
    'unique_values': df.nunique(dropna=True),
})
profile.index.name = 'column'
profile.to_csv(results_folder / 'column_profile.csv')

premises = df['PREMISES_TYPE'].value_counts()
report_months = df['REPORT_MONTH'].value_counts()
report_dow = df['REPORT_DOW'].value_counts()
divisions = df['DIVISION'].value_counts()
location_counts = df['LOCATION_TYPE'].value_counts(dropna=False)
location_counts.rename('records').rename_axis('location_type').to_csv(results_folder / 'location_type_counts.csv')
hood = df.loc[df['NEIGHBOURHOOD_158'].ne('NSA'), 'NEIGHBOURHOOD_158'].value_counts()

valid_coords = (df['LAT_WGS84'] != 0) & (df['LONG_WGS84'] != 0)
report_clock = df['REPORT_DATE'].astype(str).str.extract(r'(\d{1,2}:\d{2}:\d{2} [AP]M)$')[0].value_counts()
occ_clock = df['OCC_DATE'].astype(str).str.extract(r'(\d{1,2}:\d{2}:\d{2} [AP]M)$')[0].value_counts()
report_hour_mode = int(pd.Series(report_hour_values).mode().iloc[0])
occ_hour_mode = int(pd.Series(occ_hour_values).mode().iloc[0])

division_codes = ', '.join(sorted(code for code in divisions.index if code != 'NSA'))
range_or_values = {
    'OBJECTID': f"{int(np.min(df['OBJECTID'].to_numpy()))} to {comma(np.max(df['OBJECTID'].to_numpy()))}, one distinct value on every row.",
    'EVENT_UNIQUE_ID': f"{comma(df['EVENT_UNIQUE_ID'].nunique())} distinct identifiers. Example: {df['EVENT_UNIQUE_ID'].iloc[0]}. {comma(df['EVENT_UNIQUE_ID'].duplicated().sum())} additional rows reuse an identifier.",
    'REPORT_DATE': (
        f"{comma(df['REPORT_DATE'].nunique())} distinct values, from {report_dt.min().date().isoformat()} "
        f"to {report_dt.max().date().isoformat()}. Clock time is only "
        + ', '.join(f'{label} ({comma(count)} rows)' for label, count in report_clock.items())
        + '.'
    ),
    'OCC_DATE': (
        f"{comma(df['OCC_DATE'].nunique())} distinct values, from {occ_dt.min().date().isoformat()} "
        f"to {occ_dt.max().date().isoformat()}. Clock time is only "
        + ', '.join(f'{label} ({comma(count)} rows)' for label, count in occ_clock.items())
        + f". Non-null OCC_YEAR runs from {int(np.nanmin(occ_year_values))} to {int(np.nanmax(occ_year_values))}."
    ),
    'REPORT_YEAR': (
        f"{int(np.min(report_year_values))} to {int(np.max(report_year_values))} "
        f"({df['REPORT_YEAR'].nunique()} years). 2026 is partial: {comma((df['REPORT_YEAR'] == 2026).sum())} rows, January through September."
    ),
    'REPORT_MONTH': (
        f"All 12 month names. Most rows: {report_months.index[0]} ({comma(report_months.iloc[0])}). "
        f"Fewest rows: {report_months.index[-1]} ({comma(report_months.iloc[-1])})."
    ),
    'REPORT_DAY': f"1 to 31. Mean {np.mean(df['REPORT_DAY'].to_numpy(dtype=float)):.2f}.",
    'REPORT_DOY': f"1 to 366. Mean {np.mean(df['REPORT_DOY'].to_numpy(dtype=float)):.2f}.",
    'REPORT_DOW': (
        'Seven weekday names, each padded to 10 characters: '
        + ', '.join(repr(name) for name in sorted(df['REPORT_DOW'].unique()))
        + f". Most rows: {str(report_dow.index[0]).strip()} ({comma(report_dow.iloc[0])}). "
        f"Fewest rows: {str(report_dow.index[-1]).strip()} ({comma(report_dow.iloc[-1])})."
    ),
    'REPORT_HOUR': (
        f"0 to 23. Mean {np.nanmean(report_hour_values):.2f}, median {np.nanmedian(report_hour_values):.0f}, "
        f"most frequent hour {report_hour_mode} ({comma((df['REPORT_HOUR'] == report_hour_mode).sum())} rows)."
    ),
    'OCC_YEAR': (
        f"{int(np.nanmin(occ_year_values))} to {int(np.nanmax(occ_year_values))} "
        f"on {comma(np.sum(~np.isnan(occ_year_values)))} non-null rows "
        f"({df['OCC_YEAR'].nunique(dropna=True)} distinct years). Mean {np.nanmean(occ_year_values):.2f}. Six rows are blank."
    ),
    'OCC_MONTH': (
        f"All 12 month names on non-null rows. Most rows: {df['OCC_MONTH'].value_counts().index[0]} "
        f"({comma(df['OCC_MONTH'].value_counts().iloc[0])}). Six rows are blank."
    ),
    'OCC_DAY': f"1 to 31. Mean {np.nanmean(df['OCC_DAY'].to_numpy(dtype=float)):.2f}. Six rows are blank.",
    'OCC_DOY': f"1 to 366. Mean {np.nanmean(df['OCC_DOY'].to_numpy(dtype=float)):.2f}. Six rows are blank.",
    'OCC_DOW': 'Same seven space-padded weekday names as REPORT_DOW. Six rows are blank.',
    'OCC_HOUR': (
        f"0 to 23. Mean {np.nanmean(occ_hour_values):.2f}, median {np.nanmedian(occ_hour_values):.0f}, "
        f"most frequent hour {occ_hour_mode} ({comma((df['OCC_HOUR'] == occ_hour_mode).sum())} rows)."
    ),
    'DIVISION': (
        f"18 labels: {division_codes}, and NSA. "
        f"Most rows: {divisions.index[0]} ({comma(divisions.iloc[0])}). "
        f"Fewest division: {divisions.drop(labels='NSA').index[-1]} ({comma(divisions.drop(labels='NSA').iloc[-1])}). "
        f"NSA: {comma(divisions.get('NSA', 0))}."
    ),
    'LOCATION_TYPE': (
        f"{df['LOCATION_TYPE'].nunique(dropna=True)} categories, plus {int(df['LOCATION_TYPE'].isna().sum())} blank rows. "
        f"Most common: {location_counts.index[0]} ({comma(location_counts.iloc[0])}); "
        f"{location_counts.index[1]} ({comma(location_counts.iloc[1])}); "
        f"{location_counts.index[2]} ({comma(location_counts.iloc[2])}). "
        'The full list is in results/location_type_counts.csv.'
    ),
    'PREMISES_TYPE': '; '.join(f'{name} ({comma(count)})' for name, count in premises.items()) + '.',
    'UCR_CODE': f"{int(df['UCR_CODE'].iloc[0])} on every row.",
    'UCR_EXT': f"{int(df['UCR_EXT'].iloc[0])} on every row.",
    'OFFENCE': f"{df['OFFENCE'].iloc[0]} on every row.",
    'CSI_CATEGORY': f"{df['CSI_CATEGORY'].iloc[0]} on every row.",
    'HOOD_158': f"{df['HOOD_158'].nunique()} distinct text codes, including NSA. Example: {df['HOOD_158'].iloc[0]}.",
    'NEIGHBOURHOOD_158': (
        f"{df['NEIGHBOURHOOD_158'].nunique()} labels, including NSA ({comma((df['NEIGHBOURHOOD_158'] == 'NSA').sum())} rows). "
        f"Highest raw count: {hood.index[0]} ({comma(hood.iloc[0])} rows)."
    ),
    'HOOD_140': (
        f"{df['HOOD_140'].nunique()} distinct text codes, including NSA. "
        f"Example of zero-padding: {df['HOOD_140'].iloc[0]}."
    ),
    'NEIGHBOURHOOD_140': f"{df['NEIGHBOURHOOD_140'].nunique()} labels, including NSA. Example: {df['NEIGHBOURHOOD_140'].iloc[0]}.",
    'LONG_WGS84': (
        f"Valid values {df.loc[valid_coords, 'LONG_WGS84'].min():.3f} to {df.loc[valid_coords, 'LONG_WGS84'].max():.3f}. "
        f"{comma((~valid_coords).sum())} rows are 0."
    ),
    'LAT_WGS84': (
        f"Valid values {df.loc[valid_coords, 'LAT_WGS84'].min():.3f} to {df.loc[valid_coords, 'LAT_WGS84'].max():.3f}. "
        f"The same {comma((~valid_coords).sum())} rows are 0."
    ),
    'x': (
        f"Valid values {df.loc[valid_coords, 'x'].min():,.0f} to {df.loc[valid_coords, 'x'].max():,.0f} metres. "
        f"The same {comma((~valid_coords).sum())} rows are 0."
    ),
    'y': (
        f"Valid values {df.loc[valid_coords, 'y'].min():,.0f} to {df.loc[valid_coords, 'y'].max():,.0f} metres. "
        f"The same {comma((~valid_coords).sum())} rows are 0."
    ),
}

dictionary = pd.DataFrame({
    'description': pd.Series(DESCRIPTIONS),
    'data_type': profile['data_type'],
    'range_or_values': pd.Series(range_or_values),
    'non_null': profile['non_null'],
    'missing': profile['missing'],
    'unique_values': profile['unique_values'],
})
dictionary.index.name = 'column'
if list(dictionary.index) != list(df.columns) or dictionary['description'].isna().any() or dictionary['range_or_values'].isna().any():
    raise RuntimeError('Every column needs a description and a range or value summary.')
dictionary.to_csv(results_folder / 'column_dictionary.csv')
print('\n=== 2. COLUMN DESCRIPTIONS, TYPES, RANGES, AND VALUES ===')
print(dictionary[['data_type', 'unique_values', 'description']].to_string())
print('\nRanges and values:')
for column_name, text in dictionary['range_or_values'].items():
    print(f'- {column_name}: {text}')


# 2. STATISTICAL ASSESSMENTS (MEANS, AVERAGES, CORRELATIONS)
print('\n=== 3. DESCRIPTIVE STATISTICS ===')
quant_cols = ['REPORT_YEAR', 'OCC_YEAR', 'REPORT_HOUR', 'OCC_HOUR', 'REPORT_DAY', 'OCC_DAY', 'REPORT_DOY', 'OCC_DOY']
# Show the pandas summary, then the NumPy summary used in the report.
pandas_summary = df[quant_cols].describe().round(2)
print('pandas describe():\n', pandas_summary.to_string())
pandas_summary.to_csv(results_folder / 'descriptive_statistics_pandas.csv')

stat_rows = []
for col in quant_cols:
    values = df[col].to_numpy(dtype=float)
    stat_rows.append({
        'column': col,
        'count': int(np.sum(~np.isnan(values))),
        'mean': round(float(np.nanmean(values)), 2),
        'median': round(float(np.nanmedian(values)), 2),
        'std': round(float(np.nanstd(values, ddof=1)), 2),
        'min': round(float(np.nanmin(values)), 2),
        'q25': round(float(np.nanpercentile(values, 25)), 2),
        'q75': round(float(np.nanpercentile(values, 75)), 2),
        'max': round(float(np.nanmax(values)), 2),
    })
stats = pd.DataFrame(stat_rows).set_index('column')
print('\nNumPy mean, median, sample standard deviation, and quartiles:\n', stats.to_string())
stats.to_csv(results_folder / 'descriptive_statistics.csv')

# The mean of a year or an hour is an average of those codes. Annual volume is a separate average.
annual = df.loc[df['REPORT_YEAR'].between(2014, 2025)].groupby('REPORT_YEAR').size().rename('record_count')
annual.to_csv(results_folder / 'annual_counts.csv')
annual_mean = round(float(np.mean(annual.to_numpy(dtype=float))), 2)
annual_median = round(float(np.median(annual.to_numpy(dtype=float))), 2)
print('\nRecords per complete report year (2014–2025):\n', annual.to_string())
print(f'Average records per complete year: mean {annual_mean:,.2f}, median {annual_median:,.2f}.')
print('Most frequent occurrence hour:', occ_hour_mode, int((df['OCC_HOUR'] == occ_hour_mode).sum()))
print('Most frequent report hour:', report_hour_mode, int((df['REPORT_HOUR'] == report_hour_mode).sum()))

# Pearson correlations of the quantitative time fields. Identifiers, UCR codes,
# and map coordinates are excluded because averaging or correlating them does not measure crime.
corr_cols = ['REPORT_YEAR', 'OCC_YEAR', 'REPORT_HOUR', 'OCC_HOUR']
complete = df[corr_cols].dropna()
numeric_corr = pd.DataFrame(
    np.corrcoef(complete[corr_cols].to_numpy(dtype=float), rowvar=False),
    index=corr_cols,
    columns=corr_cols,
).round(3)
numeric_corr.to_csv(results_folder / 'numeric_correlations.csv')
print(f'\nPearson correlation (NumPy corrcoef) on {len(complete):,} complete rows:')
print(numeric_corr.to_string())

# Year-to-year movement of the four largest premises categories. These series share one city-wide trend.
top_types = premises.head(4).index
annual_frame = df.loc[df['REPORT_YEAR'].between(2014, 2025), ['REPORT_YEAR', 'PREMISES_TYPE']]
annual_by_premises = pd.crosstab(annual_frame['REPORT_YEAR'], annual_frame['PREMISES_TYPE']).reindex(columns=top_types, fill_value=0)
premises_corr = pd.DataFrame(
    np.corrcoef(annual_by_premises.to_numpy(dtype=float), rowvar=False),
    index=top_types,
    columns=top_types,
).round(3)
premises_corr.to_csv(results_folder / 'annual_premises_correlations.csv')
print('\nPearson correlation of annual premises-type counts (2014–2025, 12 years):')
print(premises_corr.to_string())
print('These correlations describe co-movement. They do not establish that one premises type causes another.')


# 3. MISSING DATA AND OTHER QUALITY CHECKS
print('\n=== 4. MISSING DATA AND QUALITY ===')
quality = profile[['missing']].copy()
quality['missing_pct'] = (quality['missing'] / len(df) * 100).round(3)
quality.to_csv(results_folder / 'missing_data.csv')
print(quality.to_string())

# Cross-check numeric blanks with NumPy. Object columns are evaluated with pandas.
numeric_columns = df.select_dtypes(include=np.number).columns
mismatch = []
for col in numeric_columns:
    pandas_missing = int(df[col].isna().sum())
    numpy_missing = int(np.isnan(df[col].to_numpy(dtype=float)).sum())
    if pandas_missing != numpy_missing:
        mismatch.append(col)
if mismatch:
    raise RuntimeError(f'Pandas and NumPy missing counts differ for: {mismatch}')
print('Numeric missing counts agree between pandas.isna and numpy.isnan.')

id_counts = df['EVENT_UNIQUE_ID'].value_counts()
blank_occurrence = df['OCC_YEAR'].isna()
sentinel = df.loc[blank_occurrence, ['OCC_DATE', 'EVENT_UNIQUE_ID', 'REPORT_YEAR']].copy()
sentinel['occ_date_parsed'] = occ_dt.loc[blank_occurrence].dt.date.astype(str)
sentinel_bits = []
for (date_text, event_id), group in sentinel.groupby(['occ_date_parsed', 'EVENT_UNIQUE_ID'], sort=True):
    rows = len(group)
    row_word = 'row' if rows == 1 else 'rows'
    sentinel_bits.append(f'{date_text} ({event_id}, {rows} {row_word}, reported {int(group["REPORT_YEAR"].iloc[0])})')
sentinel_summary = '; '.join(sentinel_bits)

zero_coord_rows = int((~valid_coords).sum())
highlights = {
    'n_rows': int(len(df)),
    'n_cols': int(df.shape[1]),
    'source_name': csv_path.name,
    'report_date_min': str(report_dt.min().date()),
    'report_date_max': str(report_dt.max().date()),
    'occ_date_min': str(occ_dt.min().date()),
    'occ_date_max': str(occ_dt.max().date()),
    'records_2014': int(annual.loc[2014]),
    'records_2023': int(annual.loc[2023]),
    'records_2024': int(annual.loc[2024]),
    'records_2025': int(annual.loc[2025]),
    'records_2026': int((df['REPORT_YEAR'] == 2026).sum()),
    'annual_mean': annual_mean,
    'annual_median': annual_median,
    'occ_hour_mean': float(stats.loc['OCC_HOUR', 'mean']),
    'occ_hour_median': float(stats.loc['OCC_HOUR', 'median']),
    'occ_hour_std': float(stats.loc['OCC_HOUR', 'std']),
    'occ_hour_mode': occ_hour_mode,
    'occ_hour_mode_n': int((df['OCC_HOUR'] == occ_hour_mode).sum()),
    'report_hour_mean': float(stats.loc['REPORT_HOUR', 'mean']),
    'report_hour_median': float(stats.loc['REPORT_HOUR', 'median']),
    'report_hour_mode': report_hour_mode,
    'report_hour_mode_n': int((df['REPORT_HOUR'] == report_hour_mode).sum()),
    'report_year_mean': float(stats.loc['REPORT_YEAR', 'mean']),
    'report_year_median': float(stats.loc['REPORT_YEAR', 'median']),
    'corr_rows': int(len(complete)),
    'corr_year': float(numeric_corr.loc['REPORT_YEAR', 'OCC_YEAR']),
    'corr_hour': float(numeric_corr.loc['REPORT_HOUR', 'OCC_HOUR']),
    'prem_outside_house': float(premises_corr.loc['Outside', 'House']),
    'prem_outside_apartment': float(premises_corr.loc['Outside', 'Apartment']),
    'prem_house_commercial': float(premises_corr.loc['House', 'Commercial']),
    'prem_commercial_apartment': float(premises_corr.loc['Commercial', 'Apartment']),
    'outside': int(premises['Outside']),
    'house': int(premises['House']),
    'commercial': int(premises['Commercial']),
    'apartment': int(premises['Apartment']),
    'other': int(premises['Other']),
    'transit': int(premises['Transit']),
    'educational': int(premises['Educational']),
    'top_neighbourhood': str(hood.index[0]),
    'top_neighbourhood_n': int(hood.iloc[0]),
    'duplicate_rows': int(df.duplicated().sum()),
    'distinct_events': int(df['EVENT_UNIQUE_ID'].nunique()),
    'extra_event_rows': int(df['EVENT_UNIQUE_ID'].duplicated().sum()),
    'multi_event_ids': int((id_counts > 1).sum()),
    'max_rows_per_event': int(id_counts.max()),
    'zero_coords': zero_coord_rows,
    'nsa_hood': int((df['NEIGHBOURHOOD_158'] == 'NSA').sum()),
    'nsa_div': int((df['DIVISION'] == 'NSA').sum()),
    'missing_occ': int(blank_occurrence.sum()),
    'missing_location': int(df['LOCATION_TYPE'].isna().sum()),
    'sentinel_summary': sentinel_summary,
}
pd.DataFrame({'check': list(highlights.keys()), 'value': list(highlights.values())}).to_csv(
    results_folder / 'quality_checks.csv', index=False
)
(results_folder / 'highlights.json').write_text(json.dumps(highlights, indent=2), encoding='utf-8')

print('Exact duplicate rows:', highlights['duplicate_rows'])
print('Distinct EVENT_UNIQUE_ID values:', highlights['distinct_events'])
print('Additional rows that reuse an event id:', highlights['extra_event_rows'])
print('Event ids that appear more than once:', highlights['multi_event_ids'])
print('Maximum rows for one event id:', highlights['max_rows_per_event'])
print('Rows with zero latitude and longitude:', highlights['zero_coords'])
print('NSA neighbourhood labels:', highlights['nsa_hood'])
print('NSA division labels:', highlights['nsa_div'])
print('Rows with blank occurrence year/month/day/day-of-year/day-of-week:', highlights['missing_occ'])
print('Those blank components have these OCC_DATE values:', sentinel_summary)
print('Weekday labels are padded to 10 characters. Trim them before using the field as a dimension.')


# 4. GRAPHS
print('\n=== 5. VISUALIZATIONS ===')

plt.figure(figsize=(10, 5))
sns.lineplot(x=annual.index, y=annual.to_numpy(), marker='o', color='steelblue')
plt.title('Annual Auto Theft Records in Toronto (2014–2025)')
plt.xlabel('Report year')
plt.ylabel('Number of records')
plt.xticks(annual.index, rotation=45)
save_chart('01_auto_theft_by_year.png')

plt.figure(figsize=(11, 5))
sns.countplot(data=df, x='OCC_HOUR', order=range(24), color='steelblue')
plt.title('Auto Theft Records by Occurrence Hour')
plt.xlabel('Hour of day (0–23)')
plt.ylabel('Number of records')
save_chart('02_auto_theft_by_hour.png')

plt.figure(figsize=(10, 5))
sns.barplot(x=premises.to_numpy(), y=premises.index, color='steelblue')
plt.title('Auto Theft Records by Premises Type')
plt.xlabel('Number of records')
plt.ylabel('Premises type')
save_chart('03_auto_theft_by_premises.png')

top_hood = hood.head(10)
plt.figure(figsize=(11, 6))
sns.barplot(x=top_hood.to_numpy(), y=top_hood.index, color='steelblue')
plt.title('Top 10 Neighbourhoods by Auto Theft Records (Raw Counts)')
plt.xlabel('Number of records')
plt.ylabel('Neighbourhood')
save_chart('04_auto_theft_by_neighbourhood.png')

plt.figure(figsize=(8, 6))
sns.heatmap(premises_corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Correlation of Annual Auto Theft Counts by Premises Type')
plt.xticks(rotation=25, ha='right')
plt.yticks(rotation=0)
save_chart('05_auto_theft_correlations.png')

plt.figure(figsize=(9, 7))
sns.heatmap(numeric_corr, annot=True, fmt='.3f', cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Correlation of Report and Occurrence Year and Hour')
plt.xticks(rotation=25, ha='right')
plt.yticks(rotation=0)
save_chart('06_numeric_field_correlations.png')

print('Done. Tables saved to results/ and figures saved to charts/.')
print('Means of year and hour describe the centre of those fields. They are not crime rates.')
print('Correlations are descriptive. The premises correlations use 12 annual values.')
