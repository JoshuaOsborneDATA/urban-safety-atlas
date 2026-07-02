import requests
import zipfile
import pandas as pd
import os

def _extract_excel(file_path: str, save_path: str) -> None:
    all_sheets = pd.read_excel(file_path, sheet_name=None)
    os.makedirs(save_path, exist_ok=True)
    for sheet_name, df in all_sheets.items():
        df.to_csv(f"{save_path}/{sheet_name}.csv", index=False)

def data_extract(url: str, save_path: str, zip_path: str = '') -> None:
    data = requests.get(url)
    os.makedirs(save_path, exist_ok=True)
    if zip_path:
        with open(zip_path, "wb") as f:
            f.write(data.content)
        zip_folder = os.path.join(save_path, os.path.basename(zip_path).rsplit(".", 1)[0])
        os.makedirs(zip_folder, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(zip_folder)
        for fname in os.listdir(zip_folder):
            if fname.endswith((".xls", ".xlsx")):
                _extract_excel(os.path.join(save_path, fname), save_path)
    else:
        ext = url.split("/")[-1].split(".")[-1].lower()
        if ext in ("xls", "xlsx"):
            _extract_excel(url, save_path)
        else:
            filename = url.split("/")[-1].rsplit(".", 1)[0] + ".csv"
            df = pd.read_csv(url)
            df.to_csv(f"{save_path}/{filename}", index=False)

root_raw_data_path = "../data/raw/"
processed_data_path = "../data/processed/"

if not os.path.exists(f"{root_raw_data_path}annual_aqi_by_county_2023"):
    zip_path = f"{root_raw_data_path}annual_aqi_by_county_2023.zip"
    url = "https://aqs.epa.gov/aqsweb/airdata/annual_aqi_by_county_2023.zip"
    data_extract(url, root_raw_data_path, zip_path=zip_path)

if not os.path.exists(f"{root_raw_data_path}FARS"):
    zip_path = f"{root_raw_data_path}FARS_2023.zip"
    url = "https://static.nhtsa.gov/nhtsa/downloads/FARS/2023/National/FARS2023NationalCSV.zip"
    data_extract(url, f"{root_raw_data_path}FARS/", zip_path=zip_path)

if not os.path.exists(f"{root_raw_data_path}SAIPE_2023"):
    url = "https://www2.census.gov/programs-surveys/saipe/datasets/2023/2023-state-and-county/est23all.txt"
    data_extract(url, f"{root_raw_data_path}SAIPE_2023", zip_path="")

if not os.path.exists(f"{root_raw_data_path}FIPS_2023"):
    url = "https://www2.census.gov/programs-surveys/popest/geographies/2023/all-geocodes-v2023.xlsx"
    data_extract(url, f"{root_raw_data_path}FIPS_2023", zip_path="")

def get_descriptive_info(file_path, **kwargs):
    is_fwf = kwargs.pop("is_fwf", False)
    if not is_fwf:
        df = pd.read_csv(file_path, **kwargs)
    else:
        df = pd.read_fwf(file_path, **kwargs)
    return df

AQI_df = get_descriptive_info(f"{root_raw_data_path}annual_aqi_by_county_2023/annual_aqi_by_county_2023.csv")
FARS_accident_df = get_descriptive_info(f"{root_raw_data_path}FARS/FARS2023NationalCSV/accident.csv")
FIPS_df = get_descriptive_info(f"{root_raw_data_path}FIPS_2023/all_geocodes_v2023.csv", skiprows=4)
PLACES_df = get_descriptive_info(f"{root_raw_data_path}PLACES_2023/PLACES__Local_Data_for_Better_Health,_County_Data_2023_release_20260629.csv")
SAIPE_df = get_descriptive_info(
    f"{root_raw_data_path}SAIPE_2023/est23all.csv",
    sep=r'\s+', header=None, is_fwf=True,
    colspecs=[(0,2),(3,6),(34,38),(133,139),(193,238),(239,241)],
    names=['state_fips','county_fips','poverty_rate','median_hhi','name','state_abbr']
)

population = (PLACES_df[['LocationID','TotalPopulation']]
              .drop_duplicates()
              .rename(columns={'LocationID':'fips','TotalPopulation':'population'}))
population['fips'] = population['fips'].astype(str).str.zfill(5)
population['population'] = population['population'].str.replace(',','').astype(int)

PLACES_df_wide = PLACES_df[PLACES_df['DataValueTypeID']=='AgeAdjPrv'].pivot_table(
    index='LocationID', columns='MeasureId', values='Data_Value', aggfunc='mean'
).reset_index()
PLACES_df_wide['fips'] = PLACES_df_wide['LocationID'].astype(str).str.zfill(5)
PLACES_df_wide.drop(columns=['LocationID'], inplace=True)

saipe = SAIPE_df[SAIPE_df['county_fips'] != 0].copy()
saipe['fips'] = saipe['state_fips'].astype(str).str.zfill(2) + saipe['county_fips'].astype(str).str.zfill(3)

FARS_accident_df['fips'] = FARS_accident_df["STATE"].astype(str).str.zfill(2) + FARS_accident_df["COUNTY"].astype(str).str.zfill(3)
accident_df = FARS_accident_df.groupby('fips').agg(total_fatalities=('FATALS','sum'), total_crashes=('ST_CASE','count'))

state_lookup = FIPS_df[FIPS_df['Summary Level']==40][['State FIPS Code','Area Name']].rename(columns={'Area Name':'State'})
FIPS_counties = FIPS_df[FIPS_df['Summary Level']==50].copy()
FIPS_counties = FIPS_counties.merge(state_lookup, on='State FIPS Code', how='left')
FIPS_counties['fips'] = FIPS_counties['State FIPS Code'].astype(str).str.zfill(2) + FIPS_counties['County FIPS Code'].astype(str).str.zfill(3)

suffixes = [' City and Borough',' Census Area',' Municipality',' Borough',' County',' parish',' Parish']
def strip_suffix(name):
    for suffix in suffixes:
        if name.endswith(suffix):
            return name[:-len(suffix)].strip()
    return name.strip()

FIPS_counties['County'] = FIPS_counties['Area Name'].apply(strip_suffix).str.lower().str.replace(r'\bst\.\s','saint ',regex=True)
FIPS_counties['State'] = FIPS_counties['State'].str.lower()
FIPS_counties['County'] = FIPS_counties['County'].str.replace('ste. genevieve','sainte genevieve')
FIPS_counties['County'] = FIPS_counties['County'].str.replace('do±a ana','dona ana')

AQI_df['County'] = AQI_df['County'].str.lower().apply(strip_suffix).str.replace(r'\bst\.\s','saint ',regex=True)
AQI_df['State'] = AQI_df['State'].str.lower()
AQI_df['County'] = AQI_df['County'].str.lower().str.replace(r'\s*\(city\)',' city',regex=True).apply(strip_suffix)
non_us = ['puerto rico','virgin islands','country of mexico']
AQI_df = AQI_df[~AQI_df['State'].isin(non_us)].copy()

AQI_FIPS_df = AQI_df.merge(FIPS_counties, on=["County","State"], how="left")
manual_patches = {
    ('missouri','sainte genevieve'):'29186',('virginia','charles'):'51036',
    ('connecticut','fairfield'):'09001',('connecticut','hartford'):'09003',
    ('connecticut','litchfield'):'09005',('connecticut','middlesex'):'09007',
    ('connecticut','new haven'):'09009',('connecticut','new london'):'09011',
    ('connecticut','tolland'):'09013',('connecticut','windham'):'09015',
}
for (state, county_clean), fips_code in manual_patches.items():
    mask = (AQI_FIPS_df['State']==state) & (AQI_FIPS_df['County']==county_clean)
    AQI_FIPS_df.loc[mask,'fips'] = fips_code

saipe_places = saipe.merge(PLACES_df_wide, on='fips', how='left')
saipe_places = saipe_places.merge(population, on='fips', how='left')
saipe_places_fars = saipe_places.merge(accident_df, on='fips', how='left')
saipe_places_fars_aqi = saipe_places_fars.merge(AQI_FIPS_df, on='fips', how='left')

cols_to_drop = ['Summary Level','State FIPS Code','County FIPS Code','County Subdivision FIPS Code','Place FIPS Code','Consolidated City FIPS Code','Area Name']
saipe_places_fars_aqi.drop(columns=cols_to_drop, inplace=True)
saipe_places_fars_aqi[['total_fatalities','total_crashes']] = saipe_places_fars_aqi[['total_fatalities','total_crashes']].fillna(0)

abbr_to_region = {
    'CT':'Northeast','ME':'Northeast','MA':'Northeast','NH':'Northeast','RI':'Northeast','VT':'Northeast','NJ':'Northeast','NY':'Northeast','PA':'Northeast',
    'IL':'Midwest','IN':'Midwest','MI':'Midwest','OH':'Midwest','WI':'Midwest','IA':'Midwest','KS':'Midwest','MN':'Midwest','MO':'Midwest','NE':'Midwest','ND':'Midwest','SD':'Midwest',
    'DE':'South','DC':'South','FL':'South','GA':'South','MD':'South','NC':'South','SC':'South','VA':'South','WV':'South','AL':'South','KY':'South','MS':'South','TN':'South','AR':'South','LA':'South','OK':'South','TX':'South',
    'AZ':'West','CO':'West','ID':'West','MT':'West','NV':'West','NM':'West','UT':'West','WY':'West','AK':'West','CA':'West','HI':'West','OR':'West','WA':'West',
}
saipe_places_fars_aqi['census_region'] = saipe_places_fars_aqi['state_abbr'].map(abbr_to_region)

os.makedirs(processed_data_path, exist_ok=True)
saipe_places_fars_aqi.to_csv(f"{processed_data_path}merged_data.csv", index=False)
print(f"Saved merged_data.csv with shape {saipe_places_fars_aqi.shape}")
