from file_config_utils.file_config_reader import FileConfigReader
import pycountry
import requests
import numpy as np
import pandas as pd
import geopandas as gpd

fcr = FileConfigReader()

def transform_iso2(iso2_code: str, output: str):
    # Returns ISO3 code or country name based on the output parameter
    try:
        country = pycountry.countries.get(alpha_2=iso2_code)
        if not country:
            return None
        if output == 'ISO3':
            return country.alpha_3
        elif output == 'name':
            return country.name
        else:
            raise ValueError("Output must be either 'ISO3' or 'name'")
    except Exception as e:
        print(f"Error transforming ISO2 code {iso2_code}: {e}")
        return None

def transform_iso3(iso3_code: str, output: str):
    # Returns ISO2 code or country name based on the output parameter
    try:
        if iso3_code == "XKX":
            # Handle Kosovo as a special case
            if output == 'ISO2':
                return "XK"
            elif output == 'name':
                return "Kosovo"
            else:
                raise ValueError("Output must be either 'ISO2' or 'name'")
        
        country = pycountry.countries.get(alpha_3=iso3_code)
        if not country:
            return None
        if output == 'ISO2':
            return country.alpha_2
        elif output == 'name':
            return country.name
        else:
            raise ValueError("Output must be either 'ISO2' or 'name'")
    except Exception as e:
        print(f"Error transforming ISO3 code {iso3_code}: {e}")
        return None

def transform_country_name(country_name: str, output: str):
    # Returns ISO2 or ISO3 code based on the output parameter
    try:
        country = pycountry.countries.get(name=country_name)
        if not country:
            # Try common name lookup
            country = next((c for c in pycountry.countries if c.name.lower() == country_name.lower()), None)
        if not country:
            return None
        if output == 'ISO2':
            return country.alpha_2
        elif output == 'ISO3':
            return country.alpha_3
        else:
            raise ValueError("Output must be either 'ISO2' or 'ISO3'")
    except Exception as e:
        print(f"Error transforming country name {country_name}: {e}")
        return None
    
def scrape_statcounter():
    # Statcounter doesn't provide a dataset with all country data, so we have to scrape it.

    # Check if the scraping has already been done
    try:
        fcr.find_path("csv/csv_processed/statcounter_2023.csv")
        print("Statcounter data already saved, skipping.")
        return
    except FileNotFoundError:
        print("Statcounter data not found, proceeding to scrape.")

    base_url = "https://gs.statcounter.com/platform-market-share/desktop-mobile/{country_code}/chart.php"
    params = {
        "bar": "1",
        "device": "Desktop & Mobile",
        "device_hidden": "desktop+mobile",
        "multi-device": "true",
        "statType_hidden": "comparison",
        "region_hidden": "",
        "granularity": "yearly",
        "statType": "Platform Comparison",
        "region": "",
        "fromInt": "2023",
        "toInt": "2023",
        "fromYear": "2023",
        "toYear": "2023",
        "csv": "1"
    }
    results = []
    for country in pycountry.countries:
        country_code = country.alpha_2
        params["region_hidden"] = country_code
        params["region"] = country.name
        print(f"Fetching data for country: {country.name} ({country_code})")
        try:
            response = requests.get(base_url.format(country_code=country_code), params=params)
            print(f"URL: {response.url}")
            response.raise_for_status()
            lines = response.text.splitlines()
            print(f"Response lines for {country.name}: {lines}")
            if len(lines) < 3:
                print(f"Unexpected data format for country {country.name} ({country_code})")
                continue
            # Request provides mobile share on line 2 and desktop share on line 3 (line 1 is header)
            mobile_share = float(lines[1].split(",")[1])
            desktop_share = float(lines[2].split(",")[1])
            print(f"Parsed data for {country.name}: Mobile Share = {mobile_share}, Desktop Share = {desktop_share}")
            results.append({
                "Country": country.name,
                "ISO2": country.alpha_2,
                "ISO3": country.alpha_3,
                "Mobile Share 2023": mobile_share,
                "Desktop Share 2023": desktop_share
            })
        except Exception as e:
            print(f"Error fetching data for country {country.name} ({country_code}): {e}")
            continue
    df = pd.DataFrame(results)
    print(f"Final DataFrame:\n{df}")
    df.to_csv("csv/csv_processed/statcounter_2023.csv", index=False)
    print("Data saved to csv/csv_processed/statcounter_2023.csv")

def repair_device_tiers():
    try:
        fcr.find_path("csv/csv_processed/device_tiers.csv")
        print("Device tiers data already saved, skipping.")
        return
    except FileNotFoundError:
        print("Device tiers data not found, proceeding to repair.")

    canva_device_tiers_name = "scif3000 [student data] - device tiers.csv"

    # Canva uses some weird country names so we need to fix them
    # Also pycountry uses verbose names for some countries
    # Print the difference between pycountry and the device_tiers.csv country names
    device_tiers = fcr.find(canva_device_tiers_name)
    device_tiers_countries = set(device_tiers["Country"].unique())
    pycountry_countries = set([country.name for country in pycountry.countries])
    difference = device_tiers_countries - pycountry_countries
    print(f"Countries in device tiers not in pycountry: {difference}")
    print(f"Countries in pycountry not in device tiers: {pycountry_countries - device_tiers_countries}")

    # These are the equivalents I could find
    country_repair = {
        "Syria": "Syrian Arab Republic",
        "Micronesia (Federated States of)": "Micronesia, Federated States of",
        "Falkland Islands": "Falkland Islands (Malvinas)",
        "Swaziland": "Eswatini",
        "Republic of the Congo": "Congo",
        "Turkey": "Türkiye",
        "Svalbard Jan Mayen": "Svalbard and Jan Mayen",
        "Guinea Bissau": "Guinea-Bissau",
        "Venezuela": "Venezuela, Bolivarian Republic of",
        "Democratic Republic of the Congo": "Congo, The Democratic Republic of the",
        "United States of America": "United States",
        "Czech Republic": "Czechia",
        "Macedonia": "North Macedonia",
        "South Korea": "Korea, Republic of",
        "Ivory Coast": "Côte d'Ivoire",
        "Virgin Islands (U.S.)": "Virgin Islands, U.S.",
        "Bolivia": "Bolivia, Plurinational State of",
        "West Bank": "Palestine, State of",
        "Virgin Islands (British)": "Virgin Islands, British",
        "Brunei": "Brunei Darussalam",
        "Russia": "Russian Federation",
        "Moldova": "Moldova, Republic of",
        "Republic of Serbia": "Serbia",
        "Bonaire Sint Eustatius Saba": "Bonaire, Sint Eustatius and Saba",
        "Laos": "Lao People's Democratic Republic",
        "Taiwan": "Taiwan, Province of China",
        "United Republic of Tanzania": "Tanzania, United Republic of",
        "Vietnam": "Viet Nam",
        "The Bahamas": "Bahamas",
        "Holy See": "Holy See (Vatican City State)",
        "East Timor": "Timor-Leste",
        "Iran": "Iran, Islamic Republic of"
    }

    # Apply the country repairs
    device_tiers['Country'] = device_tiers['Country'].replace(country_repair)

    # Print the remaining differences for posterity
    device_tiers_countries = set(device_tiers["Country"].unique())
    difference = device_tiers_countries - pycountry_countries
    print(f"Remaining countries in device tiers not in pycountry after repair: {difference}")
    print(f"Countries in pycountry not in device tiers after repair: {pycountry_countries - device_tiers_countries}")

    # Add kosovo's stats onto serbia
    kosovo_stats = device_tiers[device_tiers['Country'] == 'Kosovo']
    if not kosovo_stats.empty:
        # rows High, Mid, Low, Unknown, Total
        serbia_stats = device_tiers[device_tiers['Country'] == 'Serbia']
        if not serbia_stats.empty:
            for tier in ['High', 'Mid', 'Low', 'Unknown', 'Total']:
                serbia_value = serbia_stats.iloc[0][tier]
                kosovo_value = kosovo_stats.iloc[0][tier]
                if pd.isna(serbia_value):
                    serbia_value = 0
                if pd.isna(kosovo_value):
                    kosovo_value = 0
                device_tiers.loc[device_tiers['Country'] == 'Serbia', tier] = serbia_value + kosovo_value
            # Drop kosovo row
            device_tiers = device_tiers[device_tiers['Country'] != 'Kosovo']
    
    # Add ISO2 and ISO3 codes
    device_tiers['ISO2'] = device_tiers['Country'].apply(lambda x: transform_country_name(x, 'ISO2'))
    device_tiers['ISO3'] = device_tiers['Country'].apply(lambda x: transform_country_name(x, 'ISO3'))
    # Reorder columns
    device_tiers = device_tiers[['Country', 'ISO2', 'ISO3', 'High', 'Mid', 'Low', 'Unknown', 'Total']]
    device_tiers.to_csv("csv/csv_processed/device_tiers.csv", index=False)
    print("Device tiers data repaired and saved to csv/csv_processed/device_tiers.csv")

def join_internet_users_and_population():
    # try:
    #     fcr.find_path("csv/csv_processed/internet_users_population_2020.csv")
    #     print("Internet users and population data already saved, skipping.")
    #     return
    # except FileNotFoundError:
    #     print("Internet users and population data not found, proceeding to join.")

    # Seems like 2020 is the most recent year for internet users so we must join on that 
    internet_users = fcr.find("internet_users.csv")
    population = fcr.find("population.csv")

    # Entity and Code are the same on both, take year = 2020
    internet_users_2020 = internet_users[internet_users['Year'] == 2020][['Entity', 'Code', 'Number of Internet users']]
    population_2020 = population[population['Year'] == 2020][['Entity', 'Code', 'Population (historical)']]
    # No iso2 = drop
    internet_users_2020 = internet_users_2020[internet_users_2020['Code'].notna()]
    population_2020 = population_2020[population_2020['Code'].notna()]
    merged = pd.merge(internet_users_2020, population_2020, on=['Entity', 'Code'], how='inner')
    merged.rename(columns={'Number of Internet users': 'Internet Users', 'Population (historical)': 'Population', 'Code': 'ISO3', 'Entity': 'Country'}, inplace=True)
    merged['ISO2'] = merged['ISO3'].apply(lambda x: transform_iso3(x, 'ISO2'))
    merged['Internet Penetration'] = merged['Internet Users'] / merged['Population']
    
    print(merged.head())
    # Rearrange to be nice
    merged = merged[['Country', 'ISO2', 'ISO3', 'Internet Penetration', 'Internet Users', 'Population']]
    merged.to_csv("csv/csv_processed/internet_users_population_2020.csv", index=False)

    # Find out which countries are missing
    all_countries = set([country.alpha_3 for country in pycountry.countries])
    merged_countries = set(merged['ISO3'].unique())
    missing_countries = all_countries - merged_countries
    # print names
    missing_country_names = [transform_iso3(iso3, 'name') for iso3 in missing_countries]
    print(f"Countries missing from internet users and population data: {missing_country_names}")
    
if __name__ == "__main__":
    scrape_statcounter()
    repair_device_tiers()
    join_internet_users_and_population()