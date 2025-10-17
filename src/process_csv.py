from file_config_utils.file_config_reader import FileConfigReader
import pycountry
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn

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
    
def scrape_statcounter(check_exists=True):
    # Statcounter doesn't provide a dataset with all country data, so we have to scrape it.

    # Check if the scraping has already been done
    if check_exists:
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

def repair_device_tiers(check_exists=True):
    if check_exists:
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
    # Fill empty values with 0
    device_tiers.fillna(0, inplace=True)
    # Convert to int
    for col in ['High', 'Mid', 'Low', 'Unknown', 'Total']:
        device_tiers[col] = device_tiers[col].astype(int)
    device_tiers.to_csv("csv/csv_processed/device_tiers.csv", index=False)
    print("Device tiers data repaired and saved to csv/csv_processed/device_tiers.csv")

def join_internet_users_and_population(check_exists=True):
    if check_exists:
        try:
            fcr.find_path("csv/csv_processed/internet_users_population_2020.csv")
            print("Internet users and population data already saved, skipping.")
            return
        except FileNotFoundError:
            print("Internet users and population data not found, proceeding to join.")

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
    
def unflatten_device_tiers():
    try:
        fcr.find_path("csv/csv_processed/monthly_users_plan.csv")
        print("Unflattened monthly users data already saved, skipping.")
        return
    except FileNotFoundError:
        print("Unflattened monthly users data not found, proceeding to unflatten.")
    device_tiers_df = fcr.find("scif3000 [student data] - mau_by_plan_type.csv")
    # Rows are MONTH_END_DATE, PRIMARY_PLAN_TYPE, COUNTRY_CODE, Monthly Active Users
    # Turn MONTH_END_DATE into the second dimension (2020-01-31)
    unflattened = device_tiers_df.pivot_table(index=['COUNTRY_CODE', 'PRIMARY_PLAN_TYPE'], columns='MONTH_END_DATE', values='Monthly Active Users', aggfunc='sum').reset_index()
    # Flatten the columns
    unflattened.columns.name = None
    unflattened.columns = [str(col) for col in unflattened.columns]
    # Add ISO2 and ISO3 codes
    unflattened.rename(columns={'COUNTRY_CODE': 'ISO2'}, inplace=True)
    unflattened['ISO3'] = unflattened['ISO2'].apply(lambda x: transform_iso2(x, 'ISO3'))
    # Add country names
    unflattened['Country'] = unflattened['ISO2'].apply(lambda x: transform_iso2(x, 'name'))
    # Reorder columns
    cols = ['Country', 'ISO2', 'ISO3'] + [col for col in unflattened.columns if col not in ['Country', 'ISO2', 'ISO3']]
    unflattened = unflattened[cols]
    unflattened.to_csv("csv/csv_processed/monthly_users_plan.csv", index=False)
    print("Unflattened monthly users data saved to csv/csv_processed/monthly_users_plan.csv")

    # Also make a version where all the plan types are summed
    # Sum all plan types for each country
    summed = unflattened.groupby(['Country', 'ISO2', 'ISO3']).sum(numeric_only=True).reset_index()
    summed.to_csv("csv/csv_processed/monthly_users_total.csv", index=False)
    print("Summed monthly users data saved to csv/csv_processed/monthly_users_total.csv")

def make_mau_graphs():
    try:
        fcr.find_path("graphs/mau/global.png")
        print("MAU graphs already saved, skipping.")
        return
    except FileNotFoundError:
        print("MAU graphs not found, proceeding to generate.")
    # Generate a cumulative line graph of monthly active users by plan type for each country and global
    mau_df = fcr.find("monthly_users_plan.csv")
    mau_total_df = fcr.find("monthly_users_total.csv")
    plan_types = mau_df['PRIMARY_PLAN_TYPE'].unique()
    countries = mau_df['Country'].unique()
    # Save a graph for each country to graphs/mau/{country}.png
    for country in countries:
        country_df = mau_df[mau_df['Country'] == country]
        if country_df.empty:
            continue
        plt.figure(figsize=(10, 6))
        dates = [col for col in country_df.columns if col not in ['Country', 'ISO2', 'ISO3', 'PRIMARY_PLAN_TYPE']]
        cumulative_values = np.zeros(len(dates))
        for plan_type in plan_types:
            plan_df = country_df[country_df['PRIMARY_PLAN_TYPE'] == plan_type]
            if plan_df.empty:
                continue
            values = plan_df[dates].values.flatten()
            # Interpolate missing points
            interpolated_values = pd.Series(values).interpolate(method='linear', limit_direction='both').values
            cumulative_values += interpolated_values
            plt.fill_between(dates, cumulative_values - interpolated_values, cumulative_values, label=plan_type, alpha=0.6)
        plt.title(f'Cumulative Monthly Active Users by Plan Type in {country}')
        plt.xlabel('Year')
        plt.ylabel('Cumulative Monthly Active Users')
        plt.xticks(ticks=range(0, len(dates), 12), labels=[dates[i][:4] for i in range(0, len(dates), 12)], rotation=45)
        plt.legend()
        plt.gca().get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
        plt.tight_layout()
        plt.savefig(f'graphs/mau/{country}.png')
        plt.close()
        print(f"Saved cumulative mau graph for {country}.")
    # Save one for global
    plt.figure(figsize=(10, 6))
    cumulative_values = np.zeros(len(dates))
    for plan_type in plan_types:
        plan_df = mau_df[mau_df['PRIMARY_PLAN_TYPE'] == plan_type]
        if plan_df.empty:
            continue
        values = plan_df[dates].sum().values.flatten()
        cumulative_values += values
        plt.fill_between(dates, cumulative_values - values, cumulative_values, label=plan_type, alpha=0.6)
    plt.title('Global Cumulative Monthly Active Users by Plan Type')
    plt.xlabel('Year')
    plt.ylabel('Cumulative Monthly Active Users')
    plt.xticks(ticks=range(0, len(dates), 12), labels=[dates[i][:4] for i in range(0, len(dates), 12)], rotation=45)
    plt.legend()
    plt.gca().get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
    plt.tight_layout()
    plt.savefig('graphs/mau/global.png')
    plt.close()
    print("Saved global cumulative mau graph.")

def process_gdp_and_gdp_per_capita():
    try:
        fcr.find_path("csv/csv_processed/gdp_2023.csv")
        print("GDP data already saved, skipping.")
        return
    except FileNotFoundError:
        print("GDP data not found, proceeding to process.")
    gdp_df = fcr.find("gdp-worldbank-constant-usd.csv")
    gdp_per_capita_df = fcr.find("gdp-per-capita-worldbank.csv")
    # Take 2023 data
    gdp_2023 = gdp_df[['Entity', 'Code', 'GDP (constant 2015 US$)']][gdp_df['Year'] == 2023]
    gdp_per_capita_2023 = gdp_per_capita_df[['Entity', 'Code', 'GDP per capita, PPP (constant 2021 international $)']][gdp_per_capita_df['Year'] == 2023]
    gdp_2023.rename(columns={'Entity': 'Country', 'Code': 'ISO3', 'GDP (constant 2015 US$)': 'GDP'}, inplace=True)
    gdp_per_capita_2023.rename(columns={'Entity': 'Country', 'Code': 'ISO3', 'GDP per capita, PPP (constant 2021 international $)': 'GDP per Capita'}, inplace=True)
    # Merge on Country and ISO3 (drop areas with no iso3)
    gdp_2023 = gdp_2023[gdp_2023['ISO3'].notna()]
    gdp_per_capita_2023 = gdp_per_capita_2023[gdp_per_capita_2023['ISO3'].notna()]
    merged = pd.merge(gdp_2023, gdp_per_capita_2023, on=['Country', 'ISO3'], how='inner')
    merged['ISO2'] = merged['ISO3'].apply(lambda x: transform_iso3(x, 'ISO2'))
    merged = merged[['Country', 'ISO2', 'ISO3', 'GDP', 'GDP per Capita']]
    merged.to_csv("csv/csv_processed/gdp_2023.csv", index=False)
    print("GDP data processed and saved to csv/csv_processed/gdp_2023.csv")

def correlation_matrix(file_dict: dict, output_png: str, title: str, blank_diagonals=False, remove_top=False):
    # Pass in a dict of form {file_name: [value_column_names]} or {file_name: value_column_name} to correlate
    df = pd.DataFrame()
    for file_name, value_columns in file_dict.items():
        temp_df = fcr.find(file_name)
        if isinstance(value_columns, str):
            value_columns = [value_columns]  # Convert single column name to list for consistency
        temp_df = temp_df.sort_values(by="ISO2")  # Sort by ISO2 to ensure alignment
        df = pd.concat([df, temp_df[value_columns]], axis=1)
    df.columns = [col for cols in file_dict.values() for col in (cols if isinstance(cols, list) else [cols])]
    corr = df.corr()
    
    if blank_diagonals:
        np.fill_diagonal(corr.values, np.nan)  # Replace diagonal values with NaN
    
    if remove_top:
        mask = np.triu(np.ones_like(corr, dtype=bool))
        if blank_diagonals:
            mask[np.diag_indices_from(mask)] = True  # Mask the diagonal if blank_diagonals is True
        plt.figure(figsize=(10, 8))
        seaborn.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True, mask=mask)
    else:
        plt.figure(figsize=(10, 8))
        seaborn.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True, mask=corr.isnull() if blank_diagonals else None)
    
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_png)
    plt.close()

if __name__ == "__main__":
    scrape_statcounter()
    repair_device_tiers()
    join_internet_users_and_population()
    unflatten_device_tiers()
    make_mau_graphs()
    process_gdp_and_gdp_per_capita()
    correlation_matrix(
        {
            "device_tiers.csv": ["High", "Mid", "Low", "Total"],
            "gdp_2023.csv": "GDP per Capita",
            "internet_users_population_2020.csv": "Internet Penetration",
        },
        output_png="graphs/correlation_matrices/gdp_internet_penetration.png",
        title="Correlations",
        blank_diagonals=True,
        remove_top=True
    )

    # if True is easier than commenting and uncommenting to run
    if False:
        # Generate world heatmap is private so you might not have it
        # i.e. need to import it here in a try
        try:
            from heatmaps import generate_world_heatmap

            mobile_df = fcr.find("statcounter_2023.csv")
            mobile_heatmap_png = generate_world_heatmap(
                mobile_df,
                value_col = "Mobile Share 2023",
                value_name = "Mobile Share (%)",
                title = "Mobile Share by Country (%) (2023)",
                colour_scale = "Viridis",
                output_png = "graphs/heatmaps/mobile_heatmap.png",
                colour_range=(0, 100)
            )
            print(f"Mobile heatmap saved.")

            

        except ImportError as e:
            print(f"Could not import generate_world_heatmap: {e}")