import os
import requests
import glob
from pathlib import Path
import numpy as np
import pandas as pd
from uk_covid19 import Cov19API
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

# Set a random seed for reproducibility
np.random.seed()

# If you're using tqdm with Pandas, enable it like this.
tqdm.pandas()


class LoadData:
    """
    A class used to represent the Data Loader for COVID-19 UK data

    ...

    Attributes
    ----------
    source_data : str
        a string indicating the path where the CSVs will be stored

    Methods
    -------
    get_uk_df(area_type="utla", filename=None, min_confirmed=0):
        Gets the UK COVID data from the NHS API and returns it as a dataframe.
    merge_dataframes(covid_data_path, covid_graph_path, population_data_path, output_file_name, column):
        Merges different dataframes containing COVID data and saves the result.
    download_data(urls):
        Downloads data from a list of URLs and saves them as CSV files.
    combine_csvs():
        Combines multiple CSV files from a directory into a single dataframe.
    """

    def __init__(self, source_data_path="source_data"):
        """
        Constructs all the necessary attributes for the LoadData object.

        :param source_data_path: str, optional
            Directory path for saving CSV files
        """
        self.source_data = Path(source_data_path)
        self.source_data.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def get_uk_df(area_type="utla", filename=None, min_confirmed=0) -> pd.DataFrame:
        """
        Get the data from NHS API. This should return a dataframe.

        :param area_type: str, optional
            Specific area type for the data, default is "utla" for Upper Tier Local Authorities.
        :param filename: str, optional
            If provided, the function will save the dataframe to this file.
        :param min_confirmed: int, optional
            Minimum confirmed cases to filter the data.
        :return: pd.DataFrame
            Dataframe containing the COVID data.
        """
        all_nations = [f"areaType={area_type}"]

        cases_and_deaths = {
            "areaName": "areaName",
            "date": "date", 
            "areaCode": "areaCode",
            "dailyCases": "newCasesBySpecimenDate",
            "cumulativeCases": "cumCasesBySpecimenDate",
        }

        api = Cov19API(filters=all_nations, structure=cases_and_deaths)
        df = api.get_dataframe()

        if df.empty:
            print("No data retrieved from the API")
            return df

        df["date"] = pd.to_datetime(df["date"]).dt.date
        df.fillna(value={"dailyCases": 0}, inplace=True, downcast="int64")

        if not df['date'].sort_values().equals(df['date']):
            print("Warning: Dates are not in order. The daily data might be incorrect.")

        # Uncomment the next line if you want to filter based on min_confirmed
        # df = df[df.dailyCases > min_confirmed]
        df = df.reset_index(drop=True)

        if filename:
            df.to_csv(filename, index=False)

        return df

    @staticmethod
    def merge_dataframes(covid_data_path, covid_graph_path, population_data_path, output_file_name, column):
        """
        This function merges three dataframes: covid_data, covid_graph, and population_data.
        The merged dataframe is then saved to a CSV file.

        Parameters:
        covid_data_path (str): The file path of the covid data CSV file.
        covid_graph_path (str): The file path of the covid graph CSV file.
        population_data_path (str): The file path of the population data CSV file.
        output_file_name (str): The name of the output CSV file.
        """

        # Load the dataframes from the CSV files
        covid_data = pd.read_csv(covid_data_path)
        covid_graph = pd.read_csv(covid_graph_path)
        population_data = pd.read_csv(population_data_path)

        # Rename the relevant columns in population_data and covid_graph to match the columns in covid_data
        population_data = population_data.rename(
            columns={"Code": "areaCode", "Mid-2021": "population"}
        )
        covid_graph = covid_graph.rename(
            columns={column: "areaCode", "LONG": "long", "LAT": "lat"}
        )

        # Merge the dataframes
        merged_df = pd.merge(
            covid_data,
            population_data[["areaCode", "population"]],
            on="areaCode",
            how="left",
        )
        merged_df = pd.merge(
            merged_df, covid_graph[["areaCode", "long", "lat"]], on="areaCode", how="left"
        )


        # Save the merged dataframe to a CSV file
        merged_df.to_csv(output_file_name, index=False)

    def download_data(self, urls):
        """
        Downloads data from a list of URLs and saves them as CSV files.

        :param urls: list of str
            List of URL strings where the data will be downloaded from.
        """
        for url in urls:
            response = requests.get(url)

            if response.status_code == 200:
                content = response.content
                directory = self.source_data/"raw/NHS_region"
                directory.mkdir(parents=True, exist_ok=True)

                # Use a timestamp for unique filenames
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                filename = directory / f"covid_data_{timestamp}.csv"
                with open(filename, "wb") as csv_file:
                    csv_file.write(content)
                print(f"Data from URL saved as '{filename}'")
            else:
                print(f"Failed to retrieve data from URL")


    def combine_csvs(self):
        """
        Combines different CSVs from the specified source_data directory into a single DataFrame.

        :return: pd.DataFrame
            Combined data from all CSVs in one DataFrame.
        """
        csv_files = glob.glob(str(self.source_data / 'raw/NHS_region/covid_data*.csv'))
        dfs = [pd.read_csv(file) for file in csv_files]
        combined_df = pd.concat(dfs, ignore_index=True)
        # Drop potential duplicates
        combined_df.drop_duplicates(subset=["date", "areaCode"], inplace=True)
        return combined_df


# if __name__ == "__main__":
#     # Define your URLs here
# #     

# #     data_loader = LoadData()

# #     # Use the methods from data_loader as required, for example:
# #     # data_loader.download_data(urls)
# #     # combined_df = data_loader.combine_csvs()
# #     # print(combined_df.head())
