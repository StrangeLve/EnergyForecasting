import pandas as pd
from typing import Dict, List


class DataProcess:
    """
    Data Processing for Hierarchical Pandas Data Frame
    """

    def __init__(self,
                 data: pd.DataFrame,
                 meta: Dict,
                 cycle_val: int = 0,
                 col_names: List = ["temperature", "humidity", "cloudiness", "consumption"],
                 col_dtype: List = [float, float, float, float]):
        """
        :param data: Hierarchical Pandas Data Frame
        :param meta: Meta data such EV
        :param cycle_val: full cycle depends on data.index
        :param col_names: columns names
        :param col_dtype: columns data type to cast
        """
        self.data = data
        self.meta = meta
        self.cycle_val = cycle_val
        self.col_names = col_names
        self.col_dtype = col_dtype

    @staticmethod
    def create_window_index_for_agg_over(indexes: List,
                                         cycle_val: int = 0,
                                         attr_name: str = "hour"
                                         ) -> List:
        agg_index = []
        cur_index = indexes[0].date()
        for i in indexes:
            if getattr(i, attr_name) == cycle_val:
                cur_index = i.date()
            agg_index.append(cur_index)
        return agg_index

    @staticmethod
    def change_dtype(data: pd.DataFrame,
                     col_names: List,
                     col_dtypes: List) -> pd.DataFrame:
        return data.astype({col_n: col_t for col_n, col_t in zip(col_names, col_dtypes)})

    # NOTE: Might require more complicated method for duplicated dropping
    @staticmethod
    def drop_duplicates(data: pd.DataFrame, keep="last") -> pd.DataFrame:
        return data.reset_index().drop_duplicates(subset='time', keep=keep).set_index("time")

    def main(self) -> Dict:
        e_consumption_per_home = {}
        homes_indexes = self.data.columns.get_level_values(0).unique()
        for home_i in homes_indexes:
            # select single sample
            df = self.data[home_i]

            # cast dtypes
            df = DataProcess.change_dtype(df, self.col_names, self.col_dtype)

            # drop duplicates
            df = DataProcess.drop_duplicates(df)

            # create aggregate index, later on this will be used as index for grouping
            df["agg_index"] = DataProcess.create_window_index_for_agg_over(df.index, self.cycle_val)

            # drop data which does not account for 24h cycle
            not_full_cycle_index = df["agg_index"].value_counts()[df["agg_index"].value_counts() != 24].index
            df = df.drop(df["agg_index"][not_full_cycle_index])

            # create aggregate features and target
            consumption = pd.DataFrame({"consumption_(t+1)": df.groupby("agg_index")["consumption"].sum().shift(-2),
                                        "consumption_(t-1)": df.groupby("agg_index")["consumption"].sum()})

            temperature = pd.DataFrame({"avg_temperature": df.groupby("agg_index")["temperature"].mean(),
                                        "min_temperature": df.groupby("agg_index")["temperature"].min(),
                                        "max_temperature": df.groupby("agg_index")["temperature"].max(),
                                        "spread_temperature": df.groupby("agg_index")["temperature"].max()-df.groupby("agg_index")["temperature"].min()}).shift(1)

            humidity = pd.DataFrame({"avg_humidity": df.groupby("agg_index")["humidity"].mean(),
                                     "min_humidity": df.groupby("agg_index")["humidity"].min(),
                                     "max_humidity": df.groupby("agg_index")["humidity"].max(),
                                     "spread_humidity": df.groupby("agg_index")["humidity"].max()-df.groupby("agg_index")["temperature"].min()}).shift(1)

            cloudiness = pd.DataFrame({"avg_cloudiness": df.groupby("agg_index")["cloudiness"].mean(),
                                       "min_cloudiness": df.groupby("agg_index")["cloudiness"].min(),
                                       "max_cloudiness": df.groupby("agg_index")["cloudiness"].max(),
                                       "spread_cloudiness": df.groupby("agg_index")["cloudiness"].max()-df.groupby("agg_index")["temperature"].min()}).shift(1)

            data_preprocessed = pd.concat([consumption, temperature, humidity, cloudiness],
                                          axis=1).dropna()

            data_preprocessed["ev"] = 1 if self.meta[home_i]['has_electric_vehicle'] else 0

            # NOTE: home_num won't be used as feature for modelling
            data_preprocessed["home_num"] = home_i

            e_consumption_per_home[home_i] = data_preprocessed

        return e_consumption_per_home


def combine_data(dict_of_df: Dict) -> pd.DataFrame:
    df = pd.DataFrame()
    for _, data in dict_of_df.items():
        df = pd.concat([df, data])
    return df.sort_index()