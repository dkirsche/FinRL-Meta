from typing import List
from stockstats import StockDataFrame as Sdf

from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
import numpy as np
import pandas as pd
import pytz

try:
    import exchange_calendars as tc
except:
    print(
        "Cannot import exchange_calendars.",
        "If you are using python>=3.7, please install it.",
    )
    import trading_calendars as tc

    print("Use trading_calendars instead for alpaca processor.")
# from basic_processor import _Base
from meta.data_processors._base import _Base
from meta.data_processors._base import calc_time_zone

from meta.config import (
    TIME_ZONE_SHANGHAI,
    TIME_ZONE_USEASTERN,
    TIME_ZONE_PARIS,
    TIME_ZONE_BERLIN,
    TIME_ZONE_JAKARTA,
    TIME_ZONE_SELFDEFINED,
    USE_TIME_ZONE_SELFDEFINED,
    BINANCE_BASE_URL,
)


class Alpaca(_Base):
    def __init__(
        self,
        data_source: str,
        start_date: str,
        end_date: str,
        time_interval: str,
        **kwargs,
    ):
        super().__init__(data_source, start_date, end_date, time_interval, **kwargs)
        # no keys required for crypto data
        self.client = CryptoHistoricalDataClient()
        self.api_key =  kwargs["API_KEY"]
        self.api_secret = kwargs["API_SECRET"]

    def download_data(
        self,
        ticker_list,
        save_path: str = "./data/dataset.csv",
    ) -> pd.DataFrame:
        self.time_zone = calc_time_zone(
            ticker_list, TIME_ZONE_SELFDEFINED, USE_TIME_ZONE_SELFDEFINED
        )
        start_date = pd.Timestamp(self.start_date, tz=self.time_zone)
        end_date = pd.Timestamp(self.end_date, tz=self.time_zone) + pd.Timedelta(days=1)
        timeframe = self.time_interval
        
        request_params = CryptoBarsRequest(
                symbol_or_symbols=ticker_list,
                timeframe=timeframe,
                start=start_date,
                end=end_date
            )

        bars = self.client.get_crypto_bars(request_params)
        
        data_df = bars.df
        data_df.reset_index(inplace=True)
        print(data_df)
        
        data_df["time"] = data_df["timestamp"].apply(
            lambda x: x.strftime("%Y-%m-%d %H:%M:%S")
        )
        data_df = data_df.rename(columns={"time": "date"})
        data_df = data_df.rename(columns={"symbol": "tic"})
        self.dataframe = data_df

        self.save_data(save_path)

        print(
            f"Download complete! Dataset saved to {save_path}. \nShape of DataFrame: {self.dataframe.shape}"
        )

    def clean_data(self):
        df = self.dataframe.copy()
        tic_list = np.unique(df.tic.values)

        trading_days = self.get_trading_days(start=self.start, end=self.end)
        # produce full time index
        times = []
        for day in trading_days:
            current_time = pd.Timestamp(day + " 09:30:00").tz_localize(self.time_zone)
            for _ in range(390):
                times.append(current_time)
                current_time += pd.Timedelta(minutes=1)
        # create a new dataframe with full time series
        new_df = pd.DataFrame()
        for tic in tic_list:
            tmp_df = pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"], index=times
            )
            tic_df = df[df.tic == tic]
            for i in range(tic_df.shape[0]):
                tmp_df.loc[tic_df.iloc[i]["time"]] = tic_df.iloc[i][
                    ["open", "high", "low", "close", "volume"]
                ]

            # if the close price of the first row is NaN
            if str(tmp_df.iloc[0]["close"]) == "nan":
                print(
                    "The price of the first row for ticker ",
                    tic,
                    " is NaN. ",
                    "It will filled with the first valid price.",
                )
                for i in range(tmp_df.shape[0]):
                    if str(tmp_df.iloc[i]["close"]) != "nan":
                        first_valid_price = tmp_df.iloc[i]["close"]
                        tmp_df.iloc[0] = [
                            first_valid_price,
                            first_valid_price,
                            first_valid_price,
                            first_valid_price,
                            0.0,
                        ]
                        break
            # if the close price of the first row is still NaN (All the prices are NaN in this case)
            if str(tmp_df.iloc[0]["close"]) == "nan":
                print(
                    "Missing data for ticker: ",
                    tic,
                    " . The prices are all NaN. Fill with 0.",
                )
                tmp_df.iloc[0] = [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]

            # forward filling row by row
            for i in range(tmp_df.shape[0]):
                if str(tmp_df.iloc[i]["close"]) == "nan":
                    previous_close = tmp_df.iloc[i - 1]["close"]
                    if str(previous_close) == "nan":
                        raise ValueError
                    tmp_df.iloc[i] = [
                        previous_close,
                        previous_close,
                        previous_close,
                        previous_close,
                        0.0,
                    ]
            tmp_df = tmp_df.astype(float)
            tmp_df["tic"] = tic
            new_df = new_df.append(tmp_df)

        new_df = new_df.reset_index()
        new_df = new_df.rename(columns={"index": "time"})

        print("Data clean finished!")

        self.dataframe = new_df

    def add_technical_indicator(
         self,
         tech_indicator_list=[
             "macd",
             "boll_ub",
             "boll_lb",
             "rsi_30",
             "dx_30",
             "close_30_sma",
             "close_60_sma",
            ],
        save_path: str = "./data/dataset_tech.csv"):
       
        df = self.dataframe.copy()
        print(f"df: {df}")
        df = df.sort_values(by=["tic", "date"])
        stock = Sdf.retype(df)
        unique_ticker = stock.tic.unique()
    
        for indicator in tech_indicator_list:
            stock.get(indicator)
            # Update self.dataframe with the technical indicators added
            self.dataframe = stock
        print("Succesfully added technical indicators")
        self.save_data(save_path)

    # def add_vix(self, data):
    #     vix_df = self.download_data(["VIXY"], self.start, self.end, self.time_interval)
    #     cleaned_vix = self.clean_data(vix_df)
    #     vix = cleaned_vix[["time", "close"]]
    #     vix = vix.rename(columns={"close": "VIXY"})
    #
    #     df = data.copy()
    #     df = df.merge(vix, on="time")
    #     df = df.sort_values(["time", "tic"]).reset_index(drop=True)
    #     return df

    # def calculate_turbulence(self, data, time_period=252):
    #     # can add other market assets
    #     df = data.copy()
    #     df_price_pivot = df.pivot(index="date", columns="tic", values="close")
    #     # use returns to calculate turbulence
    #     df_price_pivot = df_price_pivot.pct_change()
    #
    #     unique_date = df.date.unique()
    #     # start after a fixed time period
    #     start = time_period
    #     turbulence_index = [0] * start
    #     # turbulence_index = [0]
    #     count = 0
    #     for i in range(start, len(unique_date)):
    #         current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
    #         # use one year rolling window to calcualte covariance
    #         hist_price = df_price_pivot[
    #             (df_price_pivot.index < unique_date[i])
    #             & (df_price_pivot.index >= unique_date[i - time_period])
    #         ]
    #         # Drop tickers which has number missing values more than the "oldest" ticker
    #         filtered_hist_price = hist_price.iloc[
    #             hist_price.isna().sum().min() :
    #         ].dropna(axis=1)
    #
    #         cov_temp = filtered_hist_price.cov()
    #         current_temp = current_price[[x for x in filtered_hist_price]] - np.mean(
    #             filtered_hist_price, axis=0
    #         )
    #         temp = current_temp.values.dot(np.linalg.pinv(cov_temp)).dot(
    #             current_temp.values.T
    #         )
    #         if temp > 0:
    #             count += 1
    #             if count > 2:
    #                 turbulence_temp = temp[0][0]
    #             else:
    #                 # avoid large outlier because of the calculation just begins
    #                 turbulence_temp = 0
    #         else:
    #             turbulence_temp = 0
    #         turbulence_index.append(turbulence_temp)
    #
    #     turbulence_index = pd.DataFrame(
    #         {"date": df_price_pivot.index, "turbulence": turbulence_index}
    #     )
    #     return turbulence_index
    #
    # def add_turbulence(self, data, time_period=252):
    #     """
    #     add turbulence index from a precalcualted dataframe
    #     :param data: (df) pandas dataframe
    #     :return: (df) pandas dataframe
    #     """
    #     df = data.copy()
    #     turbulence_index = self.calculate_turbulence(df, time_period=time_period)
    #     df = df.merge(turbulence_index, on="date")
    #     df = df.sort_values(["date", "tic"]).reset_index(drop=True)
    #     return df

    # def df_to_array(self, df, tech_indicator_list, if_vix):
    #     df = df.copy()
    #     unique_ticker = df.tic.unique()
    #     if_first_time = True
    #     for tic in unique_ticker:
    #         if if_first_time:
    #             price_array = df[df.tic == tic][["close"]].values
    #             tech_array = df[df.tic == tic][tech_indicator_list].values
    #             if if_vix:
    #                 turbulence_array = df[df.tic == tic]["VIXY"].values
    #             else:
    #                 turbulence_array = df[df.tic == tic]["turbulence"].values
    #             if_first_time = False
    #         else:
    #             price_array = np.hstack(
    #                 [price_array, df[df.tic == tic][["close"]].values]
    #             )
    #             tech_array = np.hstack(
    #                 [tech_array, df[df.tic == tic][tech_indicator_list].values]
    #             )
    #     print("Successfully transformed into array")
    #     return price_array, tech_array, turbulence_array

    def get_trading_days(self, start, end):
        nyse = tc.get_calendar("NYSE")
        df = nyse.sessions_in_range(
            pd.Timestamp(start, tz=pytz.UTC), pd.Timestamp(end, tz=pytz.UTC)
        )
        return [str(day)[:10] for day in df]

    def fetch_latest_data(
        self, ticker_list, time_interval, tech_indicator_list, limit=100
    ) -> pd.DataFrame:
        data_df = pd.DataFrame()
        for tic in ticker_list:
            barset = self.api.get_barset([tic], time_interval, limit=limit).df[tic]
            barset["tic"] = tic
            barset = barset.reset_index()
            data_df = data_df.append(barset)

        data_df = data_df.reset_index(drop=True)
        start_time = data_df.time.min()
        end_time = data_df.time.max()
        times = []
        current_time = start_time
        end = end_time + pd.Timedelta(minutes=1)
        while current_time != end:
            times.append(current_time)
            current_time += pd.Timedelta(minutes=1)

        df = data_df.copy()
        new_df = pd.DataFrame()
        for tic in ticker_list:
            tmp_df = pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"], index=times
            )
            tic_df = df[df.tic == tic]
            for i in range(tic_df.shape[0]):
                tmp_df.loc[tic_df.iloc[i]["time"]] = tic_df.iloc[i][
                    ["open", "high", "low", "close", "volume"]
                ]

                if str(tmp_df.iloc[0]["close"]) == "nan":
                    for i in range(tmp_df.shape[0]):
                        if str(tmp_df.iloc[i]["close"]) != "nan":
                            first_valid_close = tmp_df.iloc[i]["close"]
                            tmp_df.iloc[0] = [
                                first_valid_close,
                                first_valid_close,
                                first_valid_close,
                                first_valid_close,
                                0.0,
                            ]
                            break
                if str(tmp_df.iloc[0]["close"]) == "nan":
                    print(
                        "Missing data for ticker: ",
                        tic,
                        " . The prices are all NaN. Fill with 0.",
                    )
                    tmp_df.iloc[0] = [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ]

            for i in range(tmp_df.shape[0]):
                if str(tmp_df.iloc[i]["close"]) == "nan":
                    previous_close = tmp_df.iloc[i - 1]["close"]
                    if str(previous_close) == "nan":
                        raise ValueError
                    tmp_df.iloc[i] = [
                        previous_close,
                        previous_close,
                        previous_close,
                        previous_close,
                        0.0,
                    ]
            tmp_df = tmp_df.astype(float)
            tmp_df["tic"] = tic
            new_df = new_df.append(tmp_df)

        new_df = new_df.reset_index()
        new_df = new_df.rename(columns={"index": "time"})

        df = self.add_technical_indicator(new_df, tech_indicator_list)
        df["VIXY"] = 0

        price_array, tech_array, turbulence_array = self.df_to_array(
            df, tech_indicator_list, if_vix=True
        )
        latest_price = price_array[-1]
        latest_tech = tech_array[-1]
        turb_df = self.api.get_barset(["VIXY"], time_interval, limit=1).df["VIXY"]
        latest_turb = turb_df["close"].values
        return latest_price, latest_tech, latest_turb

    def get_portfolio_history(self, start, end):
        trading_days = self.get_trading_days(start, end)
        df = pd.DataFrame()
        for day in trading_days:
            df = df.append(
                self.api.get_portfolio_history(
                    date_start=day, timeframe="5Min"
                ).df.iloc[:79]
            )
        equities = df.equity.values
        cumu_returns = equities / equities[0]
        cumu_returns = cumu_returns[~np.isnan(cumu_returns)]
        return cumu_returns
