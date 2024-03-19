import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from alpaca.data.timeframe import TimeFrame
from meta.data_processors.alpaca import Alpaca


class TestDownloadData(unittest.TestCase):
    def setUp(self):
        # Setup
        
        TRAIN_START_DATE = '2021-09-01'
        TRAIN_END_DATE = '2021-09-03'

        TEST_START_DATE = '2021-09-21'
        TEST_END_DATE = '2021-09-30'

        TIME_INTERVAL = TimeFrame.Day
        API_KEY='PKNKP2YVRRDWMWYMCQJJ'
        API_SECRET='pCKxi3yAYn1IUeQO7Js1prfEEFbSNEWcrlu48WbD' 
        self.alpaca = Alpaca('alpaca', 
                   start_date = TRAIN_START_DATE, 
                   end_date = TRAIN_END_DATE,
                   time_interval = TIME_INTERVAL, 
                   API_KEY=API_KEY, 
                   API_SECRET=API_SECRET
                  )
    def test_download_data(self):
        # Call the function
        TICKER_LIST = ["BTC/USD", "ETH/USD"]
        self.alpaca.download_data(TICKER_LIST)

        # Assert the dataframe
        #self.assertEqual(alpaca.dataframe.shape, (10, 7))
        unique_symbols_in_dataframe = sorted(pd.unique(self.alpaca.dataframe['tic']).tolist())
        expected_symbols = sorted(TICKER_LIST)
        self.assertEqual
    
    def test_add_technical_indicator(self):
        TICKER_LIST = ["BTC/USD", "ETH/USD"]
        self.alpaca.download_data(TICKER_LIST)

        INDICATORS = ['macd', 'rsi_14', 'cci_14', 'dx_14'] #self-defined technical indicator list is NOT supported yet

        # Call the function
        self.alpaca.add_technical_indicator(INDICATORS)

        # Assert the dataframe
        assert 'macd' in self.alpaca.dataframe.columns, "The 'macd' column is not in the DataFrame"
if __name__ == '__main__':
    unittest.main()