import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from sklearn.preprocessing import MinMaxScaler
tqdm.pandas()

class PreprocessChangeInPrice():
    def __init__(self, csv_path: str, date_column:str, column_to_shift: str, use_scaler: bool, sequence_length: int = 120, val_percentage: float = .1, test_percentage: float = .1):
        df = pd.read_csv(csv_path, parse_dates=[date_column])
        self.df = df.sort_values(by=date_column).reset_index(drop=True)

        self.date_column = date_column
        self.column_to_shift = column_to_shift
        self.use_scaler = use_scaler
        self.sequence_length = sequence_length
        self.val_percentage = val_percentage
        self.test_percentage = test_percentage

        self.train_sequences = None
        self.val_sequences = None
        self.test_sequences = None

        self.train_df = None
        self.val_df = None
        self.test_df = None

    def create_sequences(self, input_data: pd.DataFrame, target_column, sequence_length):
            sequences = []
            data_size = len(input_data)

            for i in tqdm(range(data_size - sequence_length)):
                sequence = input_data[i:i+sequence_length]

                label_position = i + sequence_length
                label = input_data.iloc[label_position][target_column]

                sequences.append((sequence, label))

            return sequences
        
    
    def preprocess(self):
        df = self.df

        df = df.rename(columns = {'Volume BTC':'Volume'})

        # shift rows by 1 so that every row has data from day before
        df['Previous'] = df[self.column_to_shift].shift(1)

        # get how much has the price changed from day before
        print('Getting the change of price in a day...')
        df['Previous_Change'] = df.progress_apply(
            lambda row: 0 if np.isnan(row['Previous']) else row[self.column_to_shift] - row['Previous'],
            axis = 1
        )

        # shift rows by 1 so that every row has data from the next day
        df['Future'] = df[self.column_to_shift].shift(-1)

        # get how much has the price changed from the next day
        print('Getting the change of price in a day...')
        df['Future_Change'] = df.progress_apply(
            lambda row: 0 if np.isnan(row['Future']) else row['Future'] - row[self.column_to_shift],
            axis = 1
        )
        print(df.columns)

        # transform date column
        print('Transforming date column to day_of_week, day_of_month, week_of_year and year columns...')
        rows = []
        for _, row in tqdm(df.iterrows(), total=df.shape[0]):
            row_data = dict(
                day_of_week = row[self.date_column].dayofweek,
                day_of_month = row[self.date_column].day,
                week_of_year = row[self.date_column].week,
                month = row[self.date_column].month,
                year = row[self.date_column].year,
                open = row['Open'],
                high = row['High'],
                low = row['Low'],
                volume = row['Volume'],
                previous_change = row['Previous_Change'],
                future_change = row['Future_Change'],
                close = row[self.column_to_shift]
            )
            rows.append(row_data)

        features_df = pd.DataFrame(rows)

        print('Current shape after transformation of date column: ', features_df.shape)

        train_size = int(len(features_df) * (1 - self.val_percentage - self.test_percentage))
        val_size = int(len(features_df) * self.val_percentage)

        train_df = features_df[:train_size]
        val_df = features_df[train_size:train_size + val_size]
        test_df = features_df[train_size + val_size:]
    
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df


        print('Current train and test shapes: ', train_df.shape, test_df.shape)

        if self.use_scaler:
            scaler = MinMaxScaler(feature_range=(-1, 1))
            scaler = scaler.fit(train_df)

            train_df = pd.DataFrame(
                scaler.transform(train_df),
                index = train_df.index,
                columns = train_df.columns
            )

            val_df = pd.DataFrame(
                scaler.transform(val_df),
                index = val_df.index,
                columns = val_df.columns
            )

            test_df = pd.DataFrame(
                scaler.transform(test_df),
                index = test_df.index,
                columns = test_df.columns
            )

        # create sequences from pandas dataframe
        print('Creating sequences from pandas DataFrame...')
        self.train_sequences = self.create_sequences(train_df, 'future_change', self.sequence_length)
        self.val_sequences = self.create_sequences(val_df, 'future_change', self.sequence_length)
        self.test_sequences = self.create_sequences(test_df, 'future_change', self.sequence_length)
        
        print('Train shape: ', self.train_sequences[0][0].shape)
        print('Val shape: ', self.val_sequences[0][0].shape)
        print('Test shape: ', self.test_sequences[0][0].shape)