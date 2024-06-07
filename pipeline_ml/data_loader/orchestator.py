# data_loader/orchestrator.py
import pandas as pd
import os

class DataLoader:
    def __init__(self, filepath, live_prediction=False):
        self.filepath = filepath
        self.live_prediction = live_prediction
    def load_data(self):
        data = pd.read_csv(self.filepath)
        data_dtype_configurated = self.dtype_configuration(data)
        return data_dtype_configurated
    
    def dtype_configuration(self,data):
        try:
            for col in data.columns:
                if ('prize' in col.lower()) or ('bets' in col.lower()):
                    if "eur" in str(data[col].iloc[1]).lower():
                        data[col] = data[col].apply(lambda x: x.replace(' EUR','') if  isinstance(x, str) else x)
                        data[col] = data[col].astype('float')
                        data[col] = data[col]*10
                    else :
                        data[col] = data[col].apply(lambda x: x.replace(' MAD','') if  isinstance(x, str) else x)
                        data[col] = data[col].astype('float')
                elif 'x' in col.lower():
                    data[col] = data[col].astype('float')
                elif 'players' in col.lower() :
                    data[col] = data[col].astype('int')
                elif 'time' in col.lower():
                    data[col] = pd.to_datetime(data[col])
                else : 
                    continue
        except Exception as e:
            print(f'Error loading data, dtype_configuration : {e}')
        return data
    
    def load_multipule_csv(self):
        if self.live_prediction:
            direction = 'pipeline_ml/live_predictor/live_prediction'
        direction = 'pipeline_ml/data_loader/data'
        list_dataframe = []
        for filename in os.listdir(direction):
            if filename.endswith(".csv"):
                self.filepath = os.path.join(direction, filename)
                list_dataframe.append(self.load_data())
        merged_dataframe = pd.concat(list_dataframe)
        merged_dataframe = self.end_round_column(merged_dataframe)
        return merged_dataframe
    
    def end_round_column(self,data):
        # Ensure 'Value X' is numeric
        data['Value X'] = pd.to_numeric(data['Value X'], errors='coerce')

        # Drop rows with NaN values if any occurred during conversion
        data.dropna(subset=['Value X'], inplace=True)

        # Reset index to ensure continuous indexing
        data.reset_index(drop=True, inplace=True)

        # Re-iterate through the DataFrame to mark the end of rounds
        data['End of Round'] = False  # Re-initialize the column with False

        for i in range(1, len(data)):
            if data.loc[i, 'Value X'] < data.loc[i - 1, 'Value X']:
                data.loc[i - 1, 'End of Round'] = True  # Mark the previous row as end of a round
        return data
    
            
