import gzip
from io import StringIO
import pandas as pd
from datetime import datetime, timedelta

class DataLoader:
    def __init__(self):
        pass
    
    def gzip_file_to_documents_list(self, file_name, languages_filter=['en']):
        with gzip.open(file_name, 'rb') as f:
            decompressed_data = f.read()
        df_data = pd.read_csv(StringIO(str(decompressed_data,'utf-8')), index_col=0)
        documents = df_data[df_data['language'].isin(languages_filter)]['text'].tolist()
        return documents
    
    def uctd_file_name_by_date(self, month, day):
        return '0'*int(month < 10) + str(month) + '0'*int(day < 10) + str(
            day) + "_UkraineCombinedTweetsDeduped.csv.gzip"
    
    def get_uctd_documents_between_dates(self, start: str, end: str,
                                         languages_filter=['en'], verbose: int = 0):
        # Dates in the 'YYYY-DD-MM' format
        date_start = datetime.strptime(start, '%Y-%m-%d')
        date_end = datetime.strptime(end, '%Y-%m-%d')
        delta_days = date_end - date_start
        all_documents = []
        for i_d in range(delta_days.days + 1):
            date_current = date_start + timedelta(days=i_d)
            uctd_file_name = self.uctd_file_name_by_date(date_current.month, date_current.day)
            all_documents += self.gzip_file_to_documents_list(uctd_file_name, languages_filter=languages_filter)
            if verbose == 1:
                print(f'--Documents for the day {date_current.date()} processed')
        return all_documents