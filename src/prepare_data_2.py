from fnmatch import translate
import pandas as pd
import numpy as np
import datetime, time
import os.path
import sys
from tqdm import tqdm
# from tqdm.notebook import tqdm


K_DEBUG = True
def debug_print(text):
    if K_DEBUG:
        print(text)


def exit_with_error(text):
    print(text)
    sys.exit(1)


class OfferRecord:
    def __init__(self, person_id, offer_id, date, duration):
        self.person_id = person_id
        self.offer_id = offer_id
        self.start_date = date
        self.end_date = self.start_date + duration
        self.viewed = False
        self.completed = False
        self.transaction_count = 0
        self.transaction_total = 0

    def as_dict(self):
        d = {}
        d['person_id'] = self.person_id
        d['offer_id'] = self.offer_id
        d['offer_viewed'] = self.viewed
        d['offer_completed'] = self.completed
        d['offer_start_date'] = self.start_date
        d['offer_end_date'] = self.end_date
        d['transaction_count'] = self.transaction_count
        d['transaction_total'] = self.transaction_total
        return(d)

    def __str__(self):
        return(f'{self.person_id} {self.offer_id} {self.start_date} {self.end_date} {self.viewed} {self.completed} {self.transaction_count} {self.transaction_total}')

def parse_offer_records(customer_transcript, portfolio):
    offer_record = None
    offer_records = []
    for i in range(len(customer_transcript)):
        r = customer_transcript.iloc[i]
        if r.event == 'offer received':
            if offer_record is not None:
                offer_records.append(offer_record)
            offer_id = r.value['offer id']
            offer_duration = portfolio.loc[portfolio.id==offer_id].duration.item()
            offer_record = OfferRecord(r.person, offer_id, r.time, offer_duration)
        elif r.event == 'offer viewed':
            if r.value['offer id'] != offer_record.offer_id:
                exit_with_error(f'offer id mismatch (viewed): person:{offer_record.person_id} {offer_record.offer_id} {r.value["offer id"]}')
            offer_record.viewed = True
        elif r.event == 'offer completed':
            if offer_record.viewed and r.value['offer_id'] == offer_record.offer_id:  # !!! possible to have an offer complete record after new offer received
                offer_record.completed = True
        elif r.event == 'transaction':
            if offer_record and offer_record.viewed and r.time<=offer_record.end_date:   # !!! make sure offer is not expired
                offer_record.transaction_count += 1
                offer_record.transaction_total += r.value['amount']
        
    if offer_record is not None:
        offer_records.append(offer_record)
            
    return offer_records


def convert_date_to_day(x):
    x_str = str(x)
    from_date = datetime.date(2013, 7, 29)
    x_date = datetime.date(int(x_str[0:4]), int(x_str[4:6]), int(x_str[6:]))
    delta = x_date - from_date
    return delta.days


def create_dummy_columns(df, column_name, drop_column=False, prefix=None, prefix_sep='_'):
    if prefix is None:
        prefix = column_name
    print(f'creating dummies for category column: {column_name}')
    # df_category_columns = pd.concat([df_category_columns, df[i]])
    # df_category_columns[i] = df[i]
    df = pd.concat([df if drop_column==False else df.drop(column_name, axis=1), 
                   pd.get_dummies(df[column_name], prefix=prefix, prefix_sep=prefix_sep)], 
                   axis=1)   
    return df 

def main():
    if not os.path.exists('data/portfolio.pkl'):
        portfolio = pd.read_json('data/portfolio.json', orient='records', lines=True)
        profile = pd.read_json('data/profile.json', orient='records', lines=True)
        transcript = pd.read_json('data/transcript.json', orient='records', lines=True)

        portfolio.to_pickle('data/portfolio.pkl')
        profile.to_pickle('data/profile.pkl')
        transcript.to_pickle('data/transcript.pkl')

    portfolio = pd.read_pickle('data/portfolio.pkl')
    profile = pd.read_pickle('data/profile.pkl')
    transcript = pd.read_pickle('data/transcript.pkl')

    # remove users with no demographic info (gender and income are none, age is set to 118)
    #
    profile = profile.dropna()

    # fill NoN values in profile
    #
    # profile.fillna(value={'gender':'U', 'income':profile.income.median()}, inplace=True)

    # merge profile info into transcipt dataframe
    #
    # transcript = pd.merge(transcript, profile.rename(columns={'id':'person'}), on='person')

    # convert transcript time from hours to days, to match portfolio offer duration units
    #
    transcript.time /= 24

    # sort transcript by time
    #
    # transcript.sort_values(by=['time'], inplace=True)

    print(f'number of transcript rows: {transcript.shape[0]}')
    # print(f'transcripts without demographic info: {transcript.income.isna().sum()}')

    # drop transcripts with no demographic info (NaN in income or gender column)
    #
    # transcript.dropna(inplace=True)
    # print(f'number of transcript rows with demographic info: {transcript.shape[0]}')


    if not os.path.exists('data/offers.pkl'):
        offer_records = []

        for i in tqdm(range(len(profile))):
        # for i in tqdm(range(100)):
            customer = profile.iloc[i]
            customer_id = customer.id

            customer_transcript = transcript.loc[transcript.person==customer_id]

            for o in parse_offer_records(customer_transcript, portfolio):
                offer_records.append(o.as_dict())

        # print(offer_records)

        # for o in offer_records:
        #     print(o)

        print(f'number of offer records parsed: {len(offer_records)}')

        offers = pd.DataFrame.from_records(offer_records)

        offers.to_pickle('data/offers.pkl')

    offers = pd.read_pickle('data/offers.pkl')

    print(f'number of offer records: {offers.shape[0]}')

    # convert date integers to days
    #
    profile['membership_start_day'] = profile['became_member_on'].apply(convert_date_to_day)

    # print(profile.describe())
    # print(profile.became_member_on.min())

    # create age bins
    #
    profile['bins_age']=pd.cut(x=profile.age, 
                               bins=[10,20,30,40,50,60,70,80,90,100,110], 
                               labels=['_10s','_20s','_30s','_40s','_50s','_60s','_70s','_80s','_90s','100s'])

    print(profile.bins_age.value_counts().sort_index())

    # create income bins
    #
    profile['bins_income']=pd.cut(x=profile.income, 
                                  bins=[25000,50000,75000,100000,125000], 
                                  labels=['25k+','50k+','75k+','100k+'])

    print(profile.bins_income.value_counts().sort_index())

    # create dummy columns for income and age categories
    #
    profile = create_dummy_columns(profile, 'bins_income', prefix='income')
    profile = create_dummy_columns(profile, 'bins_age', prefix='age')

    # rename id -> person_id to match the offers DataFrame
    #
    profile.rename(columns={'id':'person_id'}, inplace=True)

    print(profile.head())
    
    # make dummies for offer types
    # 
    portfolio = create_dummy_columns(portfolio, 'offer_type')
                                
    # make dummies for channels
    #
    portfolio = pd.concat([portfolio,
                          pd.get_dummies(portfolio.channels.apply(pd.Series).stack()).sum(level=0).add_prefix('channel_')],
                          axis=1)

    # rename id -> offer_id to match the offers DataFrame
    #
    portfolio.rename(columns={'id':'offer_id'}, inplace=True)

    print(portfolio.head())


    offers = pd.merge(offers, profile, on='person_id')
    offers = pd.merge(offers, portfolio, on='offer_id')

    print(offers.head())
    print(offers.isna().sum())
    print(f'offers shape: {offers.shape}')

if __name__=='__main__':
    start_time = time.time()

    main()

    duration = time.time() - start_time
    duration_mins = int(duration / 60)
    duration_secs = int(duration - (duration_mins * 60))
    print('--- run time: {}m {}s ---'.format(duration_mins, duration_secs))
