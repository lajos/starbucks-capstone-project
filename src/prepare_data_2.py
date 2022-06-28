from fnmatch import translate
import pandas as pd
import numpy as np
import time
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
        d['person id'] = self.person_id
        d['offer id'] = self.offer_id
        d['offer viewed'] = self.viewed
        d['offer completed'] = self.completed
        d['offer start date'] = self.start_date
        d['offer end date'] = self.end_date
        d['transaction count'] = self.transaction_count
        d['transaction total'] = self.transaction_total
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

    # remove users with no demographic info
    #
    # profile = profile.dropna()

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




if __name__=='__main__':
    start_time = time.time()

    main()

    duration = time.time() - start_time
    duration_mins = int(duration / 60)
    duration_secs = int(duration - (duration_mins * 60))
    print('--- run time: {}m {}s ---'.format(duration_mins, duration_secs))
