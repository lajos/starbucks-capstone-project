import pandas as pd
import numpy as np
import time
import os.path
import sys

K_DEBUG = True
def debug_print(text):
    if K_DEBUG:
        print(text)

def exit_with_error(text):
    print(text)
    sys.exit(1)

class StateRecord:
    def __init__(self):
        self.state = None
        self.person = None
        self.offer_id = None
        self.expiration = None

    def __repr__(self):
        return f'state:{self.state}, person:{self.person}, offer:{self.offer_id}, exp:{self.expiration}'

class OfferParser:
    K_STATE_NO_OFFER = 0
    K_STATE_OFFER_RECEIVED = 1
    K_STATE_OFFER_VIEWED = 2
    K_STATE_OFFER_COMPLETED = 3

    def __init__(self, portfolio):
        self.states = {}
        self.transactions = []
        self.portfolio = portfolio

    def set_state_offer_received(self, record):
        debug_print(f'{sys._getframe().f_code.co_name}: {record.to_dict()}')
        if record.person not in self.states:
            s = StateRecord()
            s.state = self.K_STATE_OFFER_RECEIVED
            s.person = record.person
            s.offer_id = record.value['offer id']
            offer_info = self.portfolio.loc[self.portfolio['id'] == s.offer_id]
            s.expiration = record.time + offer_info.duration
            self.states[record.person] = s
        else:

    def set_state_offer_viewed(self, record):
        debug_print(f'{sys._getframe().f_code.co_name}: {record.to_dict()}')
        # if record.person not in self.states.keys():
        #     exit_with_error(f'no state set for person: {record.person}')

    def set_state_offer_completed(self, record):
        debug_print(f'{sys._getframe().f_code.co_name}: {record.to_dict()}')

    def record_transaction(self, record):
        debug_print(f'{sys._getframe().f_code.co_name}: {record.to_dict()}')

    def set_state_from_transcript_record(self, record):
        if record.event == 'offer received':
            self.set_state_offer_received(record)
        elif record.event == 'offer viewed':
            self.set_state_offer_viewed(record)
        elif record.event == 'offer completed':
            self.set_state_offer_completed(record)
        elif record.event == 'transaction':
            self.record_transaction(record)
        else:
            exit_with_error(f'unknown transcript event: {record.event}')


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
    transcript = pd.merge(transcript, profile.rename(columns={'id':'person'}), on='person')

    # convert transcript time from hours to days, to match portfolio offer duration units
    #
    transcript.time /= 24

    # sort transcript by time
    #
    transcript.sort_values(by=['time'], inplace=True)

    print(f'number of transcript rows: {transcript.shape[0]}')
    print(f'transcripts without demographic info: {transcript.income.isna().sum()}')

    # drop transcripts with no demographic info (NaN in income or gender column)
    #
    transcript.dropna(inplace=True)
    print(f'number of transcript rows with demographic info: {transcript.shape[0]}')

    offer_parser = OfferParser(portfolio)

    # for i in range(len(transcript)):
    for i in range(100):
        # print(transcript.iloc[i])
        offer_parser.set_state_from_transcript_record(transcript.iloc[i])

    # print(offer_parser.states)


if __name__=='__main__':
    start_time = time.time()

    main()

    duration = time.time() - start_time
    duration_mins = int(duration / 60)
    duration_secs = int(duration - (duration_mins * 60))
    print('--- run time: {}m {}s ---'.format(duration_mins, duration_secs))
