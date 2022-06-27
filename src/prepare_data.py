import pandas as pd
import numpy as np
import time
import os.path


K_STATE_NO_OFFER = 0
K_STATE_OFFER_RECEIVED = 1
K_STATE_OFFER_VIEWED = 2
K_STATE_OFFER_COMPLETED = 3

class 

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

    # print(portfolio.to_dict('index'))

    # print(portfolio.loc[portfolio.id == 'ae264e3637204a6fb9bb56bc8210ddfd'])

    # remove users with no demographic info
    #
    # profile = profile.dropna()

    # fill NoN values in profile
    #
    profile.fillna(value={'gender':'U', 'income':profile.income.median()}, inplace=True)

    # merge profile info into transcipt dataframe
    #
    transcript = pd.merge(transcript, profile.rename(columns={'id':'person'}), on='person')

    print(f'number of transcript rows: {transcript.shape[0]}')
    print(f'transcripts without demographic info: {transcript.income.isna().sum()}')

    # drop transcripts with no demographic info (NaN in income or gender column)
    #
    transcript.dropna(inplace=True)
    print(f'number of transcript rows with demographic info: {transcript.shape[0]}')




if __name__=='__main__':
    start_time = time.time()

    main()

    duration = time.time() - start_time
    duration_mins = int(duration / 60)
    duration_secs = int(duration - (duration_mins * 60))
    print('--- run time: {}m {}s ---'.format(duration_mins, duration_secs))
