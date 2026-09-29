"""
build_dataset.py
----------------
Downloads and caches every real dataset used in the paper so that all
downstream analysis runs from a fixed, dated snapshot.

    python src/build_dataset.py

Writes to data/cache/:
    us_close.csv, us_volume.csv   S&P 100 current constituents (Yahoo Finance, adjusted)
    de_close.csv, de_volume.csv   DAX 40 current constituents (Yahoo Finance, adjusted)
    index_close.csv               SPY, ^GSPC, ^GDAXI, ^VIX, ^V2TX (where available)
    usrec.csv                     NBER recession indicator (FRED series USREC)
    snapshot.txt                  download timestamp and coverage summary
"""

import os
import io
import datetime as dt
import urllib.request
import pandas as pd
import yfinance as yf

from data_loader import SP100_TICKERS

START = '2005-01-01'
END = '2026-01-01'
CACHE = os.path.join(os.path.dirname(__file__), '..', 'data', 'cache')

DAX40_TICKERS = [
    'ADS.DE', 'AIR.DE', 'ALV.DE', 'BAS.DE', 'BAYN.DE', 'BEI.DE', 'BMW.DE',
    'BNR.DE', 'CBK.DE', 'CON.DE', 'DTG.DE', 'DBK.DE', 'DB1.DE', 'DHL.DE',
    'DTE.DE', 'EOAN.DE', 'FRE.DE', 'HNR1.DE', 'HEI.DE', 'HEN3.DE', 'IFX.DE',
    'MBG.DE', 'MRK.DE', 'MTX.DE', 'MUV2.DE', 'P911.DE', 'PAH3.DE', 'QIA.DE',
    'RHM.DE', 'RWE.DE', 'SAP.DE', 'SRT3.DE', 'SIE.DE', 'ENR.DE', 'SHL.DE',
    'SY1.DE', 'VNA.DE', 'VOW3.DE', 'ZAL.DE', '1COV.DE',
]
INDEX_TICKERS = ['SPY', '^GSPC', '^GDAXI', '^VIX', '^V2TX']


def download(tickers):
    raw = yf.download(tickers, start=START, end=END, auto_adjust=True,
                      progress=False, threads=True, group_by='column')
    close = raw['Close'].dropna(how='all', axis=1)
    volume = raw['Volume'].reindex(columns=close.columns)
    return close, volume


def fred(series):
    url = f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}'
    with urllib.request.urlopen(url, timeout=30) as r:
        df = pd.read_csv(io.StringIO(r.read().decode()))
    df.columns = ['date', series]
    df['date'] = pd.to_datetime(df['date'])
    return df.set_index('date')


def main():
    os.makedirs(CACHE, exist_ok=True)
    lines = [f'Downloaded {dt.datetime.now().isoformat(timespec="seconds")}',
             f'Requested window {START} to {END}']

    # Marsh McLennan now trades as MRSH; Yahoo returns no history under MMC.
    us_tickers = ['MRSH' if t == 'MMC' else t for t in SP100_TICKERS]

    for tag, tickers in [('us', us_tickers), ('de', DAX40_TICKERS)]:
        close, volume = download(tickers)
        close.to_csv(os.path.join(CACHE, f'{tag}_close.csv'))
        volume.to_csv(os.path.join(CACHE, f'{tag}_volume.csv'))
        missing = sorted(set(tickers) - set(close.columns))
        first = close.apply(lambda s: s.first_valid_index())
        lines.append(f'{tag}: {close.shape[1]}/{len(tickers)} tickers, '
                     f'{close.index[0].date()} to {close.index[-1].date()}, '
                     f'missing={missing}, '
                     f'tickers starting after 2006: {int((first > "2006-01-01").sum())}')

    idx, _ = download(INDEX_TICKERS)
    idx.to_csv(os.path.join(CACHE, 'index_close.csv'))
    lines.append(f'indices available: {list(idx.columns)}')

    rec = fred('USREC')
    rec.to_csv(os.path.join(CACHE, 'usrec.csv'))
    lines.append(f'USREC: {rec.index[0].date()} to {rec.index[-1].date()}')

    with open(os.path.join(CACHE, 'snapshot.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
