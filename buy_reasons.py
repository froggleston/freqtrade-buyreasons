#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
buy_reasons.py

Usage:

buy_reasons.py -c <config.json> -s <strategy_name> -t <timerange> -g<[0,1,2,3,4]> [-l <path_to_data_dir>]

A script to parse freqtrade backtesting trades and display them with their buy_tag and sell_reason

Author: froggleston [https://github.com/froggleston]
Licence: MIT [https://github.com/froggleston/freqtrade-buyreasons/blob/main/LICENSE]

Donations:
    BTC: bc1qxdfju58lgrxscrcfgntfufx5j7xqxpdufwm9pv
    ETH: 0x581365Cff1285164E6803C4De37C37BbEaF9E5Bb
"""
import logging
import json, os
from pathlib import Path

from freqtrade.configuration import Configuration, TimeRange
from freqtrade.data.btanalysis import load_trades_from_db, load_backtest_data, load_backtest_stats
from freqtrade.data.history import load_pair_history
from freqtrade.data.dataprovider import DataProvider
from freqtrade.plugins.pairlistmanager import PairListManager
from freqtrade.exceptions import ExchangeError, OperationalException
from freqtrade.exchange import Exchange
from freqtrade.resolvers import ExchangeResolver, StrategyResolver

from joblib import Parallel, delayed, dump, load, wrap_non_picklable_objects

import numpy as np
import pandas as pd

import copy
import threading
from multiprocessing import get_context, Pool, cpu_count
import time

from tabulate import tabulate
import argparse

import concurrent.futures as fut
from functools import reduce

logger = logging.getLogger(__name__)
logger.setLevel("INFO")

def load_candles(pairlist, timerange, data_location, timeframe="5m", data_format="json"):
    all_candles=dict()
    print(f'Loading all candle data...')

    for pair in pairlist:
        if timerange is not None:
            ptr = TimeRange.parse_timerange(timerange)
            candles = load_pair_history(datadir=data_location,
                                        timeframe=timeframe,
                                        timerange=ptr,
                                        pair=pair,
                                        data_format = data_format,
                                        )
        else:
            candles = load_pair_history(datadir=data_location,
                                        timeframe=timeframe,
                                        pair=pair,
                                        data_format = data_format,
                                        )
        all_candles[pair] = candles
    print("done", end="\r")
    return all_candles

def do_analysis(strategy, pair, candles, trades, verbose=False, rk_tags=False, alt_tag='buy'):
    start_time = time.perf_counter()
    df = strategy.analyze_ticker(candles, {'pair': pair})
    end_time = time.perf_counter()

    print(f"Analysis elapsed time: {(end_time - start_time)/60}")

    # if verbose:
    print(f"Generated {df['buy'].sum()} buy / {df['sell'].sum()} sell signals")

    #if (pair in ["BAND/USDT","BAT/USDT","ROSE/USDT","GRT/USDT","SNX/USDT","CELO/USDT"]):
    #    print(pair, df.loc[df['buy'] == 1, ['buy_tag', 'msq_normabs']])

    data = df.set_index('date', drop=False)

    start_time = time.perf_counter()
    tb = do_trade_buys(pair, data, trades, rk_tags, alt_tag)
    end_time = time.perf_counter()

    print(f"Trade buys elapsed time: {(end_time - start_time)/60}")

    return tb

def do_trade_buys(pair, data, trades, rk_tags=False, alt_tag="buy"):
    trades_red = trades.loc[trades['pair'] == pair].copy()

    trades_inds = pd.DataFrame()

    if trades_red.shape[0] > 0:
        bg = r"^" + alt_tag
        sg = r"sell"

        buyf = data[data.filter(regex=bg, axis=1).values==1]

        if buyf.shape[0] > 0:
            for t, v in trades_red.open_date.items():
                allinds = buyf.loc[(buyf['date'] < v)]

                trade_inds = allinds.iloc[[-1]]

                trades_red.loc[t, 'signal_date'] = trade_inds['date'].values[0]

                bt = allinds.iloc[-1].filter(regex=bg, axis=0)

                bt.dropna(inplace=True)
                bt.drop(f"{alt_tag}", inplace=True)

                if (bt.shape[0] > 0):
                    if rk_tags:
                        trades_red.loc[t, 'buy_reason'] = bt.index.values[0]
                    else:
                        trades_red.loc[t, 'buy_reason'] = trades_red.loc[t, 'buy_tag']

                trade_inds.index.rename('signal_date', inplace=True)
                trades_inds = trades_inds.append(trade_inds)

        cancelf = data[data.filter(regex=r'^cancel', axis=1).values==1]
        if cancelf.shape[0] > 0:
            for t, v in trades_red.open_date.items():
                bt = cancelf.loc[(cancelf['date'] < v)].iloc[-1].filter(regex=r'^cancel', axis=0)
                bt.dropna(inplace=True)
                #bt.drop("buy", inplace=True)
                if (bt.shape[0] > 0):
                    trades_red.loc[t, 'cancel_reason'] = bt.index.values[0]

        ## comment in if you're doing plotting and want to show the values in the sell_reason
            # trades_red.loc[t, 'sell_reason'] = f"{bt.index.values[0]} / {trades_red.loc[t, 'sell_reason']}"

        trades_red['signal_date'] = pd.to_datetime(trades_red['signal_date'], utc=True)

        trades_red.set_index('signal_date', inplace=True)

        try:
            trades_red = pd.merge(trades_red, trades_inds, on='signal_date', how='outer')
        except Exception as e:
            print(e)

        return trades_red
    else:
        return pd.DataFrame()

def do_group_table_output(bigdf, glist):
    if "0" in glist:
        wins = bigdf.loc[bigdf['profit_abs'] >= 0].groupby(['buy_reason']).agg({'profit_abs': ['sum']})
        wins.columns = ['profit_abs_wins']
        loss = bigdf.loc[bigdf['profit_abs'] < 0].groupby(['buy_reason']).agg({'profit_abs': ['sum']})
        loss.columns = ['profit_abs_loss']

        new = bigdf.groupby(['buy_reason']).agg({'profit_abs':
                                                 ['count',
                                                  lambda x: sum(x > 0),
                                                  lambda x: sum(x <= 0),]})

        new = pd.merge(new, wins, left_index=True, right_index=True)
        new = pd.merge(new, loss, left_index=True, right_index=True)

        new['profit_tot'] = new['profit_abs_wins'] - abs(new['profit_abs_loss'])

        new['wl_ratio_pct'] = (new.iloc[:, 1]/new.iloc[:, 0]*100)
        new['avg_win'] = (new['profit_abs_wins']/new.iloc[:, 1])
        new['avg_loss'] = (new['profit_abs_loss']/new.iloc[:, 2])

        new.columns = ['total_num_buys', 'wins', 'losses', 'profit_abs_wins', 'profit_abs_loss', "profit_tot", "wl_ratio_pct", "avg_win", "avg_loss"]

        sortcols = ['total_num_buys']

        print_table(new, sortcols, show_index=True)
    if "1" in glist:
        new = bigdf.groupby(['buy_reason']).agg({'profit_abs': ['count', 'sum', 'median', 'mean'], 'profit_ratio': ['sum', 'median', 'mean']}).reset_index()
        new.columns = ['buy_reason', 'num_buys', 'profit_abs_sum', 'profit_abs_median', 'profit_abs_mean', 'median_profit_pct', 'mean_profit_pct', 'total_profit_pct']
        sortcols = ['profit_abs_sum', 'buy_reason']

        new['median_profit_pct'] = new['median_profit_pct']*100
        new['mean_profit_pct'] = new['mean_profit_pct']*100
        new['total_profit_pct'] = new['total_profit_pct']*100

        print_table(new, sortcols)
    if "2" in glist:
        new = bigdf.groupby(['buy_reason', 'sell_reason']).agg({'profit_abs': ['count', 'sum', 'median', 'mean'], 'profit_ratio': ['sum', 'median', 'mean']}).reset_index()
        new.columns = ['buy_reason', 'sell_reason', 'num_buys', 'profit_abs_sum', 'profit_abs_median', 'profit_abs_mean', 'median_profit_pct', 'mean_profit_pct', 'total_profit_pct']
        sortcols = ['profit_abs_sum', 'buy_reason']

        new['median_profit_pct'] = new['median_profit_pct']*100
        new['mean_profit_pct'] = new['mean_profit_pct']*100
        new['total_profit_pct'] = new['total_profit_pct']*100

        print_table(new, sortcols)
    if "3" in glist:
        new = bigdf.groupby(['pair', 'buy_reason']).agg({'profit_abs': ['count', 'sum', 'median', 'mean'], 'profit_ratio': ['sum', 'median', 'mean']}).reset_index()
        new.columns = ['pair', 'buy_reason', 'num_buys', 'profit_abs_sum', 'profit_abs_median', 'profit_abs_mean', 'median_profit_pct', 'mean_profit_pct', 'total_profit_pct']
        sortcols = ['profit_abs_sum', 'buy_reason']

        new['median_profit_pct'] = new['median_profit_pct']*100
        new['mean_profit_pct'] = new['mean_profit_pct']*100
        new['total_profit_pct'] = new['total_profit_pct']*100

        print_table(new, sortcols)
    if "4" in glist:
        new = bigdf.groupby(['pair', 'buy_reason', 'sell_reason']).agg({'profit_abs': ['count', 'sum', 'median', 'mean'], 'profit_ratio': ['sum', 'median', 'mean']}).reset_index()
        new.columns = ['pair', 'buy_reason', 'sell_reason', 'num_buys', 'profit_abs_sum', 'profit_abs_median', 'profit_abs_mean', 'median_profit_pct', 'mean_profit_pct', 'total_profit_pct']
        sortcols = ['profit_abs_sum', 'buy_reason']

        new['median_profit_pct'] = new['median_profit_pct']*100
        new['mean_profit_pct'] = new['mean_profit_pct']*100
        new['total_profit_pct'] = new['total_profit_pct']*100

        print_table(new, sortcols)

def process_one(pair, stratnames, trade_dict, all_candles, ft_config, current_count, total_job_num, use_timeframe_detail=False, verbose=False, write_out=False):
    analysed_trades_dict = {}

    time.sleep(3)

    for sname in stratnames:
        stake = "USDT"

        loc_config = ft_config.copy()

        # Generate buy/sell signals using strategy
        #timeframe = "5m"
        #loc_config['timeframe'] = timeframe

        if use_timeframe_detail:
            timeframe_detail = "1m"
            loc_config['timeframe_detail'] = timeframe_detail

        #data_location = Path('/mnt', 'm', 'freqtrade_data', 'binance_json', 'binance')
        #loc_config['datadir'] = data_location

        loc_config['strategy'] = sname
        ft_exchange = ExchangeResolver.load_exchange(loc_config['exchange']['name'], config=loc_config, validate=False)
        ft_pairlists = PairListManager(ft_exchange, loc_config)
        ft_dataprovider = DataProvider(loc_config, ft_exchange, ft_pairlists)

        # Load strategy using values set above
        strategy = StrategyResolver.load_strategy(loc_config)
        strategy.dp = ft_dataprovider

        analysed_trades_dict[sname] = {}

        try:
            print(f"Processing {sname} {pair} [{current_count}/{total_job_num}]")
            if len(all_candles[pair]) > 0:
                tb = do_analysis(strategy, pair, all_candles[pair], trade_dict[sname])
                analysed_trades_dict[sname][f'{pair.split("/")[0]}'] = tb

    #            if verbose:
    #                print(tabulate(tb[columns], headers = 'keys', tablefmt = 'psql'))

                if write_out:
                    tb.to_csv(f'{sname}_{pair.split("/")[0]}_trades.csv')
        except Exception as e:
            # print("Something got zucked: No trades with this pair or you don't have buy_tags in your dataframe:", e)
            pass
    return pair, analysed_trades_dict

def process_all(pairlist, stratnames, trade_dict, all_candles, ft_config, parallel=False, verbose=False):
    buy_reason_jobs = []

    total_job_num = len(pairlist) * len(stratnames)

    current_count = 1
    for pair in pairlist:
        buy_reason_jobs.append((pair, stratnames, trade_dict, all_candles, ft_config, current_count, total_job_num, verbose))
        current_count += 1

    if parallel:
        i = 0
        results = []

        job_split = int(cpu_count() * 0.5)
        print(f"Running {len(buy_reason_jobs)}/{job_split} concurrent jobs")

        while i < len(buy_reason_jobs):
            if job_split > 1:
                tasks = buy_reason_jobs[i:i+job_split]
            else:
                tasks = buy_reason_jobs
            #with get_context("spawn").Pool(max(job_split, 1)) as pool:
            with Pool(max(job_split, 1)) as pool:
                results.extend([job.get() for job in [pool.apply_async(process_one, p) for p in tasks]])
                pool.close()
                pool.join()
            i += len(tasks)
    else:
        results = [process_one(*p) for p in buy_reason_jobs]
    return results

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config", help="config to parse")
    parser.add_argument("-t", "--timerange", nargs='?', help="timerange as per freqtrade format, e.g. 20210401-, 20210101-20210201, etc")
    parser.add_argument("-s", "--strategy_list", nargs='?', help="strategies available in user_data/strategies to parse, e.g. AweseomeStrategy. if not supplied, will use the strategy defined in config supplied. Multiple strategies can be separated by commas")
    parser.add_argument("--indicator_list", nargs='?', help="Comma separated list of indicators to analyse")
    parser.add_argument("--buy_reason_list", nargs='?', help="Comma separated list of buy signals to analyse. Default: all")
    parser.add_argument("--sell_reason_list", nargs='?', help="Comma separated list of sell signals to analyse. Default: 'stop_loss,trailing_stop_loss'")
    parser.add_argument("-p", "--pairlist", nargs='?', help="pairlist as 'BTC/USDT,FOO/USDT,BAR/USDT'")
    parser.add_argument("-u", "--use_trades_db", action="store_true", help="use dry/live trade DB specified in config instead of backtest results DB. Default: False")
    parser.add_argument("-w", "--write_out", help="write an output CSV per pair", action="store_true")
    parser.add_argument("-g", "--group", nargs='?', help="grouping output - 0: simple wins/losses by buy reason, 1: by buy_reason, 2: by buy_reason and sell_reason, 3: by pair and buy_reason, 4: by pair, buy_ and sell_reason (this can get quite large)")
    parser.add_argument("-o", "--outfile", help="write all trades summary table output", type=argparse.FileType('w'))
    parser.add_argument("-d", "--data_format", nargs='?', choices=["json", "hdf5"], help="Specify the json or hdf5 datas. default is json")
    parser.add_argument("-l", "--data_dir_location", nargs='?', help="specify the path to the downloaded OHLCV jsons or hdf5 datas. default is user_data/data/<exchange_name>")
    parser.add_argument("-i", "--indicators", nargs='?', help="Indicator values to output, as a comma separated list, e.g. -i rsi,adx,macd")
    parser.add_argument("-x", "--cancels", action="store_true", help="Output buy cancel reasons. Default: False")
    parser.add_argument("-r", "--rk_tags", action="store_true", help="Use the ConditionLabeler tags instead of the newer buy_tag tagging feature in FT. Default: False")
    parser.add_argument("-a", "--alternative_tag_name", nargs='?', help="Supply a different buy_tag name to use instead of 'buy', e.g. 'prebuy'. This is for more complex buy_tag use in strategies.")
    parser.add_argument("-m", "--multiprocessing", action="store_true", help="Use parallel processing. Default: False")
    parser.add_argument("-v", "--verbose", help="verbose", action="store_true")
    args = parser.parse_args()

    configs = [args.config]

    ft_config = Configuration.from_files(files=configs)
    ft_exchange_name = ft_config['exchange']['name']
    ft_exchange = ExchangeResolver.load_exchange(ft_exchange_name, config=ft_config, validate=True)
    ft_pairlists = PairListManager(ft_exchange, ft_config)
    ft_dataprovider = DataProvider(ft_config, ft_exchange, ft_pairlists)

    timeframe = "5m"
    if "timeframe" in ft_config:
       timeframe = ft_config["timeframe"]

    user_data_dir = ft_config['user_data_dir']

    data_location = Path(user_data_dir, 'data', ft_exchange_name)
    if args.data_dir_location is not None:
        data_location = Path(args.data_dir_location)

    ft_config['datadir'] = data_location

    backtest_dir = Path(user_data_dir, 'backtest_results')

    if args.group is not None and args.indicators is not None:
        print("WARNING: cannot use indicator output with grouping. Ignoring indicator output")

    if args.pairlist is None:
        pairlist = ft_pairlists.whitelist
    else:
        pairlist = args.pairlist.split(",")

    if "data_format" in ft_config and args.data_format is None:
        data_format = ft_config["data_format"]
    elif args.data_format is not None:
        data_format = args.data_format
    else:
        data_format = "json"

    if args.alternative_tag_name is None:
        alternative_tag_name = "buy"
    else:
        alternative_tag_name = args.alternative_tag_name

    parallel = False
    if args.multiprocessing is not None:
      parallel = args.multiprocessing

    #timeframe = "5m"
    backtest = False

    stratnames = []
    if args.strategy_list is not None:
        for s in args.strategy_list.split(","):
            stratnames.append(s)
    else:
        stratnames.append(ft_config['strategy'])

    use_trades_db = False
    if args.use_trades_db is not None:
        use_trades_db = args.use_trades_db

    trade_dict = {}

    if use_trades_db is True:
        # Fetch trades from database
        print("Loading DB trades data...")
        trades = load_trades_from_db(ft_config['db_url'])
        trade_dict[ft_config['strategy']] = trades
    else:
        for sname in stratnames:
            print(f"Loading backtest trades data for {sname} ...")
            trades = load_backtest_data(backtest_dir, sname)
            trade_dict[sname] = trades

    all_candles = load_candles(pairlist, args.timerange, data_location, timeframe, data_format)

    columns = ['pair', 'open_date', 'close_date', 'profit_abs', 'buy_reason', 'sell_reason']

    count = 1
    tbresults = dict()

    analysed_trades_dict = {}

    results = process_all(pairlist, stratnames, trade_dict, all_candles, ft_config, parallel=parallel)

    print_results(results, analysed_trades_dict, stratnames, args.outfile, args.group, args.buy_reason_list, args.sell_reason_list, args.indicator_list, args.cancels)

def print_results(results, stratnames, outfile=None, group="0,1,2", buy_reason_list="all", sell_reason_list="all", indicator_list=None, cancels=False, columns=['pair', 'open_date', 'close_date', 'profit_abs', 'buy_reason', 'sell_reason'], start_date=None, end_date=None):

    analysed_trades_dict = {}

    for r in results:
        pair, analysed_trades = r
        coin = pair.split("/")[0]

        for stratname in analysed_trades.keys():
            if stratname not in analysed_trades_dict:
                analysed_trades_dict[stratname] = {}
            if coin in analysed_trades[stratname].keys():
                analysed_trades_dict[stratname][coin] = analysed_trades[stratname][coin]

    for sname in stratnames:
        bigdf = pd.DataFrame()

        print(f"{sname}")
        for tpair, trades in analysed_trades_dict[sname].items():
            bigdf = bigdf.append(trades, ignore_index=True)

        if outfile:
            outfile.write(bigdf.to_csv())
        else:
            bigdf.to_csv(f'{sname}_all_trade_buys.csv')

        if (start_date is not None):
            bigdf = bigdf.loc[(bigdf['date'] > start_date)]

        if (end_date is not None):
            bigdf = bigdf.loc[(bigdf['date'] < end_date)]

        if bigdf.shape[0] > 0 and ('buy_reason' in bigdf.columns):
            if group is not None:
                glist = group.split(",")
                do_group_table_output(bigdf, glist)

            if buy_reason_list is not None and not buy_reason_list == "all":
                buy_reason_list = buy_reason_list.split(",")
                bigdf = bigdf.loc[(bigdf['buy_reason'].isin(buy_reason_list))]

            if sell_reason_list is not None and not sell_reason_list == "all":
                sell_reason_list = sell_reason_list.split(",")
                bigdf = bigdf.loc[(bigdf['sell_reason'].isin(sell_reason_list))]
            #else:
            #    bigdf = bigdf.loc[(bigdf['sell_reason'] == "stop_loss") | (bigdf['sell_reason'] == "trailing_stop_loss")]

            if indicator_list is not None:
                if indicator_list == "all":
                    print(bigdf)
                else:
                    available_inds = []
                    for ind in indicator_list.split(","):
                        if ind in bigdf:
                            available_inds.append(ind)
                    ilist = ["pair", "buy_reason", "sell_reason"] + available_inds
                    print(tabulate(bigdf[ilist].sort_values(['sell_reason']), headers = 'keys', tablefmt = 'psql', showindex=False))
            else:
                print(tabulate(bigdf[columns].sort_values(['pair']), headers = 'keys', tablefmt = 'psql', showindex=False))

            if cancels:
                if bigdf.shape[0] > 0 and ('cancel_reason' in bigdf.columns):

                    cs = bigdf.groupby(['buy_reason']).agg({'profit_abs': ['count'], 'cancel_reason': ['count']}).reset_index()
                    cs.columns = ['buy_reason', 'num_buys', 'num_cancels']
                    cs['percent_cancelled'] = (cs['num_cancels']/cs['num_buys']*100)

                    sortcols = ['num_cancels']
                    print_table(cs, sortcols)

        else:
            print("\_ No trades to show")

def print_table(df, sortcols=None, show_index=False):
    if (sortcols is not None):
        data = df.sort_values(sortcols)
    else:
        data = df

    print(
        tabulate(
            data,
            headers = 'keys',
            tablefmt = 'psql',
            showindex=show_index
        )
    )

def wl(x):
    return f"{(lambda x: sum(x > 0))}/{(lambda x: sum(x <= 0))}"

if __name__ == "__main__":
    main()
