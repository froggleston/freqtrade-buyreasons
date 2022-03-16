# freqtrade-buyreasons
A script to parse and display buy_tag and sell_reason for freqtrade backtesting trades

You will end up with tables that look like the following, with various group options for pair, buy_tag, sell_reason, and if specified, indicator values for the buy signal candle based on chosen dataframe columns:

![image](https://user-images.githubusercontent.com/1872302/158602768-26f44baf-ea92-46f9-affa-b23b49126b5f.png)

## Usage

Copy the buy_reasons.py script into your freqtrade/scripts folder. You then need to backtest with the --export options enabled in freqtrade so that this script has the right data to work with:

`freqtrade backtesting -c <config.json> --timeframe 5m --strategy <strategy_name> --timerange=<timerange> --export=trades --export-filename=user_data/backtest_results/<name>-<timerange>`

Then you can run:

`buy_reasons.py -c <config.json> -s <strategy_name> -t <timerange> -g<[0,1,2,3,4]> [-l <path_to_data_dir>]`

The `-l` option is the same as the `--datadir` option in freqtrade, in case you have your downloaded historic data in a different folder to `user_data/data/`

Example:

`buy_reasons.py -c my_config.json -s DoNothingStrategy -t 20211001- -g0,1,2`
