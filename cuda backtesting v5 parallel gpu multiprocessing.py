#   Yes, the NVIDIA GTX 1660 Super can handle up to 22,528 threads in parallel at the same time under ideal conditions.



import pandas as pd

import numpy as np

import pandas_ta as ta

import cupy as cp

import multiprocessing as mp

from joblib import Memory

from datetime import datetime



import multiprocessing as mp



# Set up caching to speed up calculations

memory = Memory("./cache", verbose=0)



# Function to calculate EMA using pandas_ta with caching

@memory.cache

def calculate_ema(series, length):

    return ta.ema(series, length=length)



# Function to calculate ATR using pandas_ta with caching

@memory.cache

def calculate_atr(df, length):

    return ta.atr(df['high'], df['low'], df['close'], length=length)



# Function to run backtesting for a subset of params on GPU

def backtest_gpu(params_chunk, open_prices, close_prices, high_prices, low_prices, ema_5, ema_34, atr_values):

    process_name = mp.current_process().name  # Get current process name

    print(f"Process {process_name} is running {len(params_chunk)} parameters.")



    cp.cuda.Device(0).use()  # Ensure this function runs on the GPU

    open_prices = cp.asarray(open_prices)

    close_prices = cp.asarray(close_prices)

    high_prices = cp.asarray(high_prices)

    low_prices = cp.asarray(low_prices)

    ema_5 = cp.asarray(ema_5)

    ema_34 = cp.asarray(ema_34)

    atr_values = cp.asarray(atr_values)



    results = cp.zeros((len(params_chunk), 3))  # Store results



    for idx, param in enumerate(params_chunk):

        print(f"{process_name}: Processing Param Set {idx + 1}/{len(params_chunk)}")  # Print process progress

        print('params_chunk = ',params_chunk)

        

        long_length, short_length, long_gap, short_gap, percentage, atr_value = param

        position = None

        entry_price = 0.0

        max_gap = 0.0

        profit = 0.0

        win_trd = 0

        loss = 0



        # Main backtesting loop

        for i in range(2, len(close_prices)):

            current_candle_close = close_prices[i]

            current_candle_open = open_prices[i]

            previous_candle_close = close_prices[i - 1]

            previous_candle_open = open_prices[i - 1]

            second_previous_candle_close = close_prices[i - 2]

            second_previous_candle_open = open_prices[i - 2]

            last_atr_value = atr_values[i]



            if current_candle_close > current_candle_open:

                current_diff = ((current_candle_close - current_candle_open) * percentage) + current_candle_open

                previous_diff = ((previous_candle_close - previous_candle_open) * percentage) + previous_candle_open

            else:

                current_diff = ((current_candle_open - current_candle_close) * percentage) - current_candle_open

                previous_diff = ((previous_candle_open - previous_candle_close) * percentage) - previous_candle_open



            l1 = (current_candle_close - second_previous_candle_open) > long_length

            l2 = current_candle_close > ema_5[i] and current_candle_open > ema_5[i]

            l3 = current_diff > previous_candle_close

            l4 = previous_diff > second_previous_candle_close

            l5 = ema_5[i] > ema_34[i]



            s1 = (second_previous_candle_open - current_candle_close) > short_length

            s2 = current_candle_close < ema_5[i] and current_candle_open < ema_5[i]

            s3 = current_diff < previous_candle_close

            s4 = previous_diff < second_previous_candle_close

            s5 = ema_5[i] < ema_34[i]



            Atr_rule = last_atr_value > atr_value



            long_condition = l1 and l2 and l3 and l4 and l5 and Atr_rule

            short_condition = s1 and s2 and s3 and s4 and s5 and Atr_rule



            if position is None:

                if long_condition:

                    position = "Long"

                    entry_price = current_candle_close

                    max_gap = ema_5[i] - ema_34[i]

                elif short_condition:

                    position = "Short"

                    entry_price = current_candle_close

                    max_gap = ema_34[i] - ema_5[i]



            if position == "Long":

                current_gap = ema_5[i] - ema_34[i]

                max_gap = cp.maximum(max_gap, current_gap)

                if current_gap < max_gap * long_gap:

                    position = None

                    profit_made = current_candle_close - entry_price

                    win_trd += 1 if profit_made > 0 else 0

                    loss += 1 if profit_made <= 0 else 0

                    profit += profit_made



            if position == "Short":

                current_gap = ema_34[i] - ema_5[i]

                max_gap = cp.maximum(max_gap, current_gap)

                if current_gap < max_gap * short_gap:

                    position = None

                    profit_made = entry_price - current_candle_close

                    win_trd += 1 if profit_made > 0 else 0

                    loss += 1 if profit_made <= 0 else 0

                    profit += profit_made



        results[idx, 0] = profit

        results[idx, 1] = win_trd

        results[idx, 2] = loss

    print(f"{process_name}: Finished processing {len(params_chunk)} parameter sets.")

    return cp.asnumpy(results)  # Convert GPU results back to CPU memory





# Function to distribute the workload among multiple CUDA cores using multiprocessing

def trading_bot(df):

    df['ema_5'] = calculate_ema(df['close'], 5)

    df['ema_34'] = calculate_ema(df['close'], 34)

    df['atr'] = calculate_atr(df, 14)



    params = np.array([(long_length, short_length, long_gap, short_gap, percentage, atr_value)

                        for long_length in np.arange(220, 230, 10)

                        for short_length in np.arange(230, 240, 10)

                        for atr_value in np.arange(50, 70, 10)

                        for long_gap in np.linspace(0.70, 0.80, 2)

                        for short_gap in np.linspace(0.80, 0.90, 2)

                        for percentage in np.linspace(0.3, 0.5, 3)])



    print(f"Total PARAMS = {len(params)}")



    # Get the number of available CUDA multiprocessors

    num_cuda_cores = cp.cuda.runtime.getDeviceProperties(0)['multiProcessorCount']

    num_processes = min(num_cuda_cores, len(params))  # Don't create more processes than params

    

    print('The number of available CUDA multiprocessors = ',num_cuda_cores)

    print(f"Using {num_processes} CUDA cores for parallel processing.")



    # Split params into chunks for parallel processing

    params_chunks = np.array_split(params, num_processes)

    start_time = datetime.now()

    # Create a multiprocessing pool

    with mp.Pool(processes=num_processes) as pool:

        results_list = pool.starmap(backtest_gpu, [(chunk, df['open'].values, df['close'].values,

                                                    df['high'].values, df['low'].values,

                                                    df['ema_5'].values, df['ema_34'].values,

                                                    df['atr'].values) for chunk in params_chunks])

    end_time = datetime.now()

    print("\n\n\nBacktesting completed in a total time of ",end_time - start_time," seconds\n\n")

    # Combine results from all processes

    results = np.vstack(results_list)



    # Save results

    results_df = pd.DataFrame(results, columns=['Profit', 'Winning Trades', 'Losing Trades'])

    results_df.to_csv("output_feb2_parallel.csv", index=False)



    return results_df





# Example usage

if __name__ == "__main__":

    df = pd.read_csv("data_ohlc_feb1.csv")

    trading_results = trading_bot(df)

    print(trading_results)

