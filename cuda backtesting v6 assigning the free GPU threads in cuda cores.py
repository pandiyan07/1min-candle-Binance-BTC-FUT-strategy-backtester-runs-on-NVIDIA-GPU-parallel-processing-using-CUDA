'''

Why use the CuPy Occupancy Estimation ?

It gives you the number of available warps/threads that can execute your arithmetic tasks.

It helps in efficient workload scheduling by avoiding thread overloading.

It ensures maximum parallel execution by filling idle cores with operations

'''









'''

Potential Improvement

If you want to leverage more parallelism and execute multiple parameter sets concurrently,

you could consider restructuring your code to launch multiple kernels at once or use a different approach such as:



-   Batch Processing: Modify the kernel to accept multiple parameter sets at once and process them in parallel.

-   Stream Processing: Use CUDA streams to manage concurrent execution of multiple kernels.



This would allow you to fully utilize the GPU's capabilities and potentially speed up your backtesting process significantly.



To achieve batch processing and stream processing in your trading bot code using CuPy, you can modify the kernel to handle multiple

parameter sets simultaneously. This allows you to leverage the GPU's parallel processing capabilities more effectively. 



Key Changes Made

1)  Batch Processing: The kernel is modified to accept arrays of parameters (long_lengths, short_lengths, etc.) instead of individual parameters. 

This allows each thread to access the appropriate parameter set based on its index.

2)  Single Kernel Launch: Instead of launching the kernel multiple times in a loop over parameter sets (which would process them sequentially), 

we launch it once with all data and parameters at once.

3)  Result Storage: The results array is structured to store profits and trade counts for each candle and parameter combination.



->  Stream Processing

While this example does not explicitly implement CUDA streams due to complexity and potential overheads in managing them within this context, 

you can utilize streams by creating multiple streams and launching different kernels on those streams if you want more advanced parallelism.



To implement streams:

    Create multiple cp.cuda.Stream objects.

    Launch different kernel executions on these streams.

    Synchronize streams as necessary.

This setup allows you to take full advantage of GPU capabilities by processing multiple parameter sets simultaneously while maintaining clarity and efficiency in your code.

'''





















#   by making  threads_per_block = 1, This eliminates the need for complex thread synchronization or atomic operations within each block











#   3,13,52,832 = throws error

#   1 second = approx 5,200 rows when processing 13k

#   1 second = approx 3062 rows

#   1 second = approx 2679 rows when processing 115200

#   takes more than 1 hour = 1572864 rows - dont know how long will it take as it was abruptly cancelled when it was still running





#   ask line by line explanation with comments of the TRADING LOGIC from the deepseek

#   if the thread_block in 512 is ok, then try the max thread per block of 1024 value





















# (31352832 * 113) * 4 = 14,171,480,064 bytes, or approximately 14 GB

# 1,30,65,442 rows/parameters is the maximum amount of rows/parameters we can use



#cuda thread core parallel processing python code

# 1st portion

import pandas as pd

import numpy as np

import pandas_ta as ta

import cupy as cp

from joblib import Memory

from datetime import datetime



# Set up caching to speed up calculations

memory = Memory("./cache", verbose=0)



# Function to calculate EMA using pandas_ta with caching

@memory.cache

def CALCULATE_EMA(series, length):

    return ta.ema(series, length=length)



# Function to calculate ATR using pandas_ta with caching

@memory.cache

def CALCULATE_ATR(df, length):

    return ta.atr(df['high'], df['low'], df['close'], length=length)



# Modified CUDA Kernel with trailing stop loss

BACKTEST_KERNAL = cp.RawKernel(

    '''

    extern "C" __global__

    void BACKTEST_KERNAL(

        const float* open_price, const float* close_price,

        const float* ema_5, const float* ema_34,

        const float* atr,

        const float long_length, const float short_length,

        const float long_gap, const float short_gap,

        const float atr_value, const float percentage,

        const float tsl_long_percentage, const float tsl_short_percentage,

        const int num_params,

        const int param_index,

        float* results,

        const int num_candles,

        const int big_ema_length)

    {

        int i = blockIdx.x * blockDim.x + threadIdx.x;



        if (i != param_index) return;



        float total_profit = 0.0;

        float total_loss = 0.0;

        float wt = 0.0;

        int num_long_trades = 0;

        int num_short_trades = 0;

        int num_trades = 0;

        float profitable_trades_array[100];

        int profitable_trades_count = 0;

        const int max_profitable_trades = 100;

        float position = 0.0; // 0: None, 1: Long, 2: Short

        float entry_price = 0.0;

        float max_gap = 0.0;

        float max_profit_price = 0.0;

        int lt = 0;

        int st = 0;

        int j;



        for (j = 2; j < num_candles; j++) {

            float current_candle_close = close_price[j];

            float current_candle_open = open_price[j];

            float previous_candle_close = close_price[j - 1];

            float previous_candle_open = open_price[j - 1];

            float second_previous_candle_close = close_price[j - 2];

            float second_previous_candle_open = open_price[j - 2];

            float last_atr_value = atr[j];



            float current_diff;

            float previous_diff;



            if (current_candle_close > current_candle_open) {

                current_diff = ((current_candle_close - current_candle_open) * percentage) + current_candle_open;

                previous_diff = ((previous_candle_close - previous_candle_open) * percentage) + previous_candle_open;

            } else {

                current_diff = current_candle_open - ((current_candle_open - current_candle_close) * percentage);

                previous_diff = previous_candle_open - ((previous_candle_open - previous_candle_close) * percentage);

            }



            bool l1 = (current_candle_close - second_previous_candle_open) > long_length;

            bool l2 = (current_candle_close > ema_5[j]) && (current_candle_open > ema_5[j]);

            bool l3 = current_diff > previous_candle_close;

            bool l4 = previous_diff > second_previous_candle_close;

            bool l5 = ema_5[j] > ema_34[j];



            bool s1 = (second_previous_candle_open - current_candle_close) > short_length;

            bool s2 = (current_candle_close < ema_5[j]) && (current_candle_open < ema_5[j]);

            bool s3 = current_diff < previous_candle_close;

            bool s4 = previous_diff < second_previous_candle_close;

            bool s5 = ema_5[j] < ema_34[j];



            bool Atr_rule = last_atr_value > atr_value;



            bool long_condition = l1 && l2 && l3 && l4 && l5 && Atr_rule;

            bool short_condition = s1 && s2 && s3 && s4 && s5 && Atr_rule;



            float trade_profit = 0.0;

            if (position == 0.0) { // None

                if (long_condition) {

                    position = 1.0;

                    lt += 1;

                    entry_price = current_candle_close;

                    max_gap = ema_5[j] - ema_34[j];

                    max_profit_price = current_candle_close;

                    num_long_trades += 1;

                    num_trades += 1;

                } else if (short_condition) {

                    position = 2.0;

                    st += 1;

                    entry_price = current_candle_close;

                    max_gap = ema_34[j] - ema_5[j];

                    max_profit_price = current_candle_close;

                    num_short_trades += 1;

                    num_trades += 1;

                }

                

            } else if (position == 1.0) { // Long

                float current_gap = ema_5[j] - ema_34[j];

                max_gap = fmaxf(max_gap, current_gap);

                max_profit_price = fmaxf(max_profit_price, current_candle_close);

                float max_profit_made = max_profit_price - entry_price;

                float TSL = max_profit_made * tsl_long_percentage;

                float current_profit_made = current_candle_close - entry_price;



                bool exit_condition = (current_gap < max_gap * long_gap) || (current_profit_made < TSL && current_profit_made > 15);



                if (exit_condition) {

                    position = 0.0;

                    trade_profit = current_candle_close - entry_price;

                    if (trade_profit > 0) {

                        total_profit += trade_profit;

                        wt += 1.0;

                        if (profitable_trades_count < max_profitable_trades) {

                            profitable_trades_array[profitable_trades_count] = trade_profit;

                            profitable_trades_count++;

                        }

                    } else {

                        total_loss += trade_profit;

                    }

                }

                

            } else if (position == 2.0) { // Short

                float current_gap = ema_34[j] - ema_5[j];

                max_gap = fmaxf(max_gap, current_gap);

                max_profit_price = fminf(max_profit_price, current_candle_close);

                float max_profit_made = entry_price - max_profit_price;

                float TSL = max_profit_made * tsl_short_percentage;

                float current_profit_made = entry_price - current_candle_close;



                bool exit_condition = (current_gap < max_gap * short_gap) || (current_profit_made < TSL && current_profit_made > 15);



                if (exit_condition) {

                    position = 0.0;

                    trade_profit = entry_price - current_candle_close;

                    if (trade_profit > 0) {

                        total_profit += trade_profit;

                        wt += 1.0;

                        if (profitable_trades_count < max_profitable_trades) {

                            profitable_trades_array[profitable_trades_count] = trade_profit;

                            profitable_trades_count++;

                        }

                    } else {

                        total_loss += trade_profit;

                    }

                }

            }

        }



        int num_trades_idx = param_index * 13;

        int num_long_trades_idx = param_index * 13 + 1;

        int num_short_trades_idx = param_index * 13 + 2;

        int total_profit_idx = param_index * 13 + 3;

        int total_loss_idx = param_index * 13 + 4;

        int wt_idx = param_index * 13 + 5;

        int profitable_trades_count_idx = param_index * 13 + 6;

        int profitable_trades_idx = num_params * 13 + param_index * 100;



        results[num_trades_idx] = num_trades;

        results[num_long_trades_idx] = num_long_trades;

        results[num_short_trades_idx] = num_short_trades;

        results[total_profit_idx] = total_profit;

        results[total_loss_idx] = total_loss;

        results[wt_idx] = wt;

        results[profitable_trades_count_idx] = profitable_trades_count;



        for (int k = 0; k < profitable_trades_count; k++) {

            results[profitable_trades_idx + k] = profitable_trades_array[k];

        }

    }

    ''',

    'BACKTEST_KERNAL'

)



#cuda thread core parallel processing python code

# 2nd portion

def BACKTEST_GPU(open_price, close_price, ema_5, ema_34, atr, params):

    num_candles = open_price.shape[0]

    num_params = params.shape[0]



    print("THE NUMBER OF PARAMETERS = ", num_params)



    results = cp.zeros((num_params * 13 + num_params * 100), dtype=cp.float32)



    threads_per_block = 1

    blocks_per_grid = num_params



    num_cuda_cores_used = blocks_per_grid * threads_per_block



    open_price = np.asarray(open_price, dtype=np.float32)

    close_price = np.asarray(close_price, dtype=np.float32)

    ema_5 = np.asarray(ema_5, dtype=np.float32)

    ema_34 = np.asarray(ema_34, dtype=np.float32)

    atr = np.asarray(atr, dtype=np.float32)



    open_price_gpu = cp.asarray(open_price)

    close_price_gpu = cp.asarray(close_price)

    ema_5_gpu = cp.asarray(ema_5)

    ema_34_gpu = cp.asarray(ema_34)

    atr_gpu = cp.asarray(atr)



    for param_index in range(num_params):

        long_length = np.float32(params[param_index, 0])

        short_length = np.float32(params[param_index, 1])

        atr_value = np.float32(params[param_index, 2])

        long_gap = np.float32(params[param_index, 3])

        short_gap = np.float32(params[param_index, 4])

        percentage = np.float32(params[param_index, 5])

        tsl_long_percentage = np.float32(params[param_index, 6])

        tsl_short_percentage = np.float32(params[param_index, 7])



        BACKTEST_KERNAL((blocks_per_grid,), (threads_per_block,), (

            open_price_gpu.data.ptr,

            close_price_gpu.data.ptr,

            ema_5_gpu.data.ptr,

            ema_34_gpu.data.ptr,

            atr_gpu.data.ptr,

            long_length,

            short_length,

            long_gap,

            short_gap,

            atr_value,

            percentage,

            tsl_long_percentage,

            tsl_short_percentage,

            num_params,

            param_index,

            results.data.ptr,

            num_candles,

            34))



    return results, num_cuda_cores_used



#cuda thread core parallel processing python code

# 3rd portion

def TRADING_BOT(df):

    print('Starting Trading Bot')



    #try:

    df['ema_5'] = CALCULATE_EMA(df['close'], 5)

    df['ema_34'] = CALCULATE_EMA(df['close'], 34)

    df['atr'] = CALCULATE_ATR(df, 14)



    params_list = [

        (long_length, short_length, atr_value, long_gap, short_gap, percentage, tsl_long_percentage, tsl_short_percentage)

        for long_length in np.arange(90, 100, 10)

        for short_length in np.arange(90, 100, 10)

        for atr_value in np.arange(50, 70, 20)

        for long_gap in np.linspace(0.40, 0.90, 6)

        for short_gap in np.linspace(0.40, 0.90, 6)

        for percentage in np.linspace(0, 0.5, 6)

        for tsl_long_percentage in np.linspace(0.4, 0.90, 6)

        for tsl_short_percentage in np.linspace(0.4, 0.90, 6)

    ]



    params = np.array(params_list, dtype=np.float32)



    print(f"Total PARAMS = {params.shape[0]}")



    start_time = datetime.now()

    print('\n\nStarted the backtesting at = ', start_time)



    open_price = df['open'].values.astype(np.float32)

    close_price = df['close'].values.astype(np.float32)

    ema_5 = df['ema_5'].values.astype(np.float32)

    ema_34 = df['ema_34'].values.astype(np.float32)

    atr = df['atr'].values.astype(np.float32)



    results, num_cuda_cores_used = BACKTEST_GPU(open_price, close_price,ema_5,ema_34,atr,params)



    end_time = datetime.now()

    print('Ended the backtesting at = ', end_time)

    print("\nBacktesting completed in", end_time - start_time, "seconds")

    print("Number of CUDA Cores Used = ", num_cuda_cores_used, "\n")



    output_data = []

    num_params = params.shape[0]



    for param_index in range(num_params):

        num_trades = results[param_index * 13].get()

        num_long_trades = results[param_index * 13 + 1].get()

        num_short_trades = results[param_index * 13 + 2].get()

        total_profit = results[param_index * 13 + 3].get()

        total_loss = results[param_index * 13 + 4].get()

        wt = results[param_index * 13 + 5].get()



        profitable_trades_count = int(results[param_index * 13 + 6].get())

        profitable_trades_start_idx = num_params * 13 + param_index * 100

        profitable_trades = []

        for i in range(min(profitable_trades_count, 100)):

            trade = results[profitable_trades_start_idx + i].get()

            profitable_trades.append(trade)



        long_length = params[param_index, 0]

        short_length = params[param_index, 1]

        long_gap = params[param_index, 2]

        short_gap = params[param_index, 3]

        percentage = params[param_index, 5]

        atr_value = params[param_index, 4]

        tsl_long_percentage = params[param_index, 6]

        tsl_short_percentage = params[param_index, 7]



        win_percent = (wt / num_trades) * 100 if num_trades > 0 else 0

        net_profit = total_profit + total_loss



        top_profitable_trades = sorted(profitable_trades, reverse=True)[:5]

        top_profitable_trades_str = ','.join(map(str, top_profitable_trades))



        output_data.append([

            long_length, short_length, long_gap, short_gap, percentage, atr_value,

            tsl_long_percentage, tsl_short_percentage,  # Added new parameters

            num_trades, num_long_trades, num_short_trades, top_profitable_trades_str,

            win_percent, net_profit, total_profit, total_loss

        ])

    '''

    except Exception as e:

        print(f"An error occurred: {e}")

    '''

    return output_data



import subprocess



def GET_GPU_UTILISATION():

    #try:

    result = subprocess.run(

        ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],

        stdout=subprocess.PIPE, text=True

    )

    return int(result.stdout.strip())  # GPU Utilization %

    '''

    except Exception as e:

        return f"Error fetching GPU utilization: {e}"

    '''



def ESTIMATED_FREE_CUDA_CORE_THREADS():

    device = cp.cuda.Device(0)  # Select GPU 0

    attributes = device.attributes  # Get device attributes



    num_sms = device.attributes['MultiProcessorCount']  # Number of SMs

    warp_size = device.attributes['WarpSize']  # Warp size (Usually 32)

    max_warps_per_sm = 32  # Typical max warps per SM (1024 threads / 32 threads per warp)



    # Estimate free warps

    max_warps_total = num_sms * max_warps_per_sm

    active_warps = device.attributes['MultiProcessorCount'] * 8  # Example: 8 warps active per SM



    free_warps = max_warps_total - active_warps

    free_threads = free_warps * warp_size



    gpu_usage = GET_GPU_UTILISATION()

    print(f"Current GPU Utilization = {gpu_usage}%")

    print(f"Estimated Free CUDA Threads = {free_threads}")



#cuda thread core parallel processing python code

# 4th portion



# Example usage block

if __name__ == "__main__":

    #try:

    df = pd.read_csv("data_ohlc_feb1.csv")

    print('\n\n\nSuccessfully read data_ohlc_feb1.csv')

    #ESTIMATED_FREE_CUDA_CORE_THREADS()

    output_data = TRADING_BOT(df)

    output_df = pd.DataFrame(output_data, columns=[

        'Long length', 'Short length', 'Long Gap', 'Short Gap', 'percentage', 'ATR value',

        'TSL Long Percentage', 'TSL Short Percentage',  # Added new columns

        'Total No of trades', 'No of Long trades', 'No of Short trades', 'Top profitable trades',

        'Win percent', 'Net Profit', 'Total Profit', 'Total Loss'

    ])

    output_df = output_df.sort_values(by='Net Profit', ascending=False)

    output_df.to_csv("cuda_v6_feb8_report_on_feb1_data_gpu_with_tsl.csv", index=False)



    print(output_df[:20])

    print('\nTrading Bot Ended\n')

    #print(output_df)

    '''

    except Exception as e:

        print(f"Error: {e}")

    '''