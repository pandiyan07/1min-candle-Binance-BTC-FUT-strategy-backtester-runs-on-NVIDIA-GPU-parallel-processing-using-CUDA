# 1990656 (2 million) columns 
# Backtesting completed in 00.17 seconds
# Excel sheet generated in =  31.12 seconds

# 13716864 (13 million) columns 
# Excel sheet generated in =  6.46 minutes

import pandas as pd
import numpy as np
import pandas_ta as ta
import cupy as cp
from joblib import Memory
from datetime import datetime

# Set up caching for technical indicator calculations
memory = Memory("./cache", verbose=0)

@memory.cache
def CALCULATE_EMA(series, length):
    # Use pandas_ta's EMA calculation
    return ta.ema(series, length=length)

@memory.cache
def CALCULATE_ATR(df, length):
    # Use pandas_ta's ATR calculation; expects high, low, close columns in df
    return ta.atr(df['high'], df['low'], df['close'], length=length)

# Define the GPU kernel for backtesting.
# Note: The kernel reserves 14 result slots per parameter (only 9 are currently used)
BACKTEST_KERNAL = cp.RawKernel(
    r'''
    extern "C" __global__
    void BACKTEST_KERNAL(
        const float* open_price, const float* close_price,
        const float* ema_5, const float* ema_34,
        const float* atr,
        const float* params,
        float* results,
        const int num_candles,
        const int num_params)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= num_params) return;

        // Load parameters for this thread (8 parameters per set)
        float long_length = params[tid * 8];
        float short_length = params[tid * 8 + 1];
        float atr_value = params[tid * 8 + 2];
        float long_gap = params[tid * 8 + 3];
        float short_gap = params[tid * 8 + 4];
        float percentage = params[tid * 8 + 5];
        float tsl_long_percentage = params[tid * 8 + 6];
        float tsl_short_percentage = params[tid * 8 + 7];

        // Reserve 14 slots per parameter for main results; additional space for profitable trades
        int result_base_idx = tid * 14;
        int profitable_trades_base_idx = num_params * 14 + tid * 100;

        float total_profit = 0.0;
        float total_loss = 0.0;
        float wt = 0.0;
        int num_long_trades = 0;
        int num_short_trades = 0;
        int profitable_trades_count = 0;
        float profitable_trades[100];
        float position = 0.0;
        float entry_price = 0.0;
        float max_gap = 0.0;
        float max_profit_price = 0.0;

        for (int j = 2; j < num_candles; j++) {
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

            if (position == 0.0) {
                if (long_condition) {
                    position = 1.0;
                    entry_price = current_candle_close;
                    max_gap = ema_5[j] - ema_34[j];
                    max_profit_price = current_candle_close;
                    num_long_trades++;
                } else if (short_condition) {
                    position = 2.0;
                    entry_price = current_candle_close;
                    max_gap = ema_34[j] - ema_5[j];
                    max_profit_price = current_candle_close;
                    num_short_trades++;
                }
            } else if (position == 1.0) { // Long position management
                float current_gap = ema_5[j] - ema_34[j];
                max_gap = fmaxf(max_gap, current_gap);
                max_profit_price = fmaxf(max_profit_price, current_candle_close);
                float max_profit_made = max_profit_price - entry_price;
                float TSL = max_profit_made * tsl_long_percentage;
                float current_profit_made = current_candle_close - entry_price;

                if ((current_gap < max_gap * long_gap) || ((current_profit_made < TSL) && (current_profit_made > 15))) {
                    position = 0.0;
                    float trade_profit = current_candle_close - entry_price;
                    if (trade_profit > 0) {
                        total_profit += trade_profit;
                        wt += 1.0;
                        if (profitable_trades_count < 100) {
                            profitable_trades[profitable_trades_count++] = trade_profit;
                        }
                    } else {
                        total_loss += trade_profit;
                    }
                }
            } else if (position == 2.0) { // Short position management
                float current_gap = ema_34[j] - ema_5[j];
                max_gap = fmaxf(max_gap, current_gap);
                max_profit_price = fminf(max_profit_price, current_candle_close);
                float max_profit_made = entry_price - max_profit_price;
                float TSL = max_profit_made * tsl_short_percentage;
                float current_profit_made = entry_price - current_candle_close;

                if ((current_gap < max_gap * short_gap) || ((current_profit_made < TSL) && (current_profit_made > 15))) {
                    position = 0.0;
                    float trade_profit = entry_price - current_candle_close;
                    if (trade_profit > 0) {
                        total_profit += trade_profit;
                        wt += 1.0;
                        if (profitable_trades_count < 100) {
                            profitable_trades[profitable_trades_count++] = trade_profit;
                        }
                    } else {
                        total_loss += trade_profit;
                    }
                }
            }
        }

        // Store results in the first 14 slots for this parameter set.
        float num_trades = num_long_trades + num_short_trades;
        float win_percent = (num_trades > 0) ? (wt / num_trades) * 100.0f : 0.0f;
        float net_profit = total_profit + total_loss;

        results[result_base_idx]     = num_trades;
        results[result_base_idx + 1] = num_long_trades;
        results[result_base_idx + 2] = num_short_trades;
        results[result_base_idx + 3] = total_profit;
        results[result_base_idx + 4] = total_loss;
        results[result_base_idx + 5] = wt;
        results[result_base_idx + 6] = (float)profitable_trades_count;
        results[result_base_idx + 7] = net_profit;
        results[result_base_idx + 8] = win_percent;
        // The remaining result slots (indices 9-13) are left as zeros.

        // Store profitable trades for this parameter set
        for (int k = 0; k < profitable_trades_count; k++) {
            results[profitable_trades_base_idx + k] = profitable_trades[k];
        }
    }
    ''',
    'BACKTEST_KERNAL'
)

def BACKTEST_GPU(open_price, close_price, ema_5, ema_34, atr, params):
    num_candles = open_price.shape[0]
    num_params = params.shape[0]

    print("THE NUMBER OF PARAMETERS =", num_params)
    
    # Allocate results: 14 result slots per parameter plus 100 slots for profitable trades per parameter.
    results = cp.zeros((num_params * 14 + num_params * 100), dtype=cp.float32)

    threads_per_block = 512
    blocks_per_grid = (num_params + threads_per_block - 1) // threads_per_block

    # Convert all input arrays to CuPy arrays (if not already)
    gpu_data = {
        'open': cp.asarray(open_price, dtype=cp.float32),
        'close': cp.asarray(close_price, dtype=cp.float32),
        'ema_5': cp.asarray(ema_5, dtype=cp.float32),
        'ema_34': cp.asarray(ema_34, dtype=cp.float32),
        'atr': cp.asarray(atr, dtype=cp.float32),
        'params': cp.asarray(params, dtype=cp.float32)
    }

    # Launch the GPU kernel
    BACKTEST_KERNAL((blocks_per_grid,), (threads_per_block,), (
        gpu_data['open'],
        gpu_data['close'],
        gpu_data['ema_5'],
        gpu_data['ema_34'],
        gpu_data['atr'],
        gpu_data['params'],
        results,
        num_candles,
        num_params
    ))

    return results, blocks_per_grid * threads_per_block

def TRADING_BOT(df):
    print('Starting Trading Bot')

    # Calculate technical indicators and add as new columns in the DataFrame
    df['ema_5'] = CALCULATE_EMA(df['close'], 5)
    df['ema_34'] = CALCULATE_EMA(df['close'], 34)
    df['atr'] = CALCULATE_ATR(df, 14)

    # Create a list of parameter combinations.
    params_list = [
        (long_length, short_length, atr_value, long_gap, short_gap, percentage, tsl_long_percentage, tsl_short_percentage)
        for long_length in np.arange(90, 300, 10)         # (1000,4150,150)
        for short_length in np.arange(90, 300, 10)        # (1000,4150,150)
        for atr_value in np.arange(50, 130, 20)             # (200,1000,200)
        for long_gap in np.linspace(0.40, 0.90, 6)
        for short_gap in np.linspace(0.40, 0.90, 6)
        for percentage in np.linspace(0, 0.5, 6)
        for tsl_long_percentage in np.linspace(0.4, 0.90, 6)
        for tsl_short_percentage in np.linspace(0.4, 0.90, 6)
    ]
    params = np.array(params_list, dtype=np.float32)
    
    start_time = datetime.now()
    print('Started the backtesting at = ', start_time)

    start_time = datetime.now()
    results, _ = BACKTEST_GPU(
        df['open'].values.astype(np.float32),
        df['close'].values.astype(np.float32),
        df['ema_5'].values.astype(np.float32),
        df['ema_34'].values.astype(np.float32),
        df['atr'].values.astype(np.float32),
        params
    )
    
    end_time = datetime.now()
    print('Ended the backtesting at = ', end_time)
    print("Backtesting completed in", end_time - start_time, "seconds")

    num_params = params.shape[0]
    # The first num_params*14 values correspond to the main results.
    main_results = results[:num_params * 14].reshape((num_params, 14))
    # The remaining values correspond to the profitable trades.
    profitable_trades = results[num_params * 14:].reshape((num_params, 100))

    # Sort the profitable trades for each parameter set and pick the top 5 (largest profits)
    sorted_trades = cp.flip(cp.sort(profitable_trades, axis=1), axis=1)
    top_5_trades = sorted_trades[:, :5]

    # Transfer GPU arrays to CPU.
    main_results_cpu = main_results.get()
    # 'params' is already a NumPy array, so no need to call .get()
    params_cpu = params
    top_5_trades_cpu = top_5_trades.get()

    # Build output data for each parameter combination.
    output_data = []
    for i in range(num_params):
        res = main_results_cpu[i]
        p = params_cpu[i]
        trades = top_5_trades_cpu[i]

        output_data.append([
            p[0],              # Long length
            p[1],              # Short length
            p[3],              # Long Gap
            p[4],              # Short Gap
            p[5],              # Percentage
            p[2],              # ATR value
            p[6],              # TSL Long Percentage
            p[7],              # TSL Short Percentage
            res[0],            # Total Trades
            res[1],            # Long Trades
            res[2],            # Short Trades
            ','.join(f"{x:.2f}" for x in trades if x > 0),  # Top Profitable Trades (as a string)
            res[8],            # Win Percentage
            res[7],            # Net Profit
            res[3],            # Total Profit
            res[4]             # Total Loss
        ])

    post_processing = datetime.now()
    print("Post processing time = ", post_processing - end_time, "seconds \n")
    return output_data

if __name__ == "__main__":
    # Make sure 'data_ohlc_feb1.csv' is in the working directory.
    df = pd.read_csv("data_ohlc_feb1.csv")
    output_data = TRADING_BOT(df)
    table_generation_start_time = datetime.now()
    # Create a DataFrame from the output data with appropriate column names.
    output_df = pd.DataFrame(output_data, columns=[
        'Long length', 'Short length', 'Long Gap', 'Short Gap', 'percentage', 'ATR value',
        'TSL Long Percentage', 'TSL Short Percentage', 'Total Trades', 'Long Trades',
        'Short Trades', 'Top Profitable Trades', 'Win Percentage', 'Net Profit',
        'Total Profit', 'Total Loss'
    ])
    
    # Sort by net profit (descending) to see the best-performing parameter combinations.
    output_df = output_df.sort_values('Net Profit', ascending=False)
    output_df.to_csv("cuda_test_dummy_feb8_report_on_feb1_data_gpu_with_tsl.csv", index=False)
    table_generation_end_time = datetime.now()
    # Display the top 20 parameter combinations based on net profit.
    print('\nTop 20 highest net profit =')
    print(output_df[:20])
    print('\nTrading Bot Ended\n')
    print("Table generated in = ", table_generation_end_time - table_generation_start_time, " seconds \n")
    
    
    
    