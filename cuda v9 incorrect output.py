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


# Define the GPU kernel for backtesting.
# Note: The kernel reserves 14 result slots per parameter (only 9 are currently used)
BACKTEST_KERNAL = cp.RawKernel(
    r'''
    extern "C" __global__
    void BACKTEST_KERNAL(
        const float* open_price, const float* close_price,
        const float* ema_5, const float* ema_34,
        const float* params,
        float* results,
        const int num_candles,
        const int num_params)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= num_params) return;

        // Load parameters for this thread (8 parameters per set)
        float long_ema_deviation = params[tid * 8];
        float short_ema_deviation = params[tid * 8 + 1];
        float long_gap = params[tid * 8 + 2];
        float short_gap = params[tid * 8 + 3];
        float ema34_LONG_variation = params[tid * 8 + 4];
        float ema34_SHORT_variation = params[tid * 8 + 5];
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

        for (int j = 2; j < num_candles; j++) {
            float current_candle_close = close_price[j];

            bool l1 = ema_5[j] > ema_34[j];
            bool l2 = (ema_5[j] - ema_34[j]) >  long_ema_deviation;
            bool l3 = (ema_34[j] - ema_34[j-4]) > ema34_LONG_variation;

            bool s1 = ema_5[j] < ema_34[j];
            bool s2 = (ema_34[j] - ema_5[j]) >  short_ema_deviation;
            bool s3 = (ema_34[j-4] - ema_34[j]) > ema34_SHORT_variation;

            bool long_condition = l1 && l2 && l3;
            bool short_condition = s1 && s2 && s3;

            if (position == 0.0) {
                if (long_condition) {
                    position = 1.0;
                    entry_price = current_candle_close;
                    num_long_trades++;
                    max_gap = 0.0;
                } else if (short_condition) {
                    position = 2.0;
                    entry_price = current_candle_close;
                    num_short_trades++;
                    max_gap = 0.0;
                }
            } else if (position == 1.0) { // Long position management
                float current_gap = ema_5[j] - ema_34[j];
                max_gap = fmaxf(max_gap, current_gap);

                if ((current_gap < max_gap * long_gap) || (close_price[j] < ema_34[j])) {
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

                if ((current_gap < max_gap * short_gap) || (close_price[j] > ema_34[j])) {
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

        results[result_base_idx] = num_trades;
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

def BACKTEST_GPU(open_price, close_price, ema_5, ema_34, params):
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
        'params': cp.asarray(params, dtype=cp.float32)
    }

    # Launch the GPU kernel
    BACKTEST_KERNAL((blocks_per_grid,), (threads_per_block,), (
        gpu_data['open'],
        gpu_data['close'],
        gpu_data['ema_5'],
        gpu_data['ema_34'],
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

    # Create a list of parameter combinations.
    params_list = [
        (long_ema_deviation, short_ema_deviation, long_gap, short_gap, ema34_LONG_variation, ema34_SHORT_variation)
        for long_ema_deviation in np.arange(50, 1550, 100)         
        for short_ema_deviation in np.arange(50, 1550, 100)       
        for long_gap in np.linspace(0.4, 0.9, 6)
        for short_gap in np.linspace(0.4, 0.9, 6)
        for ema34_LONG_variation in np.arange(0, 350, 50)
        for ema34_SHORT_variation in np.arange(0, 350, 50)
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
            p[2],              # Long Gap
            p[3],              # Short Gap
            p[4],              # ema34 long Length
            p[5],              # ema34 short Length
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
    INPUT_FILE = r"data_ohlc_feb1.csv"
    OUTPUT_FILE_NET_PROFIT = r"net_profit.csv"
    
    df = pd.read_csv(INPUT_FILE)
    output_data = TRADING_BOT(df)
    table_generation_start_time = datetime.now()
    # Create a DataFrame from the output data with appropriate column names.
    output_df = pd.DataFrame(output_data, columns=[
        'Long EMA deviation', 'Short EMA deviation', 'Long Gap', 'Short Gap', 'ema34 Long variation', 'ema34 Short variation', 'Total Trades', 'Long Trades',
        'Short Trades', 'Top Profitable Trades', 'Win Percentage', 'Net Profit', 'Total Profit', 'Total Loss'
    ])
    
    # Sort by net profit (descending) to see the best-performing parameter combinations.
    output_df_net_profit = output_df.sort_values('Net Profit', ascending=False)
    output_df_net_profit.to_csv(OUTPUT_FILE_NET_PROFIT, index=False)
    
    table_generation_end_time = datetime.now()
    print('\nTop Highest net profit =')
    print(output_df_net_profit)
    print('\nTrading Bot Ended\n')
    print("Table generated in = ", table_generation_end_time - table_generation_start_time, " seconds \n")