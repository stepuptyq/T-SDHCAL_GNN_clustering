import pandas as pd

def process_particle_data(input_file, output_file):
    # 读取所需列数据
    # Read required column data
    df = pd.read_csv(
        input_file,
        usecols=['event_index', 'x_coords', 'y_coords', 'z_coords', 'time']
    )
    
    # 定义处理每个粒子组的函数
    # Define function to process each particle group
    def process_group(group):
        # 计算时间中位数和IQR
        # Calculate time median and IQR (Interquartile Range)
        time_series = group['time']
        median_time = time_series.median()
        q1 = time_series.quantile(0.25)
        q3 = time_series.quantile(0.75)
        iqr = q3 - q1
        
        # 设置过滤阈值（可调整1.5这个系数）
        # Set filtering threshold (adjustable coefficient 1.5)
        lower_bound = median_time - 1.5 * iqr
        upper_bound = median_time + 1.5 * iqr
        
        # 过滤异常时间点
        # Filter abnormal time points
        filtered = group[
            (group['time'] >= lower_bound) & 
            (group['time'] <= upper_bound)
        ].copy()
        
        # 执行标准化
        # Perform standardization
        filtered[['x_coords', 'y_coords', 'z_coords']] /= 1000
        filtered['time'] /= 5
        
        return filtered
    
    # 分组处理数据
    # Process data by groups
    processed_df = df.groupby('event_index', group_keys=False).apply(process_group)
    
    # 保存处理结果
    # Save processed results
    processed_df.to_csv(output_file, index=False)
    print(f"处理完成！结果已保存至 {output_file}")

# 使用示例
# Usage example
process_particle_data(
    input_file="D:\\GNN\\simulate_data\\transfer_9435897_files_031472cb\\20kproton_Emin1Emax100_digitized_hits_continuous_merged.csv",
    output_file="D:\\GNN\\simulate_data\\transfer_9435897_files_031472cb\\20kproton_Emin1Emax100_digitized_hits_continuous_merged_time_filtered.csv"
)