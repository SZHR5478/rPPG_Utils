# *evaluation.py*
- - -
## *read_ppg_hr_csv(data_path, wave_fs=60)*
负责对 **[CMS60D血氧仪](https://www.contecmed.com.cn/productinfo/817892.html)**  保存的PPG信号及其对应心率数据（**.csv格式**）进行读取处理, PPG信号默认采集频率为**60HZ**
- - -
## *calculate_metrics(ppg_hrs, fs=60, window_size=300)*
使用Fast Fourier transform (FFT)和Peak Detection 对PPG信号进行心率预测，并与真实心率进行对比
- - -