import pandas as pd
import os
import glob
from pathlib import Path
from datetime import datetime

# 从 camera_control.py 提取的颜色检测阈值
COLOR_THRESHOLDS = {
    "lower_hsv": [67, 88, 83],
    "upper_hsv": [67, 89, 100],
    "min_pixels_ratio": 0.6
}


def read_excel(filepath):
    df = pd.read_excel(filepath)
    return df


def check_color_threshold(hsv_value, matched_ratio):
    """
    判断HSV值是否达到驱动小车行进的阈值

    Args:
        hsv_value: [H, S, V] HSV值列表
        matched_ratio: 匹配像素比例

    Returns:
        tuple: (是否达到阈值, 当前时间字符串)
    """
    lower = COLOR_THRESHOLDS["lower_hsv"]
    upper = COLOR_THRESHOLDS["upper_hsv"]
    min_ratio = COLOR_THRESHOLDS["min_pixels_ratio"]

    # 检查HSV值是否在阈值范围内
    hsv_in_range = (lower[0] <= hsv_value[0] <= upper[0] and
                   lower[1] <= hsv_value[1] <= upper[1] and
                   lower[2] <= hsv_value[2] <= upper[2])

    # 检查像素比例是否达到阈值
    ratio_threshold_met = matched_ratio >= min_ratio

    # 记录当前时间
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    # 达到阈值条件：HSV在范围内且像素比例超过阈值
    threshold_met = hsv_in_range and ratio_threshold_met

    return threshold_met, current_time


def analyze_color_change(df, h_col='H (0-360)', s_col='S (0-100)', v_col='V (0-100)', ratio_column='matched_ratio'):
    """
    分析数据框中的颜色变化，记录达到阈值的时间和HSV值

    Args:
        df: pandas DataFrame
        h_col: H通道列名 (0-360)
        s_col: S通道列名 (0-100)
        v_col: V通道列名 (0-100)
        ratio_column: 匹配像素比例列名

    Returns:
        list: 达到阈值的事件列表，每个事件包含时间和HSV值
    """
    threshold_events = []

    for idx, row in df.iterrows():
        # 从三列分别读取H, S, V值
        try:
            h_value = float(row.get(h_col, 0))
            s_value = float(row.get(s_col, 0))
            v_value = float(row.get(v_col, 0))
            hsv_value = [h_value, s_value, v_value]
        except (ValueError, TypeError):
            continue

        matched_ratio = row.get(ratio_column, 0)
        if isinstance(matched_ratio, str):
            try:
                matched_ratio = float(matched_ratio)
            except:
                matched_ratio = 0

        threshold_met, current_time = check_color_threshold(hsv_value, matched_ratio)

        if threshold_met:
            event = {
                'timestamp': current_time,
                'hsv_value': hsv_value,
                'matched_ratio': matched_ratio,
                'row_index': idx
            }
            threshold_events.append(event)
            print(f"[阈值触发] 时间: {current_time}, HSV: {hsv_value}, 比例: {matched_ratio:.2f}")

    return threshold_events


def main():
    filepath = r'C:\Users\abc\Desktop\ml\640.xlsx'
    df = read_excel(filepath)
    print("=" * 50)
    print(f"读取数据: {len(df)} 行")
    print(f"颜色阈值配置:")
    print(f"  HSV下限: {COLOR_THRESHOLDS['lower_hsv']}")
    print(f"  HSV上限: {COLOR_THRESHOLDS['upper_hsv']}")
    print(f"  最小像素比例: {COLOR_THRESHOLDS['min_pixels_ratio']}")
    print("=" * 50)

    # 显示数据列名，帮助用户了解数据结构
    print(f"\n数据列: {list(df.columns)}")
    print(df.head())

    # 检查必要的列是否存在
    required_cols = ['H (0-360)', 'S (0-100)', 'V (0-100)']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        print(f"\n警告: 缺少以下列: {missing_cols}")
        print("可用的列:", list(df.columns))
        return

    print("\n" + "=" * 50)
    print("开始阈值检测:")
    print("=" * 50)

    # 从Excel文件读取数据进行阈值检测
    # matched_ratio列如果不存在，默认为1.0（表示只检查HSV范围）
    if 'matched_ratio' not in df.columns:
        print("注意: 未找到 'matched_ratio' 列，将默认使用1.0")
        df['matched_ratio'] = 1.0

    events = analyze_color_change(df)

    print(f"\n总共触发阈值次数: {len(events)}")

    # 保存结果到Excel
    if events:
        result_df = pd.DataFrame(events)
        output_path = r'C:\Users\abc\Desktop\ml\threshold_events.xlsx'
        result_df.to_excel(output_path, index=False)
        print(f"\n阈值触发事件已保存到: {output_path}")
    else:
        print("\n未检测到达到阈值的事件")


if __name__ == "__main__":
    main()
