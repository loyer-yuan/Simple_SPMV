import subprocess
import os
import re
from datetime import datetime
import pandas as pd
import glob
import itertools
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="实验参数配置")

    # 添加参数并设置默认值
    parser.add_argument("-s", "--skip-existing",
                        action="store_true",
                        default=False,
                        help="是否跳过已有结果(默认:False)")

    parser.add_argument("-f", "--force-run",
                        action="store_true",
                        default=False,
                        help="直接运行所有参数，结果输出到屏幕")

    parser.add_argument("-m", "--spmatrix-dir",
                        type=str,
                        default="../spmat",
                        help=f"矩阵文件目录(默认:'../spmat')")

    parser.add_argument("-t", "--threads",
                        type=lambda s: list(map(int, s.split(','))),
                        default=[1],
                        help="线程参数值，逗号分隔整数(默认:1)")

    parser.add_argument("-o", "--output-csv",
                        type=str,
                        default="experiment_results.csv",
                        help=f"结果文件名(默认:'experiment_results.csv')")

    parser.add_argument("-l", "--log-dir",
                        type=str,
                        default="./experiment_logs",
                        help=f"日志保存目录(默认:'./experiment_logs')")

    args = parser.parse_args()

    # 用户配置参数
    force_run = args.force_run          # 是否强制运行所有参数
    skip_existing = args.skip_existing  # 是否跳过已有结果
    matrix_dir = args.spmatrix_dir      # 矩阵文件目录
    thread_values = args.threads        # 线程参数值
    output_csv = args.output_csv        # 结果文件
    log_dir = args.log_dir              # 日志保存目录

    # 文件名处理函数
    def format_filename(file_path):
        """将完整路径转换为无扩展名的文件名"""
        return os.path.splitext(os.path.basename(file_path))[0]

    # 生成当前参数组合
    # file_paths = [os.path.abspath(p) for p in glob.glob(os.path.join(matrix_dir, "*.mtx"))]
    # 先检查路径是文件还是目录
    if os.path.isfile(matrix_dir):
        # 如果是文件，直接检查扩展名
        if matrix_dir.endswith('.mtx'):
            file_paths = [os.path.abspath(matrix_dir)]
        else:
            print("Invalid file type. Please provide a .mtx file.")
            exit()
    elif os.path.isdir(matrix_dir):
        # 如果是目录，保留原逻辑
        file_paths = [os.path.abspath(p) for p in glob.glob(os.path.join(matrix_dir, "*.mtx"))]
    else:
        # 处理不存在的情况（可选）
        print(f"Invalid path: {matrix_dir}. Please provide a valid file or directory.")
        exit()

    current_params = [
        {
            'file': path,  # 保留完整路径用于命令执行
            'Threads': thread,  # 合并 OMP_NUM_THREADS 和 -t 参数
            'formatted_file': format_filename(path)
        }
        for thread, path in itertools.product(thread_values, file_paths)
    ]

    # 生成参数标识符集合(使用简化后的字段)
    current_keys = set(
        (p['Threads'], p['formatted_file'])
        for p in current_params
    )

    # 读取已有结果
    existing_df = pd.DataFrame()
    if os.path.exists(output_csv) and not force_run:
        try:
            existing_df = pd.read_csv(output_csv)
        except Exception as e:
            print(f"警告:读取结果文件失败 - {e}")
            existing_df = pd.DataFrame()

    # 参数过滤逻辑
    if skip_existing and not existing_df.empty and not force_run:
        # 获取已有结果标识符
        existing_keys = set(
            existing_df[['Threads', 'Input File']]
            .astype(str)
            .apply(tuple, axis=1)
        )

        # 过滤需要执行的参数
        param_combinations = [
            p for p in current_params
            if (p['Threads'], p['formatted_file']) not in existing_keys
        ]
    else:
        param_combinations = current_params

    if not param_combinations:
        print("所有参数组合均已存在，无需执行")
        exit()

    # 配置日志保存目录(用户可修改)
    os.makedirs(log_dir, exist_ok=True)

    # 结果存储列表
    results = []

    for params in tqdm(param_combinations, desc="Running task..."):
        # 构造环境变量
        env = os.environ.copy()
        env['OMP_NUM_THREADS'] = str(params['Threads'])

        # 构造命令参数
        command = [
            './bin/test/cpu_test_all',
            '-f', params['file'],
            '-csrRef',
            '-t', str(params['Threads'])
        ]

        # 执行命令并捕获输出
        print(f"Running command: {' '.join(command)}")
        process = subprocess.run(
            command,
            env=env,
            capture_output=True,
            text=True
        )

        # 生成日志文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = (
            f"{params['formatted_file']}_"
            f"t_{params['Threads']}_{timestamp}.log"
        )
        log_path = os.path.join(log_dir, filename)

        # 保存原始日志
        with open(log_path, 'w') as f:
            f.write(process.stdout)

        # 解析日志内容
        current_test = None
        metrics = {}

        # 定义匹配模式
        patterns = {
            'Time': r'Time:\s+([\d.]+)\s*ms',
            'Flops/s': r'Flops/s:\s+([\d.]+)\s*GFlops/s',
            'Effective Bandwidth': r'Effective Bandwidth:\s+([\d.]+)\s*GB/s',
            'All Bandwidth': r'All Bandwidth:\s+([\d.]+)\s*GB/s',
            'All Size': r'All Size:\s+([\d.]+)\s*GB',
            'Total GFlops': r'Total GFlops:\s+([\d.]+)\s*GFlops'
        }

        # 逐行解析
        for line in process.stdout.split('\n'):
            # 检测测试开始
            if line.strip().startswith('Test '):
                test_name = line.split('Test ')[1].split(' kernel.')[0].strip()
                current_test = test_name
                metrics = {'Test Name': test_name}

            # 检测测试结束
            elif line.strip().startswith('End of test!'):
                if current_test:
                    # 保存结果
                    record = {
                        'Input File': params['formatted_file'],
                        'Threads': int(params['Threads']),
                        **metrics
                    }
                    results.append(record)
                    current_test = None

            # 提取指标数据
            elif current_test:
                for metric, pattern in patterns.items():
                    match = re.search(pattern, line)
                    if match:
                        try:
                            value = float(match.group(1))
                            metrics[metric] = value
                        except ValueError:
                            print(f"[Mat: {params['formatted_file']}, T: {params['Threads']}]: "
                                    "无法转换值: {match.group(1)}")

    # 结果合并逻辑
    if not skip_existing and not existing_df.empty and not force_run:
        # 创建保留掩码:保留不在当前参数中的旧记录
        preserve_mask = ~existing_df.apply(
            lambda row: (
                row['Threads'],
                str(row['Input File'])
            ) in current_keys,
            axis=1
        )
        preserved_df = existing_df[preserve_mask]
        final_df = pd.concat([preserved_df, pd.DataFrame(results)])
    else:
        final_df = pd.concat([existing_df, pd.DataFrame(results)])

    # 保存结果(确保列顺序)
    column_order = [
        'Input File', 'Threads', 'Test Name',
        'Time', 'Flops/s', 'Effective Bandwidth',
        'All Bandwidth', 'All Size', 'Total GFlops'
    ]

    if force_run:
        # 如果强制运行，直接输出到屏幕
        print("\nExperiment results:")
        print(final_df[column_order].sort_values(by=['Input File', 'Test Name', 'Threads']).to_string(index=False))
    else:
        try:
            final_df[column_order].drop_duplicates(
                subset=['Threads', 'Input File', 'Test Name'],
                keep='last'
            ).sort_values(by=['Input File', 'Test Name', 'Threads']).to_csv(output_csv, index=False)
            print(f"结果已保存至 {output_csv}")
            # 打印表格预览
            print("\nPreview of results:")
            print(final_df.head())
        except Exception as e:
            print(f"保存失败: {e}")

if __name__ == "__main__":
    main()