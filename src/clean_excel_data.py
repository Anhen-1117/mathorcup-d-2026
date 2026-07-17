import pandas as pd
from data_cleaner import DataCleaner

# 读取Excel文件
excel_file = '附件2：验证数据集.xlsx'

# 读取所有工作表
xls = pd.ExcelFile(excel_file)
print(f"Excel文件包含以下工作表: {xls.sheet_names}")

# 处理每个工作表
for sheet_name in xls.sheet_names:
    print(f"\n处理工作表: {sheet_name}")
    
    # 读取工作表数据
    data = pd.read_excel(excel_file, sheet_name=sheet_name)
    print(f"原始数据形状: {data.shape}")
    print(f"原始数据列名: {list(data.columns)}")
    
    # 简单查看数据
    print("\n数据前5行:")
    print(data.head())
    
    # 创建数据清洗器
    cleaner = DataCleaner(data)
    
    # 执行清洗流程
    # 这里需要根据实际数据结构调整参数
    # 假设数据中有总分列和分项列
    # 请根据实际数据结构修改以下参数
    total_col = None  # 替换为实际的总分列名
    component_cols = []  # 替换为实际的分项列名
    
    # 尝试自动识别总分列
    for col in data.columns:
        if '总分' in col or 'total' in col.lower():
            total_col = col
            break
    
    # 执行清洗
    cleaned_data = cleaner.clean_data(total_col=total_col, component_cols=component_cols)
    
    # 保存清洗后的数据
    output_file = f'cleaned_{sheet_name}.csv'
    cleaned_data.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n清洗后的数据已保存为: {output_file}")
    print(f"清洗后数据形状: {cleaned_data.shape}")
    
    # 输出清洗信息
    print("\n清洗信息:")
    print(f"剔除的变量: {cleaner.removed_columns}")
    print(f"进行缩尾处理的变量: {cleaner.winsorized_columns}")
    print(f"进行标准化处理的变量: {cleaner.standardized_columns}")

print("\n所有工作表处理完成！")
