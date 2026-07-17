import pandas as pd
import numpy as np

class DataCleaner:
    def __init__(self, data):
        self.data = data
        self.removed_columns = []
        self.winsorized_columns = []
        self.standardized_columns = []
    
    def handle_missing_values(self, threshold=0.1):
        """
        处理缺失值：
        1. 计算每个变量的缺失率
        2. 剔除缺失率大于10%的变量
        """
        missing_rates = self.data.isnull().mean()
        high_missing_cols = missing_rates[missing_rates > threshold].index.tolist()
        
        if high_missing_cols:
            print(f"剔除缺失率大于{threshold*100}%的变量: {high_missing_cols}")
            self.removed_columns.extend(high_missing_cols)
            self.data = self.data.drop(columns=high_missing_cols)
        
        return self.data
    
    def detect_outliers(self, method='3sigma'):
        """
        异常值检测：
        使用3σ原则检测异常值
        """
        outliers = {}
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if method == '3sigma':
                mean = self.data[col].mean()
                std = self.data[col].std()
                lower_bound = mean - 3 * std
                upper_bound = mean + 3 * std
                col_outliers = self.data[(self.data[col] < lower_bound) | (self.data[col] > upper_bound)].index
                outliers[col] = col_outliers.tolist()
        
        return outliers
    
    def handle_outliers(self, method='winsorize'):
        """
        异常值处理：
        使用Winsorize缩尾法，将异常值替换为边界值
        """
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if method == 'winsorize':
                mean = self.data[col].mean()
                std = self.data[col].std()
                lower_bound = mean - 3 * std
                upper_bound = mean + 3 * std
                
                # 替换异常值为边界值
                self.data[col] = np.clip(self.data[col], lower_bound, upper_bound)
                self.winsorized_columns.append(col)
        
        return self.data
    
    def standardize_data(self, method='zscore'):
        """
        数据标准化：
        使用Z-score标准化，使均值=0，标准差=1
        """
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if method == 'zscore':
                mean = self.data[col].mean()
                std = self.data[col].std()
                if std > 0:  # 避免除以0
                    self.data[col] = (self.data[col] - mean) / std
                    self.standardized_columns.append(col)
        
        return self.data
    
    def check_consistency(self, total_col, component_cols):
        """
        一致性检查：
        验证总分是否等于各分项之和
        """
        # 计算各分项之和
        calculated_total = self.data[component_cols].sum(axis=1)
        
        # 比较计算的总分与实际总分
        consistency = np.isclose(calculated_total, self.data[total_col])
        
        # 找出不一致的记录
        inconsistent_rows = self.data[~consistency].index.tolist()
        
        if inconsistent_rows:
            print(f"发现{len(inconsistent_rows)}条记录的总分与各分项之和不一致")
        else:
            print("所有记录的总分与各分项之和一致")
        
        return inconsistent_rows
    
    def clean_data(self, total_col=None, component_cols=None):
        """
        完整的数据清洗流程
        """
        print("1. 处理缺失值...")
        self.handle_missing_values()
        
        print("\n2. 检测异常值...")
        outliers = self.detect_outliers()
        for col, outlier_indices in outliers.items():
            if outlier_indices:
                print(f"变量 {col} 检测到 {len(outlier_indices)} 个异常值")
        
        print("\n3. 处理异常值...")
        self.handle_outliers()
        
        print("\n4. 数据标准化...")
        self.standardize_data()
        
        if total_col and component_cols:
            print("\n5. 一致性检查...")
            self.check_consistency(total_col, component_cols)
        
        print("\n数据清洗完成！")
        return self.data

# 示例用法
if __name__ == "__main__":
    # 读取数据
    # 示例1：读取CSV文件
    # data = pd.read_csv('data.csv')
    
    # 示例2：读取Excel文件
    # data = pd.read_excel('data.xlsx')
    
    # 示例3：生成示例数据
    np.random.seed(42)
    data = pd.DataFrame({
        'A': np.random.normal(100, 10, 100),
        'B': np.random.normal(50, 5, 100),
        'C': np.random.normal(20, 2, 100),
        'total': np.random.normal(170, 15, 100)
    })
    
    # 人为添加缺失值
    data.loc[0:4, 'A'] = np.nan  # 5%缺失率
    data.loc[0:14, 'B'] = np.nan  # 15%缺失率（会被剔除）
    
    # 人为添加异常值
    data.loc[5, 'A'] = 300  # 异常值
    data.loc[6, 'C'] = -100  # 异常值
    
    print("原始数据形状:", data.shape)
    print("原始数据前5行:")
    print(data.head())
    
    # 创建数据清洗器
    cleaner = DataCleaner(data)
    
    # 执行完整的清洗流程
    cleaned_data = cleaner.clean_data(total_col='total', component_cols=['A', 'C'])
    
    print("\n清洗后数据形状:", cleaned_data.shape)
    print("清洗后数据前5行:")
    print(cleaned_data.head())
    
    # 输出清洗信息
    print("\n清洗信息:")
    print(f"剔除的变量: {cleaner.removed_columns}")
    print(f"进行缩尾处理的变量: {cleaner.winsorized_columns}")
    print(f"进行标准化处理的变量: {cleaner.standardized_columns}")
