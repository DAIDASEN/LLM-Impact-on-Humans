import pandas as pd
import os

def check_clusters():
    # 1. 读取 Survey.csv，建立 Title 到 Cluster 的映射
    survey_file = 'Survey.csv'
    if not os.path.exists(survey_file):
        print(f"未找到 {survey_file}")
        return

    try:
        df_survey = pd.read_csv(survey_file)
        # 假设 Survey.csv 中包含 'Title' 和 'Cluster' (或 'Class') 列
        # 这里做一些列名的自动适配
        title_col = 'Title' if 'Title' in df_survey.columns else df_survey.columns[0] # 默认第一列为Title
        # 尝试寻找 Cluster 或 Class 列
        cluster_col = next((c for c in df_survey.columns if c in ['Cluster', 'Class']), None)
        
        if not cluster_col:
            print("在 Survey.csv 中未找到 'Cluster' 或 'Class' 列。")
            return

        # 创建映射字典：Title -> Cluster
        # 使用 strip() 去除可能存在的首尾空格
        survey_map = dict(zip(df_survey[title_col].astype(str).str.strip(), df_survey[cluster_col]))
        print(f"已加载 Survey.csv，包含 {len(survey_map)} 条记录。")

    except Exception as e:
        print(f"读取 Survey.csv 出错: {e}")
        return

    # 2. 遍历 cluster_i.csv 文件 (i=0, 1, 2)
    for i in range(3):
        file_name = f'cluster_{i}.csv'
        if os.path.exists(file_name):
            print(f"\n正在检查: {file_name} ...")
            try:
                df_c = pd.read_csv(file_name)
                
                # 假设 cluster_i.csv 只有 Title，或第一列是 Title
                c_title_col = 'Title' if 'Title' in df_c.columns else df_c.columns[0]
                
                # 执行查找
                # map 找不到的值会变为 NaN
                df_c['Found_Cluster'] = df_c[c_title_col].astype(str).str.strip().map(survey_map)
                
                # 打印前几行结果
                print(df_c.head())
                
                # 统计一下匹配到的 Cluster 分布
                print("匹配结果统计:")
                print(df_c['Found_Cluster'].value_counts(dropna=False))
                
                # 如果需要保存结果
                output_file = f'checked_cluster_{i}.csv'
                df_c.to_csv(output_file, index=False)
                print(f"检查结果已保存至: {output_file}")
                
            except Exception as e:
                print(f"处理 {file_name} 时出错: {e}")
        else:
            print(f"\n未找到文件: {file_name}")

if __name__ == "__main__":
    check_clusters()