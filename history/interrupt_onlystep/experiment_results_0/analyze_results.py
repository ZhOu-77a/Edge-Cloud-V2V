import pandas as pd
import plotly.express as px
import os

# === 路径配置 ===
CSV_PATH = "experiment_results/experiment_report.csv"
OUTPUT_HTML = "experiment_results/interactive_analysis.html"

def plot_interactive_3d():
    if not os.path.exists(CSV_PATH):
        print("❌ 无法生成图表：CSV文件不存在，请先运行 CLIP 评分。")
        return
        
    df = pd.read_csv(CSV_PATH)
    
    # 检查是否有评分数据
    if 'Quality_Score' not in df.columns:
        print("⚠️ 警告：CSV中缺少 Quality_Score 列，将使用模拟数据展示。")
        df['Quality_Score'] = 100 - (1 - df['Ratio']) * 50
    
    # 使用 Plotly 创建交互式三维散点图
    fig = px.scatter_3d(
        df, 
        x='N', 
        y='Ratio', 
        z='Latency(s)',
        color='Quality_Score',  # 颜色代表质量
        size='m',               # 点的大小代表 m
        hover_data=['ID', 'm'],  # 鼠标悬停时显示的信息
        color_continuous_scale='Viridis', # 亮色代表高质量
        title='V2X Optimization: Drag to Rotate | Scroll to Zoom',
        labels={
            'N': 'Original Steps',
            'Ratio': 'Interrupt Ratio',
            'Latency(s)': 'Core Latency',
            'Quality_Score': 'CLIP Quality'
        }
    )

    # 优化视觉效果
    fig.update_layout(
        scene=dict(
            xaxis_title='N (Steps)',
            yaxis_title='Ratio (n/N)',
            zaxis_title='Latency (s)'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    # 保存为 HTML
    fig.write_html(OUTPUT_HTML)
    print(f"✅ 交互式 3D 分析图已生成：{OUTPUT_HTML}")
    print("👉 请直接双击该文件使用浏览器打开，即可进行旋转观察。")
    
    # 如果你在本地运行，会自动弹开浏览器
    # fig.show()

if __name__ == "__main__":
    plot_interactive_3d()