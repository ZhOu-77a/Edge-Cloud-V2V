import pandas as pd
import plotly.express as px
import os

# === 路径配置 ===
CSV_PATH = "experiment_results/experiment_report.csv"
OUTPUT_HTML = "experiment_results/interactive_analysis_lpips.html"

def plot_interactive_3d():
    if not os.path.exists(CSV_PATH):
        print("❌ 无法生成图表：CSV文件不存在，请先运行 LPIPS.py。")
        return
        
    df = pd.read_csv(CSV_PATH)
    
    # 检查是否有 LPIPS 数据
    if 'LPIPS_Score' not in df.columns:
        print("❌ CSV中缺少 LPIPS_Score 列，请先运行 LPIPS.py")
        return
    
    # 确保用于颜色的列存在 (如果脚本没生成，这里补救一下)
    if 'Quality_Index_InvLPIPS' not in df.columns:
        max_val = df['LPIPS_Score'].max()
        df['Quality_Index_InvLPIPS'] = (max_val - df['LPIPS_Score']) * 100

    # 使用 Plotly 创建交互式三维散点图
    fig = px.scatter_3d(
        df, 
        x='N', 
        y='Ratio', 
        z='Latency(s)',
        color='Quality_Index_InvLPIPS',  # 使用反转后的分数上色 (高=好)
        size='m',               
        hover_data={
            'ID': True, 
            'm': True, 
            'LPIPS_Score': ':.4f', # 显示真实的 LPIPS
            'Quality_Index_InvLPIPS': False, # 隐藏反转分
            'Latency(s)': ':.2f'
        },  
        color_continuous_scale='Viridis', # 亮黄=高质量(低LPIPS)，深紫=低质量
        title='V2X Optimization (LPIPS Metric): Lower LPIPS is Better',
        labels={
            'N': 'Original Steps (N)',
            'Ratio': 'Interrupt Ratio (n/N)',
            'Latency(s)': 'Core Latency',
            'Quality_Index_InvLPIPS': 'Quality Index (Inv LPIPS)'
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
    
    # 添加注解告诉用户怎么看
    fig.add_annotation(
        text="Color Legend: Yellow = Clear Video (Low LPIPS), Purple = Noisy/Snowy (High LPIPS)",
        xref="paper", yref="paper",
        x=0, y=1, showarrow=False,
        font=dict(size=12, color="red")
    )

    # 保存为 HTML
    fig.write_html(OUTPUT_HTML)
    print(f"✅ 交互式 3D 分析图已生成：{OUTPUT_HTML}")
    print("👉 颜色越亮(黄)代表 LPIPS 越低(质量越好)。")

if __name__ == "__main__":
    plot_interactive_3d()