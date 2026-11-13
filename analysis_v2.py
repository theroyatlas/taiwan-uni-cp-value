"""由隨附 CSV 重建原有學門分析；輸出必須指向新的資料夾。

--no-plots 僅重建資料，不需 matplotlib / seaborn；排名仍是 P50 的 OLS 殘差。
"""
import argparse
import json
from pathlib import Path

from data_pipeline import DATA, load_and_check

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data-dir', type=Path, default=DATA)
parser.add_argument('--output-dir', type=Path, required=True,
                    help='必須為尚不存在的輸出資料夾，保護原始資料與既有圖片')
parser.add_argument('--no-plots', action='store_true', help='只重建 CSV，跳過歷史靜態圖')
args = parser.parse_args()

df_out, report = load_and_check(args.data_dir)
# 出圖依賴在寫檔前檢查；一般 dashboard / 離線驗證不需要安裝。
if not args.no_plots:
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats

THIS = args.output_dir.resolve()
THIS.mkdir(parents=True, exist_ok=False)
df_out.to_csv(THIS / 'final_dataset_v2.csv', index=False, encoding='utf-8-sig')
print(json.dumps(report, ensure_ascii=False, indent=2))
print(f'輸出：{THIS / "final_dataset_v2.csv"}')
if args.no_plots:
    raise SystemExit(0)

df = df_out.copy()
sns.set_theme(style='whitegrid')
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Microsoft YaHei',
                                    'Arial Unicode MS', 'SimHei']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
slope, intercept, r, p, se = stats.linregress(df['median_pr_pct'], df['P50'].astype(float))
r2 = r ** 2


# ───────────────────────────────────────────────
# 5. 圖 1: P50 散佈圖 + error bar (P25-P75)
# ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 7.5))
palette = {'應用 STEM': '#1f77b4', '基礎 STEM': '#2ca02c', '人文社科': '#d62728'}

for g, sub in df.groupby('三分群'):
    color = palette[g]
    ax.errorbar(
        sub['median_pr_pct'], sub['P50'].astype(float),
        yerr=[sub['P50'].astype(float) - sub['P25'].astype(float),
              sub['P75'].astype(float) - sub['P50'].astype(float)],
        fmt='o', color=color, alpha=0.85, markersize=8,
        elinewidth=1.5, capsize=4, label=g,
    )

# 回歸線
xs = np.linspace(df['median_pr_pct'].min() - 3,
                 df['median_pr_pct'].max() + 3, 100)
ax.plot(xs, slope * xs + intercept, '--', color='black', linewidth=1.5,
        label=f'OLS: P50 = {slope:.0f}·PR + {intercept:,.0f}  (R²={r2:.3f})')

# 標每個點
for _, row in df.iterrows():
    ax.annotate(row['short_name'],
                (row['median_pr_pct'], row['P50']),
                fontsize=8.5, xytext=(5, 5), textcoords='offset points',
                alpha=0.85)

ax.set_xlabel('中位錄取分數百分位（UAC 114 學年）', fontsize=12)
ax.set_ylabel('薪資 P50 中位數 + P25-P75 區間 (元/月)', fontsize=12)
ax.set_title('學門錄取分數 vs 薪資中位數（線=OLS, 上下線=P25/P75 區間）',
             fontsize=13)
ax.yaxis.set_major_formatter(
    plt.matplotlib.ticker.FuncFormatter(lambda v, _: f'{int(v):,}'))
ax.legend(loc='best', fontsize=10)
fig.tight_layout()
fig.savefig(THIS / 'fig_p50_scatter.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('  fig_p50_scatter.png')


# ───────────────────────────────────────────────
# 6. 圖 2: 殘差 ranking (基於 P50)
# ───────────────────────────────────────────────
df_p = df_out.copy()
df_p['kind'] = df_p['residual_P50'].apply(
    lambda v: '被低估 (殘差>0)' if v >= 0 else '被高估 (殘差<0)')

fig, ax = plt.subplots(figsize=(11, 9))
pal2 = {'被低估 (殘差>0)': '#1f77b4', '被高估 (殘差<0)': '#d62728'}
sns.barplot(data=df_p, y='short_name', x='residual_P50',
            hue='kind', dodge=False, order=df_p['short_name'],
            palette=pal2, ax=ax)
ax.axvline(0, color='black', linewidth=0.8)
for _, row in df_p.iterrows():
    ax.text(row['residual_P50'] + (200 if row['residual_P50'] >= 0 else -200),
            list(df_p['short_name']).index(row['short_name']),
            f"{row['residual_P50']:+,}",
            va='center',
            ha='left' if row['residual_P50'] >= 0 else 'right',
            fontsize=9)
cur_min, cur_max = ax.get_xlim()
span = cur_max - cur_min
ax.set_xlim(cur_min - span * 0.10, cur_max + span * 0.10)
ax.set_xlabel('殘差（P50 實際 − OLS 預測, 元/月）', fontsize=11)
ax.set_ylabel('學門', fontsize=11)
ax.set_title('控制錄取分數後，各學門 P50 殘差排名', fontsize=13)
ax.xaxis.set_major_formatter(
    plt.matplotlib.ticker.FuncFormatter(lambda v, _: f'{int(v):+,}'))
ax.legend(loc='lower right', fontsize=9)
fig.tight_layout()
fig.savefig(THIS / 'fig_residual_ranking.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('  fig_residual_ranking.png')


# ───────────────────────────────────────────────
# 7. 圖 3: 三分群 boxplot (P50) + ANOVA
# ───────────────────────────────────────────────
groups = [g['P50'].astype(float).values for _, g in df.groupby('三分群')]
f_stat, p_val = stats.f_oneway(*groups)
print(f'\n[ANOVA on P50 by 三分群]')
print(f'  F = {f_stat:.3f}, p = {p_val:.4e}')

# Also ANOVA on residual_P50
groups_r = [g['residual_P50'].astype(float).values for _, g in df.groupby('三分群')]
f2, p2 = stats.f_oneway(*groups_r)
print(f'[ANOVA on residual_P50 by 三分群]')
print(f'  F = {f2:.3f}, p = {p2:.4e}')

ordering = ['應用 STEM', '基礎 STEM', '人文社科']
fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))

for ax, ycol, ylabel, title, anova in [
    (axes[0], 'P50', 'P50 中位數薪資 (元/月)',
     f'P50 by 三分群 (ANOVA F={f_stat:.2f}, p={p_val:.2e})',
     None),
    (axes[1], 'residual_P50', '殘差 (元/月)',
     f'殘差 by 三分群 (ANOVA F={f2:.2f}, p={p2:.2e})',
     None),
]:
    sns.boxplot(data=df, x='三分群', y=ycol, order=ordering,
                hue='三分群', palette=palette, legend=False,
                width=0.55, fliersize=4, ax=ax)
    sns.stripplot(data=df, x='三分群', y=ycol, order=ordering,
                  color='black', alpha=0.55, size=6.5, ax=ax)
    for _, row in df.iterrows():
        try:
            xpos = ordering.index(row['三分群'])
            ax.annotate(row['short_name'],
                        (xpos, row[ycol]),
                        fontsize=7.5, alpha=0.7,
                        xytext=(8, 0), textcoords='offset points',
                        va='center')
        except ValueError:
            pass
    if ycol == 'residual_P50':
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('')
    ax.set_ylabel(ylabel, fontsize=11)
    ax.yaxis.set_major_formatter(
        plt.matplotlib.ticker.FuncFormatter(lambda v, _: f'{int(v):+,}'))

fig.tight_layout()
fig.savefig(THIS / 'fig_three_group_p50_box.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('  fig_three_group_p50_box.png')


# ───────────────────────────────────────────────
# 8. 圖 4: 風險指標 ranking
# ───────────────────────────────────────────────
df_risk = df.sort_values('風險指標', ascending=False).copy()
fig, ax = plt.subplots(figsize=(11, 9))
sns.barplot(data=df_risk, y='short_name', x='風險指標',
            hue='三分群', order=df_risk['short_name'],
            palette=palette, dodge=False, ax=ax)
for _, row in df_risk.iterrows():
    ax.text(row['風險指標'] + 0.005,
            list(df_risk['short_name']).index(row['short_name']),
            f"{row['風險指標']:.3f}",
            va='center', ha='left', fontsize=9)
ax.set_xlabel('風險指標 = (P75 − P25) / P50  ←越小薪資越集中  越大薪資越分散→', fontsize=11)
ax.set_ylabel('學門', fontsize=11)
ax.set_title('學門薪資離散度排名（風險指標越大表示薪資範圍越廣）', fontsize=13)
cur_min, cur_max = ax.get_xlim()
span = cur_max - cur_min
ax.set_xlim(cur_min, cur_max + span * 0.08)
ax.legend(loc='lower right', fontsize=10)
fig.tight_layout()
fig.savefig(THIS / 'fig_risk_ranking.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('  fig_risk_ranking.png')


# 印 top 5 / bottom 5 風險排名
print('\n=== 風險指標 Top 5 (薪資最分散) ===')
print(df_risk[['short_name', '三分群', 'P25', 'P50', 'P75', '風險指標']].head(5).to_string(index=False))
print('\n=== 風險指標 Bottom 5 (薪資最集中) ===')
print(df_risk[['short_name', '三分群', 'P25', 'P50', 'P75', '風險指標']].tail(5).to_string(index=False))
