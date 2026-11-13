"""
Phase 1.1 + 1.2 + 1.3：
從 yoursalary Cognos 抓下來的 HTML 重新抽出「學門 × 13 個薪資 bucket 人數」，
- 用 bucket 內線性內插算 P25 / P50 / P75
- 計算「風險指標」= (P75 - P25) / P50

輸出：data/salary_with_quantiles.csv
"""
import re
from pathlib import Path

import pandas as pd

DATA = Path(__file__).parent / 'data'
HTML = DATA / 'scrape_frame_2.html'
OUT = DATA / 'salary_with_quantiles.csv'

# 13 個薪資 bucket 的下界 / 上界 (元/月)
BUCKETS = [
    (18_000, 25_000),     # 「未滿25,000元」假設下界 = 18,000 (約基本工資)
    (25_000, 30_000),
    (30_000, 35_000),
    (35_000, 40_000),
    (40_000, 45_000),
    (45_000, 50_000),
    (50_000, 55_000),
    (55_000, 60_000),
    (60_000, 70_000),
    (70_000, 80_000),
    (80_000, 90_000),
    (90_000, 100_000),
    (100_000, 150_000),   # 「100,000 以上」假設上界 = 150,000
]


def parse_int(s: str) -> int:
    """'1,229' → 1229 ; '-' → 0"""
    s = (s or '').strip().replace(',', '')
    if s in ('', '-', 'D'):
        return 0
    try:
        return int(s)
    except ValueError:
        return 0


# ───────────────────────────────────────────────
# 1. 從 HTML 抽 學門 + 13 buckets
# ───────────────────────────────────────────────
def extract_table(html_path: Path) -> pd.DataFrame:
    from bs4 import BeautifulSoup

    html = html_path.read_text(encoding='utf-8')
    soup = BeautifulSoup(html, 'html.parser')

    # 找包含整個 Cognos 表格的最大 tr (cells > 500)
    big_tr = None
    for tr in soup.find_all('tr'):
        if '未滿25,000元' in tr.get_text(' ', strip=True):
            cells = tr.find_all(['td', 'th'])
            if len(cells) > 500:
                big_tr = tr
                break
    if big_tr is None:
        raise SystemExit('找不到主資料表')

    cells = [c.get_text(strip=True) for c in big_tr.find_all(['td', 'th'])]

    # 學門 列表（順序需與 bucket 一致）
    # 從 cell 124 「合計」+ 25 學門 依序排列；但安全起見，
    # 我們從 cell list 中找「合計」位置，往後跑 27 次（每次跳 4 cells）。
    # 「合計」之 cell idx 透過 mean salary 47,446 識別
    xuemen_starts = []   # cell index of 學門 name
    for i in range(len(cells) - 3):
        a, b, c, d = cells[i:i + 4]
        # 學門 row 是 4 個 cell：name / total_n / contrib_n / mean_wage
        if (a == '合計' or a.endswith('學門')) and \
           re.fullmatch(r'[\d,]+', b) and \
           re.fullmatch(r'[\d,]+', c) and \
           re.fullmatch(r'[\d,]+', d):
            xuemen_starts.append(i)
    # 去除重疊 / 子串：只取第一輪
    valid_starts = []
    last = -10
    for idx in xuemen_starts:
        if idx > last + 3:
            valid_starts.append(idx)
            last = idx

    print(f'偵測到 {len(valid_starts)} 個學門 row（含合計）')

    xuemen_rows = []
    for idx in valid_starts:
        xuemen_rows.append({
            '學門': cells[idx],
            'employed_n': parse_int(cells[idx + 1]),
            'contributors_n': parse_int(cells[idx + 2]),
            'mean_salary': parse_int(cells[idx + 3]),
        })

    # bucket header 後 13×28 個數字（合計 + 27 學門）
    # 找 bucket header「未滿25,000元」位置
    bucket_header_idx = None
    for i, c in enumerate(cells):
        if c == '未滿25,000元':
            # 確認後面 12 cells 是 bucket labels
            if cells[i + 12] == '100,000元及以上':
                bucket_header_idx = i
                break
    if bucket_header_idx is None:
        raise SystemExit('找不到 bucket header')
    print(f'bucket header 起點 cell idx = {bucket_header_idx}')

    bucket_start = bucket_header_idx + 13
    # 連續 28 組（合計 + 27 學門）× 13 buckets
    bucket_data = []
    for g in range(len(valid_starts)):
        row = []
        for b in range(13):
            v = parse_int(cells[bucket_start + g * 13 + b])
            row.append(v)
        bucket_data.append(row)

    # 合併 4 欄基本資訊 + 13 個 bucket counts
    rows = []
    for meta, buckets in zip(xuemen_rows, bucket_data):
        out = dict(meta)
        for i, cnt in enumerate(buckets):
            out[f'b{i:02d}'] = cnt
        rows.append(out)
    return pd.DataFrame(rows)


# ───────────────────────────────────────────────
# 2. Quantile interpolation
# ───────────────────────────────────────────────
def quantile_from_buckets(counts: list[int], q: float) -> float | None:
    """Linear interpolation within bucket. q in [0,1]"""
    total = sum(counts)
    if total == 0:
        return None
    target = q * total
    cum = 0
    for i, n in enumerate(counts):
        new_cum = cum + n
        if new_cum >= target and n > 0:
            lo, hi = BUCKETS[i]
            frac = (target - cum) / n
            return lo + frac * (hi - lo)
        cum = new_cum
    # fallback (shouldn't reach)
    return BUCKETS[-1][1]


# ───────────────────────────────────────────────
# 3. 主流程
# ───────────────────────────────────────────────
def main():
    df = extract_table(HTML)
    print('\n=== 學門 × 4 欄 + 13 buckets ===')
    print(df[['學門', 'employed_n', 'contributors_n', 'mean_salary']].to_string(index=False))
    print('\n=== bucket counts sanity check (合計 row) ===')
    row = df.iloc[0]
    bucket_sum = sum(row[f'b{i:02d}'] for i in range(13))
    print(f'合計 employed_n = {row["employed_n"]}, bucket_sum = {bucket_sum}')
    if abs(bucket_sum - row['employed_n']) > row['employed_n'] * 0.02:
        print('⚠️ bucket counts 加總跟 employed_n 差太多')

    # 計算 P25 / P50 / P75 / 風險指標
    for q_name, q in [('P25', 0.25), ('P50', 0.50), ('P75', 0.75)]:
        df[q_name] = df.apply(
            lambda r: quantile_from_buckets(
                [r[f'b{i:02d}'] for i in range(13)], q),
            axis=1,
        )
    df['風險指標'] = (df['P75'] - df['P25']) / df['P50']
    df['風險指標'] = df['風險指標'].round(3)
    for c in ('P25', 'P50', 'P75'):
        df[c] = df[c].round(0).astype('Int64')

    print('\n=== 分位數結果 (按 P50 排序) ===')
    cols = ['學門', 'employed_n', 'mean_salary', 'P25', 'P50', 'P75', '風險指標']
    print(df[cols].sort_values('P50', ascending=False).to_string(index=False))

    df.to_csv(OUT, index=False, encoding='utf-8-sig')
    print(f'\n輸出: {OUT}')


if __name__ == '__main__':
    main()
