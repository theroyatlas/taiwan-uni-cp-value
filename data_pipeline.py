"""離線核對隨附資料並重建學門主表；不抓資料、不覆寫來源。"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from extract_quantiles import quantile_from_buckets

DATA = Path(__file__).resolve().parent / 'data'
BUCKET_COLS = [f'b{i:02d}' for i in range(13)]
SALARY_COLS = ['employed_n', 'contributors_n', 'mean_salary',
               'P25', 'P50', 'P75', '風險指標']
FINAL_COLS = ['學門', 'short_name', '三分群', 'median_pr_pct',
              'mean_salary', 'P25', 'P50', 'P75', '風險指標',
              'P50_pred', 'residual_P50', 'dept_count',
              'employed_n', 'contributors_n']
APPLIED_STEM = {'資訊通訊科技學門', '工程及工程業學門', '製造及加工學門',
                '建築及營建工程學門', '醫藥衛生學門', '衛生及職業衛生服務學門',
                '運輸服務學門'}
BASIC_STEM = {'物理、化學及地球科學學門', '數學及統計學門', '生命科學學門',
              '環境學門', '農業學門', '林業學門', '漁業學門', '獸醫學門'}


class DataValidationError(ValueError):
    """資料不符合明確契約，不允許靜默丟列或補零。"""


def require(condition, message):
    if not condition:
        raise DataValidationError(message)


def require_columns(frame, columns, label):
    missing = sorted(set(columns) - set(frame.columns))
    require(not missing, f'{label} 缺少欄位：{missing}')
    require(not frame.empty, f'{label} 不得為空表')


def require_key(frame, key, label):
    values = frame[key]
    require(values.notna().all(), f'{label}.{key} 不得缺值')
    require(values.map(lambda v: isinstance(v, str) and bool(v.strip())
                       and v == v.strip()).all(), f'{label}.{key} 必須為非空且無首尾空白的文字')
    require(not values.duplicated().any(), f'{label}.{key} 不得重複（join 粒度錯誤）')


def numbers(frame, columns, label, nullable=False):
    for col in columns:
        values = frame[col]
        require(pd.api.types.is_numeric_dtype(values), f'{label}.{col} 必須為數值')
        if not nullable:
            require(values.notna().all(), f'{label}.{col} 不得缺值')
        require(np.isfinite(values.dropna().astype(float)).all(), f'{label}.{col} 不得為無限值')


def classify(field):
    if field in APPLIED_STEM:
        return '應用 STEM'
    if field in BASIC_STEM:
        return '基礎 STEM'
    return '人文社科'


def validate_sources(uac, salary):
    """保留合法缺值，明列排除數；薪資桶以實際提繳人數核對。"""
    require_columns(uac, ['系組代碼', '校名', '系組名', '學門',
                          '錄取分數', '加權滿分', '達成率', '錄取分數百分位'], '校系')
    require_key(uac, '系組代碼', '校系')
    for col in ['校名', '系組名']:
        require(uac[col].notna().all() and uac[col].str.strip().ne('').all(), f'校系.{col} 不得缺值')
    fields = uac['學門'].dropna()
    require(fields.map(lambda v: isinstance(v, str) and bool(v.strip())
                       and v == v.strip()).all(), '校系.學門 不得為空白文字')
    numbers(uac, ['加權滿分'], '校系')
    require(uac['加權滿分'].gt(0).all(), '校系.加權滿分 必須大於零')
    nullable = ['錄取分數', '達成率', '錄取分數百分位']
    numbers(uac, nullable, '校系', nullable=True)
    missing = uac['錄取分數'].isna()
    require(all(uac[c].isna().equals(missing) for c in nullable), '校系錄取分數、達成率、百分位缺值必須一致')
    valid = uac.loc[~missing]
    require(valid['錄取分數'].ge(0).all(), '校系.錄取分數 不得為負數')
    require(valid['達成率'].ge(0).all(), '校系.達成率 不得為負數')
    require(valid['錄取分數百分位'].between(0, 100).all(), '校系.錄取分數百分位 必須介於 0 至 100')
    expected_rate = (valid['錄取分數'] / valid['加權滿分'] * 100).round(2)
    require(np.allclose(valid['達成率'], expected_rate, rtol=0, atol=1e-8), '校系.達成率 與錄取分數/加權滿分不符')

    require_columns(salary, ['學門'] + SALARY_COLS + BUCKET_COLS, '薪資')
    require_key(salary, '學門', '薪資')
    # 缺分數不代表非空學門可以略過完整性檢查；先驗證所有來源 key。
    unmatched = sorted(set(fields) - set(salary.loc[salary['學門'] != '合計', '學門']))
    require(not unmatched, f'校系有而無薪資的學門：{unmatched}')
    numbers(salary, SALARY_COLS + BUCKET_COLS, '薪資')
    counts = salary[['employed_n', 'contributors_n'] + BUCKET_COLS]
    require((counts.ge(0) & counts.mod(1).eq(0)).all().all(), '薪資人數必須為非負整數')
    require(salary['contributors_n'].gt(0).all(), '薪資 contributors_n 必須大於零，否則分位數無定義')
    require(salary['employed_n'].ge(salary['contributors_n']).all(), '薪資提繳人數不得超過在職人數')
    require(salary['mean_salary'].gt(0).all(), '薪資 mean_salary 必須大於零')
    require(salary[BUCKET_COLS].sum(axis=1).eq(salary['contributors_n']).all(), '薪資桶合計必須等於 contributors_n（不是 employed_n）')
    require(((salary['P25'] > 0) & (salary['P25'] <= salary['P50'])
             & (salary['P50'] <= salary['P75'])).all(), '薪資分位數必須滿足 0 < P25 <= P50 <= P75')
    raw_quantiles = {}
    for col, q in [('P25', .25), ('P50', .5), ('P75', .75)]:
        raw_quantiles[col] = salary[BUCKET_COLS].apply(
            lambda row: quantile_from_buckets(row.tolist(), q), axis=1)
        require(np.array_equal(salary[col], raw_quantiles[col].round(0)), f'薪資.{col} 與薪資桶重算結果不符')
    risk = ((raw_quantiles['P75'] - raw_quantiles['P25']) / raw_quantiles['P50']).round(3)
    require(np.allclose(salary['風險指標'], risk, rtol=0, atol=1e-8), '薪資.風險指標 與未取整分位數重算結果不符')


def rebuild(uac, salary):
    """以有效校系的 PR 中位數，作學門一對一 join 與原有 OLS 殘差排序。"""
    validate_sources(uac, salary)
    usable = uac[uac['學門'].notna() & uac['錄取分數百分位'].notna()]
    adm = usable.groupby('學門').agg(
        median_pr_pct=('錄取分數百分位', 'median'),
        dept_count=('系組代碼', 'size')).reset_index()
    sal = salary[salary['學門'] != '合計']
    df = adm.merge(sal[['學門'] + SALARY_COLS], on='學門', how='left',
                   validate='one_to_one', indicator=True)
    unmatched = df.loc[df['_merge'] != 'both', '學門'].tolist()
    require(not unmatched, f'有錄取資料但無薪資的學門：{unmatched}')
    require(len(df) >= 2 and df['median_pr_pct'].nunique() >= 2, 'OLS 至少需要兩個不同 PR 的學門')
    df = df.drop(columns='_merge')
    df['三分群'] = df['學門'].map(classify)
    x = df[['median_pr_pct']].to_numpy()
    y = df['P50'].astype(float).to_numpy()
    model = LinearRegression().fit(x, y)
    df['P50_pred'] = model.predict(x).round(0).astype(int)
    df['residual_P50'] = (y - df['P50_pred']).round(0).astype(int)
    df['short_name'] = df['學門'].str.replace('學門', '')
    report = {
        '校系總列數': len(uac),
        '未對應學門': int(uac['學門'].isna().sum()),
        '缺錄取分數': int(uac['錄取分數百分位'].isna().sum()),
        '兩者皆缺': int((uac['學門'].isna() & uac['錄取分數百分位'].isna()).sum()),
        '有效校系': len(usable),
        '排除校系聯集': len(uac) - len(usable),
        '分析學門': len(df),
        '薪資有而錄取無的學門': sorted(set(sal['學門']) - set(adm['學門'])),
        '薪資桶母數': 'contributors_n',
        '薪資粒度': '學門層級 proxy，非校系或個人薪資預測',
        '來源品質待複核': [
            {'系組代碼': row['系組代碼'], '校名': row['校名'],
             '系組名': row['系組名'], '達成率': row['達成率'],
             '原因': '達成率超過100%；未更正原始分數、滿分或排名'}
            for _, row in uac.loc[uac['達成率'] > 100].iterrows()
        ],
        'R2': float(model.score(x, y)),
    }
    return df[FINAL_COLS].sort_values('residual_P50', ascending=False).reset_index(drop=True), report


def check_snapshot(actual, expected):
    require_columns(actual, FINAL_COLS, '主表')
    require(list(actual.columns) == FINAL_COLS, '主表欄位及順序與契約不符')
    require_key(actual, '學門', '主表')
    require(set(actual['學門']) == set(expected['學門']), '主表學門集合與重建不符')
    require(actual['residual_P50'].is_monotonic_decreasing, '主表必須依原有殘差由高至低排序')
    actual = actual.set_index('學門').sort_index()
    expected = expected.set_index('學門').sort_index()
    for col in ['short_name', '三分群']:
        require(actual[col].equals(expected[col]), f'主表.{col} 與重建不符')
    numeric = [c for c in actual.columns if c not in ['short_name', '三分群']]
    numbers(actual, numeric, '主表')
    for col in numeric:
        require(np.allclose(actual[col], expected[col], rtol=0, atol=1e-8), f'主表.{col} 與重建不符')


def load_and_check(data_dir=DATA):
    data_dir = Path(data_dir)
    uac = pd.read_csv(data_dir / 'uac_with_xuemen.csv', dtype={'系組代碼': str})
    salary = pd.read_csv(data_dir / 'salary_with_quantiles.csv')
    snapshot = pd.read_csv(data_dir / 'final_dataset_v2.csv')
    rebuilt, report = rebuild(uac, salary)
    check_snapshot(snapshot, rebuilt)
    return rebuilt, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DATA)
    parser.add_argument('--output', type=Path, help='選填：另存重建 CSV；檔案已存在則拒絕覆寫')
    parser.add_argument('--strict-quality', action='store_true',
                        help='來源品質待複核項目也視為失敗；隨附資料有7筆達成率超過100%')
    args = parser.parse_args()
    try:
        rebuilt, report = load_and_check(args.data_dir)
        if args.strict_quality and report['來源品質待複核']:
            print(json.dumps({'一致性通過': True, '嚴格品質通過': False,
                              **report}, ensure_ascii=False, indent=2))
            return 1
        if args.output:
            with args.output.open('x', encoding='utf-8-sig', newline='') as output:
                rebuilt.to_csv(output, index=False)
        print(json.dumps({'一致性通過': True, **report}, ensure_ascii=False, indent=2))
        return 0
    except (DataValidationError, OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        print(json.dumps({'通過': False, '錯誤': str(exc)}, ensure_ascii=False))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
