"""以固定來源、手算小例、損毀反例核對驗證器，不以放寬門檻取得通過。"""
import json
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np
import pandas as pd

from data_pipeline import (BUCKET_COLS, DATA, DataValidationError,
                           check_snapshot, load_and_check, rebuild)


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.uac = pd.read_csv(DATA / 'uac_with_xuemen.csv', dtype={'系組代碼': str})
        cls.salary = pd.read_csv(DATA / 'salary_with_quantiles.csv')
        cls.snapshot = pd.read_csv(DATA / 'final_dataset_v2.csv')

    def test_shipped_data_reproduces_every_field_without_mutation(self):
        before_uac, before_sal = self.uac.copy(deep=True), self.salary.copy(deep=True)
        result, report = rebuild(self.uac, self.salary)
        check_snapshot(self.snapshot, result)
        self.assertEqual((report['校系總列數'], report['有效校系'], report['分析學門']), (1787, 1703, 25))
        self.assertEqual((report['未對應學門'], report['缺錄取分數'], report['兩者皆缺'], report['排除校系聯集']), (71, 15, 2, 84))
        self.assertEqual(report['薪資有而錄取無的學門'], ['其他學門', '安全服務學門'])
        self.assertEqual({r['系組代碼'] for r in report['來源品質待複核']},
                         {'0146', '1335', '2045', '2120', '2542', '2651', '2665'})
        self.assertEqual(result['dept_count'].sum(), 1703)
        self.assertEqual(result.iloc[0]['學門'], '物理、化學及地球科學學門')
        self.assertEqual(result.iloc[0]['residual_P50'], 17676)
        pd.testing.assert_frame_equal(self.uac, before_uac)
        pd.testing.assert_frame_equal(self.salary, before_sal)

    def test_small_hand_calculated_grain_and_missingness(self):
        # 兩學門：PR [10,30] 的中位為20（非加權20人）；另一學門PR80。
        uac = pd.DataFrame({
            '系組代碼': ['0001', '0002', '0003', '0004', '0005'],
            '校名': ['測試大學'] * 5, '系組名': list('甲乙丙丁戊'),
            '學門': ['甲學門', '甲學門', '乙學門', None, '乙學門'],
            '錄取分數': [10., 30., 80., 40., np.nan], '加權滿分': [100.] * 5,
            '達成率': [10., 30., 80., 40., np.nan],
            '錄取分數百分位': [10., 30., 80., 40., np.nan],
        })
        salary = pd.DataFrame({'學門': ['甲學門', '乙學門'],
                               'employed_n': [50, 100], 'contributors_n': [40, 80],
                               'mean_salary': [27500, 42500],
                               'P25': [26250, 41250], 'P50': [27500, 42500],
                               'P75': [28750, 43750], '風險指標': [.091, .059]})
        for col in BUCKET_COLS:
            salary[col] = 0
        salary.loc[0, 'b01'], salary.loc[1, 'b04'] = 40, 80
        result, report = rebuild(uac, salary)
        keyed = result.set_index('學門')
        self.assertEqual(keyed.loc['甲學門', 'median_pr_pct'], 20)
        self.assertEqual(keyed.loc['甲學門', 'dept_count'], 2)
        self.assertEqual(keyed.loc['乙學門', 'dept_count'], 1)
        self.assertEqual(result['residual_P50'].tolist(), [0, 0])
        self.assertEqual(report['排除校系聯集'], 2)
        self.assertEqual(report['來源品質待複核'], [])
        self.assertEqual(uac.iloc[0]['系組代碼'], '0001')

    def test_duplicate_or_missing_join_keys_are_rejected(self):
        for target, key in [('uac', '系組代碼'), ('salary', '學門')]:
            for bad in ['duplicate', None, ' ']:
                with self.subTest(target=target, key=key, bad=bad):
                    uac, sal = self.uac.copy(), self.salary.copy()
                    frame = uac if target == 'uac' else sal
                    frame.loc[1, key] = frame.loc[0, key] if bad == 'duplicate' else bad
                    with self.assertRaisesRegex(DataValidationError, key):
                        rebuild(uac, sal)

    def test_unknown_field_is_not_silently_lost_by_inner_join(self):
        uac = self.uac.copy()
        uac.loc[0, '學門'] = '未知學門'
        with self.assertRaisesRegex(DataValidationError, '無薪資.*未知學門'):
            rebuild(uac, self.salary)

    def test_unknown_field_with_missing_scores_is_rejected_before_filtering(self):
        uac = self.uac.copy(deep=True)
        missing_row = uac.index[uac['系組代碼'].eq('1223')][0]
        self.assertTrue(uac.loc[missing_row, ['錄取分數', '達成率', '錄取分數百分位']].isna().all())
        uac.loc[missing_row, '學門'] = '未知學門'
        before = uac.copy(deep=True)
        with self.assertRaisesRegex(DataValidationError, '無薪資.*未知學門'):
            rebuild(uac, self.salary)
        pd.testing.assert_frame_equal(uac, before)

    def test_invalid_salary_values_are_rejected(self):
        cases = [('b00', -1, '非負整數'), ('b00', .5, '非負整數'),
                 ('b00', np.nan, '不得缺值'), ('b00', np.inf, '無限值'),
                 ('contributors_n', 0, '大於零'), ('P50', np.nan, '不得缺值'),
                 ('P25', 100000, '分位數'), ('P50', 41722, '重算'),
                 ('風險指標', .9, '重算')]
        for col, value, message in cases:
            with self.subTest(col=col, value=value):
                sal = self.salary.copy()
                sal[col] = sal[col].astype(float)
                sal.loc[0, col] = value
                with self.assertRaisesRegex(DataValidationError, message):
                    rebuild(self.uac, sal)
        sal = self.salary.copy()
        sal.loc[0, 'b00'] += 1
        with self.assertRaisesRegex(DataValidationError, 'contributors_n'):
            rebuild(self.uac, sal)

    def test_invalid_admission_values_and_partial_missingness_are_rejected(self):
        for col, value, message in [('錄取分數百分位', 101, '0 至 100'),
                                     ('錄取分數百分位', np.nan, '缺值必須一致'),
                                     ('錄取分數百分位', np.inf, '無限值'),
                                     ('加權滿分', 0, '大於零'),
                                     ('達成率', 1, '不符')]:
            with self.subTest(col=col, value=value):
                uac = self.uac.copy()
                uac[col] = uac[col].astype(float)
                uac.loc[0, col] = value
                with self.assertRaisesRegex(DataValidationError, message):
                    rebuild(uac, self.salary)

    def test_snapshot_tampering_is_rejected(self):
        expected, _ = rebuild(self.uac, self.salary)
        for col in ['dept_count', 'median_pr_pct', 'P50', 'residual_P50']:
            with self.subTest(col=col):
                bad = self.snapshot.copy()
                bad.loc[0, col] += 1
                with self.assertRaisesRegex(DataValidationError, col):
                    check_snapshot(bad, expected)
        with self.assertRaisesRegex(DataValidationError, '排序'):
            check_snapshot(self.snapshot.iloc[::-1], expected)
        with self.assertRaisesRegex(DataValidationError, '集合'):
            check_snapshot(self.snapshot.iloc[1:], expected)

    def test_cli_success_and_strict_quality_failure(self):
        root = Path(__file__).resolve().parents[1]
        for flags, code in [([], 0), (['--strict-quality'], 1)]:
            run = subprocess.run([sys.executable, '-X', 'utf8', str(root / 'data_pipeline.py'), *flags],
                                 cwd=root, capture_output=True, encoding='utf-8', check=False)
            self.assertEqual(run.returncode, code, run.stdout + run.stderr)
            report = json.loads(run.stdout)
            self.assertTrue(report['一致性通過'])
            self.assertEqual(len(report['來源品質待複核']), 7)
            if flags:
                self.assertFalse(report['嚴格品質通過'])


if __name__ == '__main__':
    unittest.main()
