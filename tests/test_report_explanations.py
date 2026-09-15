import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from report_explanations import FIGURES, figure_guide, metric_explanation, preprocessing


class ReportExplanationTests(unittest.TestCase):
    def test_values_and_undefined_metrics_keep_their_meaning(self):
        self.assertIn('0.3125', metric_explanation('mae', .3125, True)[1])
        for zh in (True, False):
            text = metric_explanation('roc_auc', None, zh)[1]
            self.assertNotIn('0.000', text)
            self.assertIn('未能计算' if zh else 'Undefined', text)
        self.assertIn('不是预测正确率', metric_explanation('r2', -.2, True)[1])

    def test_preprocessing_describes_only_enabled_operations(self):
        basic = {'preprocessing': {'impute': 'none', 'scale': False, 'select_k': 0, 'pca': 0, 'resample': 'none'}}
        text = ' '.join(preprocessing(basic, True))
        self.assertNotIn('PCA', text)
        self.assertNotIn('中位数填补', text)
        self.assertIn('不做特征筛选', text)
        detailed = {'preprocessing': {'impute': 'median', 'scale': True, 'scaler': 'minmax',
                                     'selection': 'forward', 'select_k': 3, 'pca': 2, 'resample': 'smote'}}
        text = ' '.join(preprocessing(detailed, True))
        for value in ('中位数', '0–1', '前向', '2 个', 'smote'):
            self.assertIn(value, text)

    def test_all_figure_families_have_reading_and_limits_in_both_languages(self):
        summary = {'test_metrics': {}, 'selected_model': 'ModelLRRegressor', 'search': {}}
        for key in FIGURES:
            for zh in (True, False):
                with self.subTest(key=key, zh=zh):
                    name = key + '-2' if key in ('parameter-search', 'strata-performance') else key
                    title, paragraphs = figure_guide({'name': name, 'caption': 'technical caption'}, summary,
                                                    {'findings': []}, zh)
                    self.assertNotEqual(title, name)
                    self.assertGreater(len(paragraphs), 1)
                    self.assertTrue(all(text for _, text in paragraphs))

    def test_figure_finding_uses_current_evidence_not_a_fixed_score(self):
        finding = {'code': 'errors', 'zh': '实际误差=0.777', 'en': 'Measured error=0.777'}
        _, rows = figure_guide({'name': 'regression-diagnostics', 'caption': ''}, {},
                               {'findings': [finding]}, True)
        self.assertIn(('本次结果', finding['zh']), rows)
