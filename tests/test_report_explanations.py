import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from report_explanations import (FIGURES, figure_guide, metric_explanation, preprocessing,
                                 validation_methods, uncertainty_description, model_name,
                                 PER_MODEL_SPLIT_THRESHOLD)


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
        self.assertIn(('结果', finding['zh']), rows)

    def test_methods_and_scope_follow_actual_design(self):
        summary = {'task': 'classification', 'split': 'group', 'external_test': True,
                   'seed': 42, 'cv_folds': 3, 'gap': 2, 'test_n': 24}
        for zh in (True, False):
            text = ' '.join(validation_methods(summary, zh))
            self.assertIn('外部' if zh else 'external', text)
            self.assertIn('GroupKFold', text)
            self.assertNotIn('时间块' if zh else 'time blocks', text)
        summary['split'] = 'time'
        self.assertIn('2 个时间块', ' '.join(validation_methods(summary, True)))

    def test_f_threshold_includes_boundary_in_description(self):
        s = {'preprocessing': {'selection': 'f-threshold', 'f_threshold': 5}}
        self.assertIn('不低于 5', ' '.join(preprocessing(s, True)))

    def test_omitted_intervals_do_not_describe_resampling_as_performed(self):
        interval = {'omitted_reason': 'Time dependence requires a justified block length; row bootstrap is omitted.'}
        for zh in (True, False):
            text = ' '.join(uncertainty_description(interval, zh))
            self.assertIn('未计算置信区间' if zh else 'without confidence intervals', text)
            self.assertNotIn('95%', text)
            self.assertIn('时间' if zh else 'Time', text)

    def test_model_names_and_split_threshold(self):
        self.assertIn('随机森林', model_name('ModelRandomForestRegressor', True))
        self.assertGreaterEqual(PER_MODEL_SPLIT_THRESHOLD, 4)

    def test_interval_description_is_concise_and_scoped(self):
        interval = {'method': 'row percentile bootstrap', 'requested_repeats': 200, 'units': 60, 'omitted_reason': None}
        for zh in (True, False):
            text = ' '.join(uncertainty_description(interval, zh))
            self.assertIn('95%', text)
            self.assertIn('200', text)
            self.assertIn('模型选择' if zh else 'selection', text)


if __name__ == '__main__':
    unittest.main()
