import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from report_explanations import (FIGURES, figure_guide, metric_explanation, preprocessing,
                                 abstract, validation_methods, result_summary, discussion, uncertainty_description)


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

    def test_primary_result_respects_loss_direction_and_interval_crossing_zero(self):
        summary = {'metric': 'mae', 'direction': 'lower'}
        row = {'metric': 'mae', 'estimate': .3, 'baseline': .5, 'gain': .2,
               'lower': .1, 'upper': .6, 'gain_lower': -.1, 'gain_upper': .4}
        text = result_summary(summary, [row], True)
        self.assertIn('低于基线', text)
        self.assertIn('区间包含零', text)
        self.assertNotIn('显著', text)
        row.update(estimate=.7, gain=-.2, gain_lower=-.5, gain_upper=-.1)
        self.assertIn('高于基线', result_summary(summary, [row], True))
        self.assertIn('区间位于零以下', result_summary(summary, [row], True))
        row.update(estimate=.5, gain=0, lower=None, upper=None, gain_lower=None, gain_upper=None)
        self.assertIn('等于基线', result_summary(summary, [row], True))
        self.assertNotIn('CI', result_summary(summary, [row], True))

    def test_primary_metric_alias_and_undefined_value(self):
        summary = {'metric': 'precision', 'direction': 'higher'}
        row = {'metric': 'precision_macro', 'estimate': .8, 'baseline': .3, 'gain': .5,
               'gain_lower': .1, 'gain_upper': .7}
        self.assertIn('高于基线', result_summary(summary, [row], True))
        self.assertIn('above zero', result_summary(summary, [row], False))
        row['estimate'] = None
        for zh in (True, False):
            text = result_summary(summary, [row], zh)
            self.assertIn('未能计算' if zh else 'undefined', text)
            self.assertNotIn('0.000', text)

    def test_methods_and_scope_follow_actual_design(self):
        summary = {'task': 'classification', 'split': 'group', 'external_test': True,
                   'seed': 42, 'cv_folds': 3, 'gap': 2, 'test_n': 24}
        for zh in (True, False):
            text = ' '.join(validation_methods(summary, zh))
            self.assertIn('外部' if zh else 'external', text)
            self.assertIn('GroupKFold', text)
            self.assertNotIn('时间块' if zh else 'time blocks', text)
            notes = ' '.join(discussion(summary, {'interval': {'units': 6}}, zh))
            self.assertNotIn('内部留出' if zh else 'internal holdout', notes)
            self.assertIn('新的独立测试' if zh else 'new independent test', notes)
            self.assertIn('因果' if zh else 'causal', notes)
        summary['split'] = 'time'
        self.assertIn('2 个时间块', ' '.join(validation_methods(summary, True)))

    def test_abstract_does_not_invent_primary_score(self):
        s = {'target': 'outcome', 'feature_n': 4, 'development_n': 96, 'test_n': 24,
             'cv_folds': 3, 'selected_model': 'ModelLRClassifier', 'metric': 'precision',
             'split': 'group', 'external_test': True, 'test_metrics': {'precision_macro': None}}
        for zh in (True, False):
            text = abstract(s, zh)
            self.assertIn('未能计算' if zh else 'undefined', text)
            self.assertIn('外部' if zh else 'external', text)

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

    def test_undefined_constant_baseline_is_explained(self):
        s = {'metric': 'pearson_r', 'direction': 'higher'}
        rows = [{'metric': 'pearson_r', 'estimate': .8, 'baseline': None, 'gain': None}]
        for zh in (True, False):
            self.assertIn('常量' if zh else 'constant', result_summary(s, rows, zh))
