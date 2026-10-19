"""Narrative rules: answer the question, cite evidence once, flag only triggered issues."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from report_facts import standing, validate_context
from report_narrative import Narrator, fmt, interval, stat, tidy


def summary(**overrides):
    s = {"task": "regression", "target": "score", "split": "random", "external_test": False,
         "development_n": 96, "test_n": 24, "feature_n": 4, "cv_folds": 3, "metric": "mae",
         "direction": "lower", "selected_model": "ModelLRRegressor", "seed": 42,
         "question": "", "classes": [], "baseline": {"strategy": "median"}, "importance_repeats": 3}
    s.update(overrides)
    return s


def facts(primary=None, comparison=None, **extra):
    comparison = comparison or [
        {"model": "ModelLRRegressor", "family": "linear", "mean": .17, "sd": .03, "folds": 3,
         "standing": "selected", "fold_scores": [.13, .19, .18], "min": .13, "max": .19,
         "baseline_gain_positive_folds": 3, "baseline_gain_mean": 1.5, "train_gap": .01},
        {"model": "ModelRandomForestRegressor", "family": "bagging", "mean": .38, "sd": .11, "folds": 3,
         "standing": "weaker", "margin": .21, "margin_sd": 2.5, "relative_margin": .55},
    ]
    f = {"context": {}, "comparison": comparison, "selected": comparison[0],
         "primary": primary or {"metric": "mae", "estimate": .163, "lower": .108, "upper": .209,
                                "baseline": 1.71, "gain": 1.55, "gain_lower": .98, "gain_upper": 2.09,
                                "relative": .905},
         "tests": {}, "regression": {}, "classification": {}, "importance": {}, "shift": {},
         "strata": {}, "target": {}, "interval": {}, "features": [],
         "strength": {"strengths": [], "weaknesses": []}, "family_reading": "linear_best"}
    f["tests"]["mae"] = f["primary"]
    f.update(extra)
    return f


class NarrativeTests(unittest.TestCase):
    def test_conclusion_answers_question_by_evidence(self):
        n = Narrator(summary(), facts(), True)
        self.assertEqual(n.verdict(), "supported")
        self.assertIn("预测能力", n.conclusion())
        self.assertIn("MAE = 0.16, 95% CI [0.11, 0.21]", n.conclusion())
        crossing = dict(facts()["primary"], gain_lower=-.2)
        self.assertEqual(Narrator(summary(), facts(primary=crossing), True).verdict(), "uncertain")
        self.assertIn("包含 0", Narrator(summary(), facts(primary=crossing), True).conclusion())
        worse = dict(facts()["primary"], gain=-.1, gain_lower=-.3, gain_upper=.05)
        text = Narrator(summary(), facts(primary=worse), False).conclusion()
        self.assertIn("did not outperform the baseline", text)
        no_ci = dict(facts()["primary"], gain_lower=None, gain_upper=None)
        self.assertEqual(Narrator(summary(), facts(primary=no_ci), True).verdict(), "supported_point")

    def test_no_mechanical_claim_labels_or_significance(self):
        n = Narrator(summary(question="信号能否预测分数？"), facts(), True)
        text = " ".join(t for _, t in n.abstract()) + " ".join(n.discussion()) + " ".join(n.key_points())
        for label in ("主张：", "证据：", "阐释：", "显著"):
            self.assertNotIn(label, text)
        self.assertIn("信号能否预测分数？", text)

    def test_model_comparison_reports_cv_mean_and_sd(self):
        text = Narrator(summary(), facts(), True).model_comparison_sentence()
        self.assertIn("3 折交叉验证", text)
        self.assertIn("M = 0.17, SD = 0.03", text)
        self.assertIn("M = 0.38, SD = 0.11", text)
        self.assertNotIn("折间", text)

    def test_baseline_is_named_and_improvement_reported_apa_style(self):
        n = Narrator(summary(), facts(), False)
        self.assertIn("MAE = 0.16, 95% CI [0.11, 0.21]; baseline MAE = 1.71", n.baseline_comparison())
        self.assertIn("90% lower (difference = 1.55, 95% CI [0.98, 2.09])", n.improvement_clause())
        self.assertIn("training-set median", n.baseline_full())
        zh = Narrator(summary(), facts(), True)
        self.assertIn("DummyRegressor", zh.baseline_full())
        for term in ("开发集", "配对", "正向重要性"):
            self.assertNotIn(term, zh.conclusion() + zh.baseline_comparison() + zh.improvement_clause())

    def test_data_flags_only_when_triggered(self):
        quiet = Narrator(summary(), facts(shift={"missing": [], "mean_shift": [], "out_of_range": []}), True)
        self.assertEqual(quiet.data_flags(), [])
        noisy = Narrator(summary(), facts(shift={"missing": [{"feature": "noise", "development": .05, "test": .21}],
                                                 "mean_shift": [], "out_of_range": []}), True)
        self.assertIn("noise", noisy.data_flags()[0])
        self.assertIn("21%", noisy.data_flags()[0])

    def test_context_labels_units_and_tolerance(self):
        f = facts(context={"target_label": "分数", "target_unit": "分", "tolerance": .25, "feature_labels": {"signal": "信号"}},
                  regression={"median_ae": .13, "p90_ae": .36, "max_ae": .5, "within_tolerance": .75, "tolerance": .25},
                  importance={"ranked": [{"feature": "signal", "mean": 2.0, "sd": .1, "share": .8, "stable": True}],
                              "leaders": ["signal"], "leader_share": .8, "negligible": []})
        n = Narrator(summary(), f, True)
        self.assertIn("信号（signal）", n.leaders_text())
        self.assertIn("ΔMAE = 2.00, SD = 0.10", n.predictor_sentence())
        self.assertNotIn("%", n.predictor_sentence())
        self.assertIn("分数", n.conclusion())
        self.assertIn("75%", n.secondary_test_clause())
        self.assertNotIn("可接受误差", n.limitations_paragraph())

    def test_classification_names_positive_class(self):
        s = summary(task="classification", metric="balanced_accuracy", direction="higher",
                    classes=["case", "control"], selected_model="ModelLRClassifier")
        f = facts(classification={"ece": .2, "calibration_direction": "under"},
                  context={"class_labels": {"control": "阴性"}})
        self.assertIn("阴性（control）", Narrator(s, f, True).calibration_sentence())

    def test_overfitting_and_close_candidates_drive_next_steps(self):
        f = facts(strength={"strengths": [], "weaknesses": [("close_runner_up", ["ModelRandomForestRegressor"])]})
        f["selected"]["train_gap"] = .2
        steps = " ".join(Narrator(summary(), f, True).next_steps())
        self.assertIn("过拟合", steps)
        self.assertIn("不要依据本次测试结果重新排序", steps)


class FormattingTests(unittest.TestCase):
    def test_numbers_and_intervals(self):
        self.assertEqual(fmt(1.967e-15), "0")
        self.assertEqual(fmt(1234.5), "1,234")
        self.assertEqual(fmt(.123456), "0.12")
        self.assertEqual(fmt(.0031), "0.003")
        self.assertEqual(fmt(-1.4e-05), "0.00")
        self.assertEqual(fmt(.9961), "0.996")
        self.assertEqual(interval(-.117, .15), "[-0.12, 0.15]")
        self.assertIsNone(interval(None, 1))

    def test_apa_statistic(self):
        self.assertEqual(stat("MAE", .163, .108, .209), "MAE = 0.16, 95% CI [0.11, 0.21]")
        self.assertEqual(stat("R²", .989), "R² = 0.99")

    def test_cjk_latin_spacing(self):
        self.assertEqual(tidy("线性回归 优于K 近邻，测试集MAE为0.16。 下一句", True),
                         "线性回归 优于 K 近邻，测试集 MAE 为 0.16。下一句")


class FactTests(unittest.TestCase):
    def test_standing_uses_between_fold_sd(self):
        sel = {"model": "a", "mean": .2, "sd": .05}
        rows = [sel, {}]
        s = {"direction": "lower"}
        self.assertEqual(standing({"model": "b", "mean": .24, "sd": .05}, sel, rows, s), "near")
        self.assertEqual(standing({"model": "b", "mean": .3, "sd": .05}, sel, rows, s), "weaker")
        self.assertEqual(standing({"model": "b", "mean": .5, "sd": .05}, sel, rows, s), "clearly_weaker")

    def test_context_validation(self):
        self.assertEqual(validate_context({"tolerance": 2})["tolerance"], 2)
        for bad in ({"unknown": 1}, {"tolerance": -1}, {"feature_labels": ["x"]}, []):
            with self.assertRaises(ValueError):
                validate_context(bad)


if __name__ == "__main__":
    unittest.main()


class FigureTextTests(unittest.TestCase):
    def test_figures_use_clear_english_labels(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from pipeline_report import localize_figure
        fig, ax = plt.subplots()
        ax.set_xlabel("mae (lower is better)")
        ax.set_ylabel("Residual (observed - predicted)")
        ax.set_title("a  Where predictions fail")
        ax.set_yticks([0], ["LRRegressor"])
        localize_figure(fig)
        self.assertEqual(ax.get_xlabel(), "MAE (lower is better)")
        self.assertEqual(ax.get_ylabel(), "Residual (observed − predicted)")
        self.assertEqual(ax.get_title(), "a  Confusion matrix (row-normalised)")
        fig.canvas.draw()
        self.assertEqual(ax.get_yticklabels()[0].get_text(), "Linear regression")
        plt.close(fig)
