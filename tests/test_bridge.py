import copy
import csv
from pathlib import Path
import tempfile
import unittest
from workflow.bridge import aggregate_samples, join_sensitivity, read_ppa, rescore_selection, write_csv, REGISTER_COLUMNS


class BridgeTests(unittest.TestCase):
    def samples(self):
        return [{'source_path': '/raw', 'graph_id': 1, 'node_id': 1, 'partition': 'train', 'label': 1},
                {'source_path': '/raw', 'graph_id': 2, 'node_id': 1, 'partition': 'test', 'label': 3},
                {'source_path': '/raw', 'graph_id': 1, 'node_id': 2, 'partition': 'val', 'label': 0},
                {'source_path': '/raw', 'graph_id': 2, 'node_id': 2, 'partition': 'test', 'label': 0}]

    def predictions(self):
        return [{'source_path': '/raw', 'graph_id': 2, 'node_id': 1, 'predicted_label': 0},
                {'source_path': '/raw', 'graph_id': 2, 'node_id': 2, 'predicted_label': 2}]

    def test_hybrid_labels_and_reference_are_separate(self):
        sensitivity, measured, outcomes = aggregate_samples(self.samples(), self.predictions(), [1, 2, 3, 4])
        self.assertEqual([r['sensitivity'] for r in sensitivity], [0.5, 0.5])
        self.assertEqual([r['measured_sensitivity'] for r in measured], [1.0, 0.0])
        self.assertEqual(sum(r['label_source'] == 'model' for r in outcomes), 2)

    def test_missing_predictions_never_use_truth(self):
        with self.assertRaisesRegex(ValueError, 'Missing prediction'):
            aggregate_samples(self.samples(), self.predictions()[:1], [1, 2, 3, 4])

    def test_partition_overlap_extra_and_duplicate_keys_rejected(self):
        cases = [self.predictions() + [self.predictions()[0]],
                 self.predictions() + [{'source_path': '/raw', 'graph_id': 1, 'node_id': 1, 'predicted_label': 0}],
                 self.predictions() + [{'source_path': '/other', 'graph_id': 1, 'node_id': 1, 'predicted_label': 0}]]
        for predictions in cases:
            with self.assertRaises(ValueError):
                aggregate_samples(self.samples(), predictions, [1, 2, 3, 4])
        rows = self.samples()
        rows[0]['partition'] = 'unknown'
        with self.assertRaises(ValueError):
            aggregate_samples(rows, self.predictions(), [1])

    def make_ppa(self, path):
        with path.open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['total_area', 'total_power', 'worst_slack(ns)', 'design_sensitivity', 'frequency(MHz)'])
            writer.writerow([100, 100, 1, 0.7, 200])
            writer.writerow([])
            writer.writerow(REGISTER_COLUMNS)
            writer.writerow([9, 'reg_B', 'DFF', 2, 3, 2, 1, 1, 0.7])
            writer.writerow([42, 'reg_A', 'DFF', 1, 2, 1, 1, 1, 0.7])

    def test_join_uses_names_not_index_offsets_or_row_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_ppa(root / 'ppa.csv')
            write_csv(root / 'map.csv', [{'node_id': 1, 'instance_name': 'reg_A'}, {'node_id': 2, 'instance_name': 'reg_B'}])
            write_csv(root / 'sens.csv', [{'node_id': 1, 'sensitivity': 0.2}, {'node_id': 2, 'sensitivity': 0.8}])
            join_sensitivity(root / 'ppa.csv', root / 'map.csv', root / 'sens.csv', root / 'merged.csv')
            _, design, regs = read_ppa(root / 'merged.csv')
            self.assertEqual(float(design[3]), 0.5)
            self.assertEqual([float(r['sensitivity']) for r in regs], [0.8, 0.2])
            self.assertEqual([r['index'] for r in regs], [9, 42])
            write_csv(root / 'map.csv', [{'node_id': 1, 'instance_name': 'reg_A'}])
            with self.assertRaises(ValueError):
                join_sensitivity(root / 'ppa.csv', root / 'map.csv', root / 'sens.csv', root / 'bad.csv')

    def test_malformed_ppa_has_no_random_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'invalid.csv'
            path.write_text('index,area,sensitivity\n1,2,0.5\n')
            with self.assertRaises(ValueError):
                read_ppa(path)

    def test_feedback_keeps_original_and_backend_denominators(self):
        mapping = [{'node_id': 1, 'instance_name': 'A'}, {'node_id': 2, 'instance_name': 'B'}]
        sensitivity = [{'node_id': 1, 'sensitivity': 0.8}, {'node_id': 2, 'sensitivity': 0.2}]
        result = rescore_selection(sensitivity, mapping, ['A'], 'sensitivity')
        self.assertAlmostEqual(result['before'], 0.5)
        self.assertAlmostEqual(result['ideal_tmr_after_original_denominator'], 0.1)
        self.assertAlmostEqual(result['backend_after_expanded_denominator'], 0.05)
        with self.assertRaises(ValueError):
            rescore_selection(sensitivity, mapping, ['unknown'], 'sensitivity')


if __name__ == '__main__':
    unittest.main()
