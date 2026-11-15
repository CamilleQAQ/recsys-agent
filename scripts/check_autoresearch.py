"""Targeted behavior tests: real subprocesses, scripted LLM only in unit tests."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from recsys.platform.autoresearch import run_research, evaluate, write_json
from recsys.platform.budget import BudgetPolicy, TaskBudget
from recsys.platform.shared_experiments import SharedExperimentStore, reuse_reasons


class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir='experiments')
        self.root = Path(self.temp.name).resolve()
        self.case = {'business': 'recommendation', 'dataset': 'unit_fixture', 'split': 'validation',
            'objective': 'NDCG@1', 'ks': [1, 2], 'context': {}, 'groups': [
                {'id': 'u', 'candidates': [{'item': 1, 'cold': False}, {'item': 2, 'cold': True}], 'positives': [2]}]}
        self.path = self.root / 'case.json'
        write_json(self.path, self.case)

    def tearDown(self):
        self.temp.cleanup()

    def client(self, code, analysis='上一轮反馈支持修改排序'):
        return {'choices': [{'message': {'role': 'assistant', 'content': '', 'tool_calls': [
            {'id': 'proposal', 'type': 'function', 'function': {'name': 'submit_algorithm',
             'arguments': json.dumps({'hypothesis': '改变顺序', 'feedback_analysis': analysis, 'code': code})}}]}}]}

    def test_feedback_code_versions_and_memory(self):
        calls = []
        def client(config, messages, **kwargs):
            calls.append(json.loads(json.dumps(messages)))
            code = ('def rank(groups, context):\n    return {str(g["id"]): [r["item"] for r in reversed(g["candidates"])] for g in groups}\n'
                    if len(calls) == 1 else
                    'def rank(groups, context):\n    return {str(g["id"]): [r["item"] for r in g["candidates"]] for g in groups}\n')
            return self.client(code)
        store = SharedExperimentStore(self.root / 'store')
        result = run_research('fixture only', business='recommendation', case_path=self.path,
                              work_dir=self.root, store=store, rounds=2, client=client)
        self.assertTrue(result['ok'])
        self.assertEqual([a['status'] for a in result['attempts']], ['keep', 'keep', 'discard'])
        self.assertEqual(result['best']['iteration'], 1)
        feedback = json.loads(calls[1][-1]['content'])
        self.assertEqual(feedback['last_experiment']['value'], 1)
        self.assertEqual(feedback['last_experiment']['feedback']['segments']['cold']['metrics']['Recall@1'], 1)
        self.assertEqual(result['budget']['counters']['experiment'], 3)
        for record in store.latest():
            self.assertFalse(reuse_reasons(record, {'dataset': 'unit_fixture', 'split': 'validation'}))
        worker_input = json.loads((Path(result['attempts'][1]['algorithm']).parent/'input.json').read_text(encoding='utf8'))
        self.assertNotIn('positives', worker_input['groups'][0])
        second = run_research('reuse', business='recommendation', case_path=self.path,
                             work_dir=self.root, store=store, rounds=1, client=client)
        self.assertEqual(len(second['history_records_read']), 3)

    def test_child_uses_parent_budget_without_extra_llm(self):
        parent = TaskBudget('budget-test', BudgetPolicy(max_experiments=1, max_tool_calls=4, max_history_lookups=1))
        with patch('recsys.platform.autoresearch.post_chat') as llm:
            result = run_research('budget', business='recommendation', case_path=self.path,
                work_dir=self.root, scope=parent.child('rec'), rounds=2, client=llm)
            llm.assert_not_called()
        self.assertFalse(result['ok'])
        self.assertEqual(parent.counters['experiment'], 1)
        self.assertEqual(result['stop_reason'], 'budget_exhausted')

    def test_crash_is_feedback_not_success(self):
        result = run_research('crash', business='recommendation', case_path=self.path,
            work_dir=self.root, rounds=1, client=lambda *a, **kw:self.client('raise RuntimeError("experiment failure")\n'))
        self.assertFalse(result['ok'])
        self.assertEqual(result['attempts'][-1]['status'], 'crash')
        self.assertIn('experiment failure', result['attempts'][-1]['log'])

    def test_unknown_or_duplicate_items_rejected(self):
        for rows in [{'u': [1, 1]}, {'u': [1, 3]}]:
            with self.assertRaises(ValueError):evaluate(self.case, rows)

    def test_segment_denominators_include_unrecalled_targets(self):
        self.case['groups'][0]['positives'] = [2, 3]
        self.case['cold_items'] = [2, 3]
        result = evaluate(self.case, {'u': [2, 1]})
        self.assertEqual(result['metrics']['Recall@1'], .5)
        self.assertEqual(result['segments']['cold']['positive_count'], 2)
        self.assertEqual(result['segments']['cold']['metrics']['Recall@1'], .5)


if __name__ == '__main__':
    unittest.main()
