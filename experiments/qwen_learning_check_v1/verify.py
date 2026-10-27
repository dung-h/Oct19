"""CPU integration checks; synthetic fixtures are never GPU/quality evidence."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('learning_check_runner',HERE/'run.py')
runner=importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ControlContract(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.cfg=json.loads((HERE/'config.json').read_text())
        self.cfg['bootstrap_repeats']=8
        self.members=dict(pool=list(range(70)),train=list(range(50)),validation=list(range(50,70)),train_probes=list(range(15)))
        self.targets={i:i%5+1 for i in range(70)}
        self.docs={i:dict(id=str(i),group=str(i//2)) for i in range(70)}

    def tearDown(self):
        self.temp.cleanup()

    def fixture(self,dest):
        dest.mkdir()
        labels=np.array([self.targets[i] for i in range(70)])
        rng=np.random.default_rng(42)
        h=np.eye(5)[labels-1]*5+rng.normal(0,.05,(70,5))
        np.savez(dest/'features.npz',indices=np.arange(70),hidden=h,logits=h)
        rows=[dict(index=i,text=f'topic{self.targets[i]} topic{self.targets[i]} abstract body') for i in range(70)]
        (dest/'feature_texts.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        (dest/'EXTRACT_COMPLETE.json').write_text(json.dumps(dict(status='FEATURES_COMPLETE')))

    def test_changing_validation_labels_cannot_change_fitted_controls(self):
        a,b=self.root/'a',self.root/'b'
        self.fixture(a);self.fixture(b)
        runner.controls(a,self.cfg,self.targets,self.members)
        mutated={i:(self.targets[i]%5+1 if i in self.members['validation'] else self.targets[i]) for i in range(70)}
        runner.controls(b,self.cfg,mutated,self.members)
        pa=np.load(a/'control_predictions.npz');pb=np.load(b/'control_predictions.npz')
        self.assertEqual(pa.files,pb.files)
        for name in pa.files:
            np.testing.assert_array_equal(pa[name],pb[name])
        pred=pa['hidden_head_validation'].argmax(1)+1
        self.assertGreater(np.mean(pred==[self.targets[i] for i in self.members['validation']]),.95)
        self.assertFalse(json.loads((a/'CONTROLS_COMPLETE.json').read_text())['validation_labels_used'])

    def test_evaluation_requires_all_epochs_and_preserves_prediction_order(self):
        root=self.root/'eval';self.fixture(root)
        runner.controls(root,self.cfg,self.targets,self.members)
        training=root/'training';training.mkdir()
        (training/'COMPLETE.json').write_text(json.dumps(dict(status='TRAINING_COMPLETE')))
        def probabilities(indices,sharp):
            p=np.full((len(indices),5),.01 if sharp else .2)
            if sharp:
                for j,i in enumerate(indices):p[j,self.targets[i]-1]=.96
            return p
        for epoch in (1,2):
            np.savez(training/f'epoch{epoch}_predictions.npz',validation=probabilities(self.members['validation'],epoch==2),train_probes=probabilities(self.members['train_probes'],epoch==2))
        with self.assertRaises(FileNotFoundError):runner.evaluate(root,self.cfg,self.docs,self.targets,self.members)
        self.assertFalse((root/'metrics.csv').exists())
        np.savez(training/'epoch3_predictions.npz',validation=probabilities(self.members['validation'],True),train_probes=probabilities(self.members['train_probes'],True))
        import contextlib,io
        with contextlib.redirect_stdout(io.StringIO()):runner.evaluate(root,self.cfg,self.docs,self.targets,self.members)
        decision=json.loads((root/'decision.json').read_text())
        self.assertTrue(decision['extra_epochs_promising'])
        self.assertGreater(decision['validation_AUC_epoch3_minus_epoch1'],.4)
        with self.assertRaises(RuntimeError):runner.evaluate(root,self.cfg,self.docs,self.targets,self.members)


if __name__=='__main__':unittest.main()
