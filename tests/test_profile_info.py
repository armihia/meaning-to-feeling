import copy
import unittest

import numpy as np

from meaning_to_feeling.core import CompositeResponseModel
from meaning_to_feeling.profile_info import history_coverage
from meaning_to_feeling.prompts import DEFAULT_MODEL, EMOTIONS


class ReplayStore:
    semantic_interface = 'nine-emotion-activation-composition-v1'

    def request(self, _):
        return dict(message=dict(content='{"major":[1],"minor":[]}')), 0

    def one(self, _):
        return dict(emotions={}, values=[.2]+[0]*8+[.8]), {}

    def state_readout(self, answer):
        return answer['values']


class ProfileInfoTest(unittest.TestCase):
    def engine(self):
        head = dict(name='query', type='trace_affine', semantic_model=DEFAULT_MODEL,
            semantic_interface=ReplayStore.semantic_interface, emotion_order=EMOTIONS,
            prior=[0]*9+[1], semantic_mean=[0]*9+[1], gain=1)
        history = dict(head, name='history', gain=.5)
        return CompositeResponseModel(ReplayStore(), head, history)

    def history(self):
        return dict(emotion_order=EMOTIONS, history=[dict(text='我，回家。', context_text='下班。',
            target_character='我', candidate_ends=[2, 3, 5],
            emotion_observations=[dict(end_char=3, values=[.4]+[0]*8+[.6]), dict(end_char=5, values=[.6]+[0]*8+[.4])],
            boundary_observations=[dict(end_char=2, marked=True), dict(end_char=3, marked=False)])])

    def test_coverage_uses_history_head_and_real_observations(self):
        value = self.history()
        before = copy.deepcopy(value)
        profile = self.engine().generate_profile(value)
        self.assertEqual(value, before)
        coverage = profile['history_coverage']
        self.assertEqual(coverage['emotion_points'], 2)
        self.assertEqual(coverage['nonpunctuation_points'], 1)
        self.assertEqual(coverage['punctuation_points'], 1)
        joy = coverage['emotions'][0]
        self.assertAlmostEqual(joy['reference_min'], .1)
        self.assertAlmostEqual(joy['reference_max'], .1)
        self.assertEqual((joy['observed_min'], joy['observed_max']), (.4, .6))
        self.assertAlmostEqual(profile['vector'][0], .1)
        self.assertAlmostEqual(profile['vector'][9], -.1)

    def test_explanation_metadata_does_not_change_F(self):
        engine = self.engine()
        profile = engine.generate_profile(self.history())
        before = copy.deepcopy(profile)
        original = {k: v for k, v in profile.items() if k != 'history_coverage'}
        query = dict(text='我终于回来了。', context_text='外出很久。', target_character='我', candidate_ends=[2, 4, 7])
        full, stripped = engine.predict(query, profile), engine.predict(query, original)
        self.assertEqual(profile, before)
        for key in ['units', 'emotion_matrix', 'delta_matrix', 'personality_vector']:
            self.assertEqual(full[key], stripped[key])

    def test_empty_history_is_known_empty_not_fake_observations(self):
        info = history_coverage([], [0, 0])
        self.assertEqual(info['emotion_points'], 0)
        self.assertEqual(info['emotions'], [])
        profile = self.engine().generate_profile(dict(emotion_order=EMOTIONS, history=[]))
        np.testing.assert_array_equal(profile['vector'], np.zeros(12))


if __name__ == '__main__':
    unittest.main()
