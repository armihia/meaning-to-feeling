const test = require('node:test');
const assert = require('node:assert/strict');
const {profileDetails} = require('../meaning_to_feeling/web/app.js');
const emotions = ['joy','sadness','anger','fear','surprise','anxiety','frustration','confusion','guilt','neutral'].map((key) => ({key, label: key}));
const zero = () => new Array(12).fill(0);

test('原始向量的历史范围未知，不补成零次观测', () => {
  const value = profileDetails(zero(), emotions);
  assert.equal(value.historyTrials, null);
  assert.equal(value.emotionPoints, null);
  assert.equal(value.traits, null);
  assert.deepEqual(value.ranges, []);
});

test('拒绝错维度、非数值与不合法偏移', () => {
  for (const vector of [new Array(128).fill(0), [true, ...zero().slice(1)], [NaN, ...zero().slice(1)], new Array(12).fill(.1)]) assert.equal(profileDetails(vector, emotions), null);
});

test('量表零值是最低计分，缺失掩码保持未知', () => {
  const value = profileDetails({vector: zero(), personality_measurement: {instrument_id: 'CBF-PI-15', trait_order: ['O','C','E','A','N'], normalised_trait_values_0_1: [0, 0, .5, 1, null], observed_mask: [true, false, true, true, false]}}, emotions);
  assert.deepEqual(value.traits.map((t) => t.mean), [1, null, 3.5, 6, null]);
});

test('新增解释保持保存向量与来源元数据不变', () => {
  const vector = zero(); vector[0] = .15; vector[9] = -.15; vector[10] = .2;
  const profile = {vector, history_trials: 3, emotion_history_trials: 2, history_coverage: {emotion_points: 4, emotions: [{key: 'joy', label: '错误名称', reference_min: .1, reference_max: .2, observed_min: 0, observed_max: .8}]}};
  const before = JSON.stringify(profile);
  const value = profileDetails(profile, emotions);
  assert.equal(value.parameters[0].value, .15);
  assert.equal(value.nonpunctuation, .2);
  assert.equal(value.ranges.length, 1);
  assert.equal(value.ranges[0].label, 'joy');
  value.vector[0] = 0;
  assert.equal(JSON.stringify(profile), before);
});

test('损坏的可选覆盖不影响有效参数显示', () => {
  const value = profileDetails({vector: zero(), history_coverage: {emotions: [null, {}, {key: 'joy', reference_min: .8, reference_max: .1, observed_min: 0, observed_max: 1}]}}, emotions);
  assert.deepEqual(value.ranges, []);
  assert.equal(value.parameters.length, 10);
});

test('明确的空历史保留已知零与空范围', () => {
  const value = profileDetails({vector: zero(), history_trials: 0, emotion_history_trials: 0, history_coverage: {emotion_points: 0, emotions: []}}, emotions);
  assert.equal(value.historyTrials, 0);
  assert.equal(value.emotionPoints, 0);
});

test('时点筛选显示选入和两种排除计数', () => {
  const history_time_scope = {as_of: '2026-11-19T01:00:00+00:00', provided_trials: 4, eligible_trials: 2, after_cutoff_trials: 1, unverified_completion_trials: 1};
  assert.deepEqual(profileDetails({vector: zero(), history_time_scope}, emotions).historyTimeScope, history_time_scope);
  history_time_scope.eligible_trials = 3;
  assert.equal(profileDetails({vector: zero(), history_time_scope}, emotions).historyTimeScope, null);
});

test('情绪坐标顺序必须与当前模型相符', () => {
  assert.equal(profileDetails({vector: zero(), emotion_order: emotions.map((e) => e.key).reverse()}, emotions), null);
});

const dynamicProfile = () => ({
  profile_kind: 'initial_reference_body_gain23',
  emotion_order: emotions.map((e) => e.key),
  parameter_order: [...emotions.map((e) => 'emotion_bias_' + e.key), ...emotions.map((e) => 'initial_emotion_bias_' + e.key), 'nonpunctuation_rate_shift', 'punctuation_rate_shift', 'body_response_gain'],
  vector: [.1, ...new Array(8).fill(0), -.1, .7, ...new Array(8).fill(0), -.7, -.2, .3, 1.3],
});

test('23维将初始偏移、切分和变化强度分开读取', () => {
  const profile = dynamicProfile();
  const before = JSON.stringify(profile);
  const value = profileDetails(profile, emotions);
  assert.equal(value.parameters[0].value, .1);
  assert.equal(value.initialParameters[0].value, .7);
  assert.equal(value.nonpunctuation, -.2);
  assert.equal(value.punctuation, .3);
  assert.equal(value.bodyResponseGain, 1.3);
  assert.equal(JSON.stringify(profile), before);
  const old = profileDetails(zero(), emotions);
  assert.equal(old.nonpunctuation, 0);
  assert.equal(old.initialParameters, null);
  assert.equal(old.bodyResponseGain, null);
});

test('23维必须带正确格式与两个坐标顺序', () => {
  const profile = dynamicProfile();
  assert.equal(profileDetails(profile.vector, emotions), null);
  assert.equal(profileDetails({vector: profile.vector}, emotions), null);
  assert.equal(profileDetails({...profile, profile_kind: 'historical_gain23'}, emotions), null);
  assert.equal(profileDetails({...profile, emotion_order: profile.emotion_order.toReversed()}, emotions), null);
  assert.equal(profileDetails({...profile, parameter_order: profile.parameter_order.toReversed()}, emotions), null);
});

test('正文变化强度允许0至2，初始偏移也须总和为零', () => {
  for (const gain of [0, 1, 2]) {
    const profile = dynamicProfile(); profile.vector[22] = gain;
    assert.equal(profileDetails(profile, emotions).bodyResponseGain, gain);
  }
  for (const gain of [-.01, 2.01, Infinity, '1']) {
    const profile = dynamicProfile(); profile.vector[22] = gain;
    assert.equal(profileDetails(profile, emotions), null);
  }
  const profile = dynamicProfile(); profile.vector[10] += .1;
  assert.equal(profileDetails(profile, emotions), null);
});

test('历史均值分开解释开始前与正文，缺失阶段保持未知', () => {
  const keys = emotions.map((e) => e.key);
  const profile = {profile_kind: 'phase_history_mean22', emotion_order: keys,
    parameter_order: [...keys.map((e) => 'initial_history_sum_' + e), ...keys.map((e) => 'body_history_sum_' + e), 'nonpunctuation_rate_shift', 'punctuation_rate_shift'],
    phase_history_trials: [0, 2], vector: [...new Array(10).fill(0), .8, ...new Array(8).fill(0), 1.2, .1, 0]};
  const before = JSON.stringify(profile);
  const info = profileDetails(profile, emotions);
  assert.equal(info.historyMean.initial[0], null);
  assert.equal(info.historyMean.body[0], .4);
  assert.deepEqual(info.historyMean.weights, [0, .4]);
  assert.equal(info.nonpunctuation, .1);
  assert.equal(JSON.stringify(profile), before);
  assert.equal(profileDetails(profile.vector, emotions), null);
  assert.equal(profileDetails({...profile, phase_history_trials: [0, 3]}, emotions), null);
});
