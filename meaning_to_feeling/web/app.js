const $ = (id) => document.getElementById(id);
const state = { data: null, mode: "preset", view: "trajectory", result: null, run: 0, point: 0, axes: new Set(), busy: false, job: null, revision: 0, resultRevision: -1, toastTimer: null };
const node = (tag, className, text) => {
  const value = document.createElement(tag);
  if (className) value.className = className;
  if (text !== undefined) value.textContent = text;
  return value;
};
const percent = (value, signed = false) => {
  const p = value * 100;
  if (Math.abs(p) < .05) return signed ? "0" : "0%";
  const digits = Math.abs(p) < 10 ? 1 : 0;
  return (signed && p > 0 ? "+" : "") + p.toFixed(digits) + (signed ? "" : "%");
};
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function profileDetails(profile, emotions) {
  const metadata = profile && !Array.isArray(profile) ? profile : {};
  const vector = profile === null ? new Array(12).fill(0) : metadata.vector || profile;
  if (!Array.isArray(vector) || ![12, 22, 23].includes(vector.length) || vector.some((v) => typeof v !== "number" || !Number.isFinite(v))) return null;
  const dynamic = vector.length === 23;
  const meanProfile = vector.length === 22;
  const emotionOrder = emotions.map((e) => e.key);
  let historyMean = null;
  if (meanProfile) {
    const order = [...emotionOrder.map((e) => "initial_history_sum_" + e), ...emotionOrder.map((e) => "body_history_sum_" + e), "nonpunctuation_rate_shift", "punctuation_rate_shift"];
    const counts = metadata.phase_history_trials;
    if (metadata.profile_kind !== "phase_history_mean22" || JSON.stringify(metadata.parameter_order) !== JSON.stringify(order) || JSON.stringify(metadata.emotion_order) !== JSON.stringify(emotionOrder) || !Array.isArray(counts) || counts.length !== 2 || counts.some((n) => !Number.isSafeInteger(n) || n < 0) || vector.slice(20).some((v) => Math.abs(v) > 1.000001)) return null;
    const sums = [vector.slice(0, 10), vector.slice(10, 20)];
    if (sums.some((row, phase) => row.some((v) => v < 0 || v > counts[phase] + 1e-6) || Math.abs(row.reduce((a, b) => a + b, 0) - counts[phase]) > 1e-6)) return null;
    historyMean = {counts: [...counts], initial: sums[0].map((v) => counts[0] ? v / counts[0] : null), body: sums[1].map((v) => counts[1] ? v / counts[1] : null), weights: counts.map((n) => n / (n + 3))};
  }
  const dynamicOrder = [...emotionOrder.map((e) => "emotion_bias_" + e), ...emotionOrder.map((e) => "initial_emotion_bias_" + e), "nonpunctuation_rate_shift", "punctuation_rate_shift", "body_response_gain"];
  if (dynamic && (metadata.profile_kind !== "initial_reference_body_gain23" || JSON.stringify(metadata.parameter_order) !== JSON.stringify(dynamicOrder) || JSON.stringify(metadata.emotion_order) !== JSON.stringify(emotionOrder))) return null;
  if (!meanProfile && (vector.slice(0, dynamic ? 22 : 12).some((v) => Math.abs(v) > 1.000001) || Math.abs(vector.slice(0, 10).reduce((a, b) => a + b, 0)) > .00001 || dynamic && (Math.abs(vector.slice(10, 20).reduce((a, b) => a + b, 0)) > .00001 || vector[22] < 0 || vector[22] > 2))) return null;
  if (metadata.emotion_order && JSON.stringify(metadata.emotion_order) !== JSON.stringify(emotions.map((e) => e.key))) return null;
  const count = (v) => Number.isInteger(v) && v >= 0 ? v : null;
  const scope = metadata.history_time_scope;
  const scopeCounts = [scope?.provided_trials, scope?.eligible_trials, scope?.after_cutoff_trials, scope?.unverified_completion_trials].map(count);
  const historyTimeScope = typeof scope?.as_of === "string" && scopeCounts.every((n) => n !== null) &&
    scopeCounts[0] === scopeCounts[1] + scopeCounts[2] + scopeCounts[3] ? {...scope} : null;
  const coverage = metadata.history_coverage || {};
  const fraction = (v) => typeof v === "number" && Number.isFinite(v) && v >= 0 && v <= 1;
  const ranges = emotions.map((e) => {
    const row = Array.isArray(coverage.emotions) ? coverage.emotions.find((r) => r && r.key === e.key) : null;
    return row && [row.reference_min, row.reference_max, row.observed_min, row.observed_max].every(fraction) && row.reference_min <= row.reference_max && row.observed_min <= row.observed_max ? {...row, label: e.label} : null;
  });
  const measurement = metadata.personality_measurement;
  const order = ["O", "C", "E", "A", "N"];
  const labels = ["开放性", "尽责性", "外向性", "宜人性", "神经质"];
  let traits = null;
  if (measurement?.instrument_id === "CBF-PI-15" && JSON.stringify(measurement.trait_order) === JSON.stringify(order) && Array.isArray(measurement.normalised_trait_values_0_1) && measurement.normalised_trait_values_0_1.length === 5) {
    traits = order.map((key, i) => {
      const value = measurement.normalised_trait_values_0_1[i];
      const observed = fraction(value) && (!Array.isArray(measurement.observed_mask) || measurement.observed_mask[i] === true);
      return {key, label: labels[i], mean: observed ? 1 + 5*value : null};
    });
  }
  return {vector: [...vector], historyMean, parameters: emotions.map((e, i) => ({key: e.key, label: e.label, value: vector[i]})),
    initialParameters: dynamic ? emotions.map((e, i) => ({key: e.key, label: e.label, value: vector[10 + i]})) : null,
    bodyResponseGain: dynamic ? vector[22] : null,
    nonpunctuation: vector[dynamic || meanProfile ? 20 : 10], punctuation: vector[dynamic || meanProfile ? 21 : 11], historyTrials: count(metadata.history_trials),
    emotionTrials: count(metadata.emotion_history_trials), emotionPoints: count(coverage.emotion_points),
    nonpunctuationPoints: count(coverage.nonpunctuation_points), punctuationPoints: count(coverage.punctuation_points),
    ranges: ranges.filter(Boolean), traits, historyTimeScope};
}

function renderProfileDetails() {
  const target = $("profileDetailsBody");
  target.replaceChildren();
  if (!state.data || state.mode === "history") {
    target.append(node("p", "profile-note", "生成向量后，这里会显示各项参数和历史覆盖范围。"));
    return;
  }
  const preset = state.mode === "preset" ? state.data.profiles.find((p) => p.id === $("profileSelect").value) : null;
  let profile;
  try { profile = preset ? preset.profile : JSON.parse($("profileJSON").value); }
  catch (_) { target.append(node("p", "profile-note", "填好个人向量后，这里会显示每项参数的作用。")); return; }
  const info = profileDetails(profile, state.data.emotions);
  if (!info) { target.append(node("p", "profile-note", "请填写12维向量，或导入从历史生成的完整profile.json。")); return; }
  const signed = (v) => (v > 0 ? "+" : "") + (100*v).toFixed(2);
  target.append(node("p", "profile-provenance", preset?.source_label || "从输入文件读取"));
  const history = info.historyTrials === null ? "历史覆盖尚未提供" : `历史包含${info.historyTrials}段材料`;
  target.append(node("p", "profile-note", history + (info.emotionPoints === null ? "。" : `、${info.emotionPoints}个感受记录位置。`)));
  if (info.historyTimeScope) {
    const scope = info.historyTimeScope;
    target.append(node("p", "profile-note",
      `开读时间 ${scope.as_of}。选入 ${scope.eligible_trials} / ${scope.provided_trials} 段。跳过开读后提交的 ${scope.after_cutoff_trials} 段，以及缺少有效提交时间的 ${scope.unverified_completion_trials} 段。`));
  }
  if (info.historyMean) {
    const mean = info.historyMean;
    target.append(node("h3", "", "本人历史均值融合"), node("p", "profile-note", `开始前参考${mean.counts[0]}段历史，阅读中参考${mean.counts[1]}段。本人历史占比依次为${percent(mean.weights[0])}、${percent(mean.weights[1])}，其余来自模型对当前语境和原句的预测。`));
    const table = node("table", "history-mean-table");
    const header = node("tr"); ["感受", "开始前均值", "阅读中均值"].forEach((text) => header.append(node("th", "", text))); table.append(header);
    state.data.emotions.forEach((e, i) => { const row = node("tr"); [e.label, mean.initial[i] === null ? "暂无记录" : percent(mean.initial[i]), mean.body[i] === null ? "暂无记录" : percent(mean.body[i])].forEach((text) => row.append(node("td", "", text))); table.append(row); });
    target.append(table, node("p", "profile-note", "每段历史等权，标记多的段落不会额外加权。未记录的阶段使用模型预测。"));
  } else {
  target.append(node("h3", "", (info.initialParameters ? "正文感受校准" : "感受校准") + " · 单位：百分点"));
  target.append(node("p", "profile-note", "模型先读文本，再按这些数值校准感受比例。正值向上调整，负值向下；最后共同调整为总和100%。"));
  const grid = node("dl", "parameter-grid");
  info.parameters.forEach((p) => { const item = node("div", "parameter-item"); item.append(node("dt", "", p.label), node("dd", p.value > 0 ? "positive" : p.value < 0 ? "negative" : "", signed(p.value))); grid.append(item); });
  target.append(grid);
  if (info.initialParameters) {
    const details = node("details", "coverage-details");
    details.append(node("summary", "", "查看开始前的感受校准"), node("p", "profile-note", "这10项单独调整开始阅读前的感受，单位为百分点。"));
    const initialGrid = node("dl", "parameter-grid");
    info.initialParameters.forEach((p) => { const item = node("div", "parameter-item"); item.append(node("dt", "", p.label), node("dd", p.value > 0 ? "positive" : p.value < 0 ? "negative" : "", signed(p.value))); initialGrid.append(item); });
    details.append(initialGrid);
    target.append(details, node("h3", "", "正文变化强度"), node("p", "response-gain", info.bodyResponseGain.toFixed(3)), node("p", "profile-note", "0更平缓，1保留参考变化，2放大参考变化。它作用于正文；开始前的感受单独校准。"));
  }
  }
  target.append(node("h3", "", "切分判断"));
  const boundary = node("dl", "boundary-parameters");
  [["无标点细分修正", info.nonpunctuation], ["原版标点修正", info.punctuation]].forEach(([label, value]) => { const item = node("div"); item.append(node("dt", "", label), node("dd", "", signed(value) + " 个百分点")); boundary.append(item); });
  target.append(boundary, node("p", "profile-note", "无标点修正用于提示模型的细分倾向。当前句读标点固定保留，原版标点修正仅用于原版切分。"));
  target.append(node("h3", "", "大五量表记录 · 1–6分"));
  if (info.traits) {
    const traits = node("dl", "trait-parameters");
    info.traits.forEach((trait) => { const item = node("div"); item.append(node("dt", "", trait.label), node("dd", "", trait.mean === null ? "未测得" : trait.mean.toFixed(2))); traits.append(item); });
    target.append(traits, node("p", "profile-note", "CBF-PI-15计分均值。量表分数单独保存；当前预测使用上面的历史校准向量。"));
  } else target.append(node("p", "profile-note", "尚未提供量表记录。导入含有量表记录的profile.json后可在这里查看。"));
  if (info.ranges.length) {
    const details = node("details", "coverage-details");
    details.append(node("summary", "", "历史覆盖了哪些感受范围？"), node("p", "profile-note", "参考预测是模型在历史材料上的输入范围，本人作答是实际记录的范围。范围外的使用属于外推。"));
    const table = node("table");
    const header = node("tr"); ["感受", "参考预测", "本人作答"].forEach((text) => header.append(node("th", "", text))); table.append(header);
    info.ranges.forEach((row) => { const tr = node("tr"); [row.label, `${percent(row.reference_min)}–${percent(row.reference_max)}`, `${percent(row.observed_min)}–${percent(row.observed_max)}`].forEach((text) => tr.append(node("td", "", text))); table.append(tr); });
    details.append(table); target.append(details);
  }
}

async function api(path, value) {
  const response = await fetch(path, value === undefined ? { cache: "no-store" } : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(value),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "暂时无法完成请求，请重试。");
  return data;
}

function readJSON(value, label) {
  if (!value.trim()) throw new Error("请先填写" + label + "。");
  try { return JSON.parse(value); }
  catch (error) { throw new Error(label + "的JSON格式不正确：" + error.message); }
}

function toast(message) {
  $("toast").textContent = message;
  $("toast").hidden = false;
  clearTimeout(state.toastTimer);
  state.toastTimer = setTimeout(() => { $("toast").hidden = true; }, 3200);
}

function showError(error) {
  $("formError").textContent = error.message || String(error);
  $("formError").hidden = false;
  toast(error.message || String(error));
}

function markDraft() {
  state.revision += 1;
  if (state.result) {
    $("staleNotice").hidden = false;
    $("resultFlag").textContent = "上一次分析";
  }
}

function setMode(mode) {
  state.mode = mode;
  document.querySelectorAll("[data-mode]").forEach((button) => {
    const selected = button.dataset.mode === mode;
    button.classList.toggle("selected", selected);
    button.setAttribute("aria-selected", String(selected));
  });
  ["preset", "custom", "history"].forEach((name) => { $(name + "Panel").hidden = name !== mode; });
  renderProfileDetails();
  markDraft();
}

function chooseProfile() {
  const profile = state.data.profiles.find((p) => p.id === $("profileSelect").value);
  $("profileSource").textContent = profile.source_label;
  $("profileDescription").textContent = profile.description;
  const vector = profile.profile?.vector || profile.profile || new Array(12).fill(0);
  $("presetVector").textContent = JSON.stringify(vector, null, 2);
  renderProfileDetails();
  markDraft();
}

function chooseExample(id) {
  const value = state.data.examples.find((example) => example.id === id);
  if (!value) return;
  $("textInput").value = value.text;
  $("contextInput").value = value.context_text;
  $("characterInput").value = value.target_character;
  countText();
  markDraft();
}

function countText() { $("textCount").textContent = Array.from($("textInput").value).length + " 字"; }

function inputValue() {
  if (!$("textInput").value.trim()) throw new Error("请先写下要分析的句子。");
  return { text: $("textInput").value, context_text: $("contextInput").value, target_character: $("characterInput").value || "我" };
}

function profileValue() {
  if (state.mode === "preset") return { preset_id: $("profileSelect").value };
  if (state.mode === "history") throw new Error("先点击“从这些历史生成向量”，生成后再分析新句子。");
  return { value: readJSON($("profileJSON").value, "个人向量"), name: $("customName").value || "我的向量" };
}

function busy(value, kind = "predict") {
  const activity = {profile:"生成向量",load:"加载模型",unload:"释放显存",predict:"分析"}[kind] || "分析";
  state.busy = value;
  $("analyzeButton").disabled = value;
  $("generateProfile").disabled = value;
  $("profileGenerator").disabled = value;
  $("emptyAnalyze").disabled = value;
  $("reanalyzeButton").disabled = value;
  $("emptyAnalyze").firstChild.textContent = value ? "正在" + activity + "… " : "分析当前内容 ";
  $("emptyProgress").hidden = !value;
  $("progressPanel").hidden = !value;
  $("analyzeButton").firstElementChild.textContent = value ? "正在" + activity + "…" : "开始分析";
  $("loadModel").disabled = value;
  $("unloadModel").disabled = value;
}

async function refreshStatus() {
  try {
    const status = await api("/api/status");
    $("modelStatus").className = "status-pill" + (status.busy ? " busy" : status.loaded ? " loaded" : "");
    $("modelStatus").replaceChildren(node("span"), document.createTextNode(status.loading ? "正在加载" : status.busy ? "分析中" : status.loaded ? "模型已就绪" : "模型未加载"));
    $("loadModel").disabled = status.busy || status.loaded;
    $("unloadModel").disabled = status.busy || !status.loaded;
    return status;
  } catch (_) {
    $("modelStatus").className = "status-pill";
    $("modelStatus").replaceChildren(node("span"), document.createTextNode("服务未连接"));
    return null;
  }
}

async function runTask(payload) {
  if (state.busy) return;
  $("formError").hidden = true;
  const revision = state.revision;
  busy(true, payload.kind);
  $("progressMessage").textContent = "正在提交";
  $("progressElapsed").textContent = "已等待 0 秒";
  try {
    const job = await api("/api/jobs", payload);
    await followJob(job.id, job.kind, revision);
  } catch (error) {
    showError(error);
  } finally {
    busy(false);
    await refreshStatus();
  }
}

async function followJob(id, kind, revision = state.revision) {
  state.job = id;
  let networkErrors = 0;
  while (state.job === id) {
    let job;
    try { job = await api("/api/jobs/" + id); networkErrors = 0; }
    catch (error) {
      networkErrors += 1;
      if (networkErrors > 3) throw error;
      $("progressMessage").textContent = "正在重新连接本机服务";
      await sleep(1000);
      continue;
    }
    $("progressMessage").textContent = job.message;
    $("emptyProgress").textContent = job.message;
    $("progressElapsed").textContent = "已等待 " + Math.max(0, Math.round(Date.now() / 1000 - job.created_at)) + " 秒";
    if (job.state === "failed") throw new Error(job.error || "本次分析没有完成，请重试。");
    if (job.state === "complete") {
      if (kind === "predict") {
        state.result = job.result;
        state.run = 0;
        state.point = 0;
        state.resultRevision = revision;
        chooseAxes();
        renderResult();
        if (window.innerWidth <= 760) $("resultContent").scrollIntoView({ behavior: "smooth", block: "start" });
      } else if (kind === "profile") {
        const profile = job.result.profile;
        $("profileJSON").value = JSON.stringify(profile, null, 2);
        $("customName").value = "我的历史向量";
        setMode("custom");
        const scope = profile.history_time_scope;
        if (scope) {
          const selected = scope.eligible_trials
            ? `选入 ${scope.eligible_trials} / ${scope.provided_trials} 段历史。`
            : "该时点没有可用历史，已生成无历史向量。";
          toast(selected + "可分析新句子，也可保存向量。");
        } else {
          toast("向量已生成，现在可以分析新句子。也可以保存JSON文件。");
        }
      } else if (kind === "unload") toast("本轮模型显存已释放。");
      else toast("模型已加载，可以开始分析。");
      state.job = null;
      return;
    }
    await refreshStatus();
    await sleep(650);
  }
}

async function analyze(event) {
  if (event) event.preventDefault();
  if (state.busy || !state.data) return;
  try {
    const profiles = [profileValue()];
    if ($("compareToggle").checked) profiles.push($("compareSelect").value === "__custom__" ? {value: readJSON($("compareJSON").value, "第二个向量"), name: "自填对比向量"} : { preset_id: $("compareSelect").value });
    await runTask({ kind: "predict", input: inputValue(), profiles });
  } catch (error) { showError(error); }
}

function activeRun() { return state.result.runs[state.run]; }
function activePrediction() { return activeRun().prediction; }
function allStates() { const p = activePrediction(); return [p.initial_emotion, ...p.emotion_matrix]; }

function topFeelings(values, count = 2) {
  return values.map((value, index) => ({ value, index })).sort((a, b) => b.value - a.value).slice(0, count).filter((v) => v.value > 0).map((v) => state.data.emotions[v.index].label + " " + percent(v.value)).join(" · ");
}

function chooseAxes() {
  const values = allStates();
  const rank = state.data.emotions.map((_, index) => ({ index, variation: Math.max(...values.map((v) => v[index])) - Math.min(...values.map((v) => v[index])), peak: Math.max(...values.map((v) => v[index])) }));
  rank.sort((a, b) => b.variation - a.variation || b.peak - a.peak);
  state.axes = new Set(rank.slice(0, 4).map((v) => v.index));
}

function renderResult() {
  $("emptyState").hidden = true;
  $("resultContent").hidden = false;
  const current = activeRun();
  const prediction = current.prediction;
  $("resultTitle").textContent = current.name + " · " + prediction.units.length + " 个意元";
  const stale = state.resultRevision !== state.revision;
  $("staleNotice").hidden = !stale;
  $("resultFlag").textContent = stale ? "上一次分析" : "分析结果";
  $("runSource").textContent = prediction.personalization_mode === "phase_history_mean" ? "本人历史均值融合" : current.source_label + " · 原版校准";
  $("runTabs").replaceChildren();
  $("runTabs").hidden = state.result.runs.length < 2;
  state.result.runs.forEach((run, index) => {
    const button = node("button", index === state.run ? "selected" : "", String.fromCharCode(65 + index) + " · " + run.name);
    button.type = "button";
    button.addEventListener("click", () => { state.run = index; state.point = 0; chooseAxes(); renderResult(); });
    $("runTabs").append(button);
  });
  $("unitList").replaceChildren();
  prediction.units.forEach((unit, index) => {
    const button = node("button", "unit-token" + (state.point === index + 1 ? " active" : ""));
    button.type = "button";
    const fixed = prediction.segmentation?.fixed_punctuation_ends || [];
    const boundaryKind = index === prediction.units.length-1 ? "句末" : fixed.includes(unit.end_char) ? "标点" : "语义";
    button.append(node("span", "unit-index", "意元 " + (index + 1) + " · " + boundaryKind), document.createTextNode(unit.text));
    button.title = "查看读到这里时的感受";
    button.addEventListener("click", () => { state.point = index + 1; renderResult(); });
    $("unitList").append(button);
  });
  $("endFeeling").textContent = "句末：" + topFeelings(prediction.emotion_matrix.at(-1));
  $("timingInfo").textContent = "本轮 " + state.result.wall_seconds.toFixed(1) + " 秒" + (state.result.loaded_model_this_job ? " · 含模型加载" : "") + " · 前缀复用 " + (prediction.timing?.prefix_cache_hits || 0) + "/" + (prediction.timing?.prefix_queries || 0);
  $("resultContext").textContent = state.result.input.context_text || "本次未填写额外语境。";
  $("resultCharacter").textContent = "代入人物：" + state.result.input.target_character;
  $("rawJSON").textContent = JSON.stringify(prediction, null, 2);
  renderComparison();
  renderLegend();
  renderChart();
  renderBars();
  renderMatrix(false);
  renderMatrix(true);
  setView(state.view);
}

function renderComparison() {
  const box = $("comparisonSummary");
  box.replaceChildren();
  box.hidden = state.result.runs.length !== 2;
  if (box.hidden) return;
  const [a, b] = state.result.runs;
  const endsA = new Set(a.prediction.units.slice(0, -1).map((u) => u.end_char));
  const endsB = new Set(b.prediction.units.slice(0, -1).map((u) => u.end_char));
  const difference = [...endsA].filter((v) => !endsB.has(v)).length + [...endsB].filter((v) => !endsA.has(v)).length;
  box.append(node("p", "", "切分：" + a.name + " " + a.prediction.units.length + " 块，" + b.name + " " + b.prediction.units.length + " 块；" + (difference ? difference + " 个切分位置不同。" : "切分位置相同。")));
  box.append(node("p", "", a.name + "的句末感受：" + topFeelings(a.prediction.emotion_matrix.at(-1))));
  box.append(node("p", "", b.name + "的句末感受：" + topFeelings(b.prediction.emotion_matrix.at(-1))));
  const policy = a.prediction.segmentation?.policy || "legacy";
  const info = node("details", "segmentation-reason");
  info.append(node("summary", "", "这次哪些参数影响了切分？"));
  const shift = (run) => {
    const info = profileDetails(run.profile, state.data.emotions);
    return info ? info.nonpunctuation.toFixed(3) : "未提供";
  };
  if (policy === "detailed") {
    info.append(node("p", "", "细分倾向：" + a.name + " " + shift(a) + "，" + b.name + " " + shift(b) + "。该条件直接进入切点判断，不限制块数；句读标点固定保留。"));
    info.append(node("p", "", difference ? "以上不同位置来自各自的语义判断，未强制要求两人切得不同。" : "本句得到了相同的切点。情绪部分不同，不等于每句话的切分都必然不同。"));
  } else info.append(node("p", "", "这是原版数量校准：相同目标块数会取同一份语义排序。默认细分模式已取消这一步。"));
  box.append(info);
}

function renderLegend() {
  $("emotionLegend").replaceChildren();
  state.data.emotions.forEach((emotion, index) => {
    const button = node("button", "legend-button" + (state.axes.has(index) ? "" : " off"));
    button.type = "button";
    button.setAttribute("aria-pressed", String(state.axes.has(index)));
    const dot = node("i"); dot.style.background = emotion.color;
    button.append(dot, document.createTextNode(emotion.label));
    button.addEventListener("click", () => {
      if (state.axes.has(index)) state.axes.delete(index); else state.axes.add(index);
      renderLegend(); renderChart();
    });
    $("emotionLegend").append(button);
  });
}

function svgNode(tag, attributes, text) {
  const value = document.createElementNS("http://www.w3.org/2000/svg", tag);
  Object.entries(attributes).forEach(([key, attribute]) => value.setAttribute(key, String(attribute)));
  if (text !== undefined) value.textContent = text;
  return value;
}

function renderChart() {
  const matrix = allStates();
  const width = 720, height = 244, left = 38, right = 16, top = 19, bottom = 35;
  const peak = state.axes.size ? Math.max(...matrix.flatMap((row) => [...state.axes].map((i) => row[i]))) : 1;
  const upper = $("autoScale").checked ? Math.max(.1, Math.ceil(peak * 10) / 10) : 1;
  const x = (index) => left + index / Math.max(1, matrix.length - 1) * (width - left - right);
  const y = (value) => height - bottom - value / upper * (height - top - bottom);
  const svg = svgNode("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": "感受随意元序列变化的曲线" });
  for (let tick = 0; tick <= 4; tick += 1) {
    const value = upper * tick / 4;
    svg.append(svgNode("line", { x1: left, x2: width - right, y1: y(value), y2: y(value), class: "chart-grid" }));
    svg.append(svgNode("text", { x: left - 8, y: y(value) + 3, "text-anchor": "end", class: "chart-axis-text" }, percent(value)));
  }
  matrix.forEach((_, index) => {
    if (index === 0 || index === matrix.length - 1 || index % Math.ceil(matrix.length / 8) === 0) svg.append(svgNode("text", { x: x(index), y: height - 10, "text-anchor": index === 0 ? "start" : index === matrix.length - 1 ? "end" : "middle", class: "chart-axis-text" }, index === 0 ? "开始前" : "意元 " + index));
  });
  svg.append(svgNode("line", { x1: x(state.point), x2: x(state.point), y1: top, y2: height - bottom, stroke: "#bdcdb2", "stroke-dasharray": "3 5" }));
  [...state.axes].forEach((index) => {
    const emotion = state.data.emotions[index];
    const path = matrix.map((row, point) => `${point ? "L" : "M"} ${x(point)} ${y(row[index])}`).join(" ");
    svg.append(svgNode("path", { d: path, fill: "none", stroke: emotion.color, "stroke-width": 2.2, "stroke-linejoin": "round", "stroke-linecap": "round" }));
    matrix.forEach((row, point) => {
      const group = svgNode("g", { class: "chart-node", tabindex: 0, role: "button", "aria-label": (point ? "意元" + point : "开始前") + emotion.label + percent(row[index]) });
      group.append(svgNode("circle", { cx: x(point), cy: y(row[index]), r: 9, fill: "transparent" }));
      group.append(svgNode("circle", { cx: x(point), cy: y(row[index]), r: point === state.point ? 4.6 : 3, fill: "#fff", stroke: emotion.color, "stroke-width": 1.8 }));
      group.append(svgNode("title", {}, (point ? "意元 " + point : "开始前") + " · " + emotion.label + " " + percent(row[index])));
      const select = () => { state.point = point; renderResult(); };
      group.addEventListener("click", select);
      group.addEventListener("keydown", (event) => { if (["Enter", " "].includes(event.key)) { event.preventDefault(); select(); } });
      svg.append(group);
    });
  });
  if (state.axes.size === 0) svg.append(svgNode("text", { x: width / 2, y: height / 2, "text-anchor": "middle", class: "chart-axis-text" }, "点击上方感受标签，选择要看的曲线"));
  $("trajectoryChart").replaceChildren(svg);
}

function renderBars() {
  const prediction = activePrediction();
  const values = allStates()[state.point];
  $("pointHeading").textContent = state.point ? "意元 " + state.point + " · " + prediction.units[state.point - 1].text : "开始前的感受";
  $("pointHeading").title = $("pointHeading").textContent;
  $("emotionBars").replaceChildren();
  state.data.emotions.forEach((emotion, index) => {
    const bar = node("div", "emotion-bar");
    const track = node("div", "bar-track");
    const fill = node("div", "bar-fill");
    fill.style.width = values[index] * 100 + "%";
    fill.style.backgroundColor = emotion.color;
    track.append(fill);
    bar.append(node("span", "", emotion.label), track, node("span", "", percent(values[index])));
    $("emotionBars").append(bar);
  });
}

function renderMatrix(delta) {
  const prediction = activePrediction();
  const values = delta ? prediction.delta_matrix : allStates();
  const table = node("table", "matrix-table");
  const head = node("thead"); const row = node("tr"); row.append(node("th", "", "读到哪里"));
  state.data.emotions.forEach((emotion) => row.append(node("th", "", emotion.label)));
  head.append(row); table.append(head);
  const body = node("tbody");
  values.forEach((point, index) => {
    const position = delta ? index + 1 : index;
    const line = node("tr", position === state.point ? "active" : "");
    const label = node("th", "", position ? "意元 " + position : "开始前");
    if (position) label.title = prediction.units[position - 1].text;
    label.scope = "row";
    line.append(label);
    point.forEach((value, emotion) => {
      const cell = node("td", "", percent(value, delta));
      const intensity = Math.min(1, Math.abs(value));
      cell.style.backgroundColor = delta && value < 0 ? `rgba(180,108,83,${.05 + intensity * .7})` : `rgba(90,139,96,${.045 + intensity * .75})`;
      cell.style.color = intensity > .6 ? "#fff" : delta && value < 0 ? "#9e6451" : "#58764b";
      cell.title = state.data.emotions[emotion].label + "：" + (value * 100).toFixed(4) + (delta ? " 个百分点" : "%");
      line.append(cell);
    });
    line.addEventListener("click", () => { state.point = position; renderResult(); });
    body.append(line);
  });
  table.append(body);
  $(delta ? "deltaMatrix" : "emotionMatrix").replaceChildren(table);
}

function setView(view) {
  state.view = view;
  document.querySelectorAll("[data-view]").forEach((button) => {
    const selected = button.dataset.view === view;
    button.classList.toggle("selected", selected);
    button.setAttribute("aria-selected", String(selected));
  });
  ["trajectory", "matrix", "delta", "json"].forEach((name) => { $(name === "matrix" ? "matrixView" : name + "View").hidden = name !== view; });
}

function downloadJSON(value, filename) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(value, null, 2)], { type: "application/json;charset=utf-8" }));
  const link = node("a"); link.href = url; link.download = filename; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function wire() {
  $("analysisForm").addEventListener("submit", analyze);
  $("emptyAnalyze").addEventListener("click", analyze);
  $("reanalyzeButton").addEventListener("click", analyze);
  $("profileSelect").addEventListener("change", chooseProfile);
  $("exampleSelect").addEventListener("change", () => chooseExample($("exampleSelect").value));
  document.querySelectorAll("[data-mode]").forEach((button) => button.addEventListener("click", () => setMode(button.dataset.mode)));
  document.querySelectorAll("[data-view]").forEach((button) => button.addEventListener("click", () => setView(button.dataset.view)));
  ["textInput", "contextInput", "characterInput", "profileJSON", "customName", "compareJSON"].forEach((id) => $(id).addEventListener("input", () => { if (["textInput", "contextInput", "characterInput"].includes(id)) $("exampleSelect").value = ""; if (id === "profileJSON") renderProfileDetails(); countText(); markDraft(); }));
  $("compareToggle").addEventListener("change", () => { $("comparePanel").hidden = !$("compareToggle").checked; markDraft(); });
  $("compareSelect").addEventListener("change", () => { $("customComparePanel").hidden = $("compareSelect").value !== "__custom__"; markDraft(); });
  $("clearInput").addEventListener("click", () => { $("textInput").value = ""; $("contextInput").value = ""; $("exampleSelect").value = ""; countText(); markDraft(); $("textInput").focus(); });
  $("editPreset").addEventListener("click", () => {
    const preset = state.data.profiles.find((p) => p.id === $("profileSelect").value);
    $("profileJSON").value = JSON.stringify(preset.profile || new Array(12).fill(0), null, 2);
    $("customName").value = preset.name + "（调整）";
    setMode("custom");
  });
  $("importVector").addEventListener("click", () => $("profileFile").click());
  $("profileFile").addEventListener("change", async () => {
    try {
      const file = $("profileFile").files[0];
      if (!file) return;
      if (file.size > 512 * 1024) throw new Error("向量文件超过512KiB，请检查是否选成了完整数据集。");
      const value = readJSON(await file.text(), "个人向量");
      $("profileJSON").value = JSON.stringify(value, null, 2);
      $("customName").value = file.name.replace(/\.json$/i, "");
      setMode("custom"); toast("向量已导入。");
    } catch (error) { showError(error); }
    finally { $("profileFile").value = ""; }
  });
  $("downloadVector").addEventListener("click", () => { try { downloadJSON(readJSON($("profileJSON").value, "个人向量"), "personal_profile.json"); } catch (error) { showError(error); } });
  $("historyExample").addEventListener("click", () => { $("historyJSON").value = JSON.stringify(state.data.history_example, null, 2); toast("已填入格式演示，请替换为自己的历史作答。"); });
  $("generateProfile").addEventListener("click", async () => {
    try {
      const payload = {kind: "profile", history: readJSON($("historyJSON").value, "历史作答"), profile_generator: $("profileGenerator").value};
      const asOf = $("historyAsOf")?.value.trim();
      if (asOf && state.data.supports_history_time_cutoff === true) payload.as_of = asOf;
      await runTask(payload);
    } catch (error) { showError(error); }
  });
  $("loadModel").addEventListener("click", () => runTask({ kind: "load" }));
  $("unloadModel").addEventListener("click", () => runTask({ kind: "unload" }));
  $("autoScale").addEventListener("change", () => { if (state.result) renderChart(); });
  $("exportResult").addEventListener("click", () => { if (state.result) downloadJSON(state.result, "meaning_feeling_result.json"); });
  $("copyJSON").addEventListener("click", async () => { try { await navigator.clipboard.writeText($("rawJSON").textContent); toast("JSON已复制。"); } catch (_) { toast("浏览器没有允许复制，可以使用“导出结果”。"); } });
  ["helpButton", "footerHelp"].forEach((id) => $(id).addEventListener("click", () => $("helpDialog").showModal()));
  ["closeHelp", "finishHelp"].forEach((id) => $(id).addEventListener("click", () => $("helpDialog").close()));
  $("helpDialog").addEventListener("click", (event) => { if (event.target === $("helpDialog")) { const box = event.target.getBoundingClientRect(); if (event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom) event.target.close(); } });
  document.addEventListener("keydown", (event) => { if ((event.ctrlKey || event.metaKey) && event.key === "Enter" && !$("helpDialog").open) { event.preventDefault(); analyze(); } });
}

async function init() {
  wire();
  try {
    state.data = await api("/api/bootstrap");
    const cutoffOptions = $("historyCutoffOptions");
    if (cutoffOptions) cutoffOptions.hidden = state.data.supports_history_time_cutoff !== true;
    $("releaseInfo").textContent = state.data.release ? `版本 ${state.data.release} · 本人历史均值融合已接入` : "";
    const generators = state.data.profile_generators || [{id: "original", name: "原版校准 · 12维", configured: true}, {id: "reliability", name: "可靠性校准 · 12维", configured: false}, {id: "body_gain", name: "初始参考与正文变化 · 23维", configured: false}];
    $("profileGenerator").replaceChildren();
    generators.forEach((g) => { const option = node("option", "", g.name + (g.configured ? "" : "（未配置）")); option.value = g.id; option.disabled = !g.configured; $("profileGenerator").append(option); });
    $("profileGenerator").value = state.data.default_profile_generator || "original";
    for (const id of ["profileSelect", "compareSelect"]) {
      $(id).replaceChildren();
      state.data.profiles.forEach((p) => { const option = node("option", "", p.name + " · " + p.source_label); option.value = p.id; $(id).append(option); });
    }
    const customOption = node("option", "", "输入第二个向量…"); customOption.value = "__custom__"; $("compareSelect").append(customOption);
    const real = state.data.profiles.filter((p) => p.source === "human_calibrated");
    const latest = real.filter((p) => p.profile?.profile_kind === "phase_history_mean22");
    $("profileSelect").value = latest[0]?.id || real[0]?.id || "shared";
    $("compareSelect").value = latest[1]?.id || real[1]?.id || "shared";
    $("exampleSelect").append(node("option", "", "选择一个例子，或直接填写下面的内容"));
    $("exampleSelect").firstElementChild.value = "";
    state.data.examples.forEach((example) => { const option = node("option", "", example.name + " · " + example.category); option.value = example.id; $("exampleSelect").append(option); });
    $("exampleSelect").value = state.data.examples[0].id;
    chooseExample(state.data.examples[0].id);
    chooseProfile();
    const status = await refreshStatus();
    if (status?.busy && status.active_job) {
      const job = await api("/api/jobs/" + status.active_job);
      busy(true, job.kind);
      followJob(job.id, job.kind).catch(showError).finally(async () => { busy(false); await refreshStatus(); });
    }
    setInterval(() => { if (!document.hidden && !state.busy) refreshStatus(); }, 10000);
  } catch (error) { showError(error); }
}

if (typeof module !== "undefined" && module.exports) module.exports = {profileDetails};
else init();
