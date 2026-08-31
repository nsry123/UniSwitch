const energyCanvas = document.querySelector("#energy-chart");
const waveformCanvas = document.querySelector("#waveform-chart");
const energyValue = document.querySelector("#energy-value");
const thresholdValue = document.querySelector("#threshold-value");
const thresholdLegend = document.querySelector("#threshold-legend");
const peakValue = document.querySelector("#peak-value");
const captureState = document.querySelector("#capture-state");
const sampleRate = document.querySelector("#sample-rate");
const connectionStatus = document.querySelector("#connection-status");
const resultPanel = document.querySelector("#result-panel");
const resultStatus = document.querySelector("#result-status");
const resultLabel = document.querySelector("#result-label");
const resultReason = document.querySelector("#result-reason");
const resultCandidate = document.querySelector("#result-candidate");
const resultConfidence = document.querySelector("#result-confidence");
const resultDistance = document.querySelector("#result-distance");
const resultMargin = document.querySelector("#result-margin");
const calibrationPanel = document.querySelector("#calibration-panel");
const calibrationState = document.querySelector("#calibration-state");
const calibrationTitle = document.querySelector("#calibration-title");
const calibrationMessage = document.querySelector("#calibration-message");
const calibrationProgress = document.querySelector("#calibration-progress");
const calibrationClasses = document.querySelector("#calibration-classes");
const calibrationUnknowns = document.querySelector("#calibration-unknowns");
const calibrationKnown = document.querySelector("#calibration-known");
const calibrationRejection = document.querySelector("#calibration-rejection");
const calibrationParameters = document.querySelector("#calibration-parameters");

const history = [];
const maxPoints = 240;
let latestWaveform = [];
let threshold = 0.015;
let lastSequence = -1;
let lastResultSequence = -1;
let latestFeatureWindow = null;

function renderResult(result) {
  if (!result) return;
  let state = "registration";
  let badge = "Registered";
  if (result.accepted === true) {
    state = "accepted";
    badge = result.status === "EVENT" ? "Accepted" : "Accepted · suppressed";
  } else if (result.accepted === false) {
    state = "unknown";
    badge = "Unknown";
  }
  resultPanel.className = `panel result-panel ${state}`;
  resultStatus.className = `result-badge ${state}`;
  resultStatus.textContent = badge;
  resultLabel.textContent = result.label;
  resultReason.textContent = result.reason;
  resultCandidate.textContent = result.candidate || "—";
  resultConfidence.textContent =
    result.confidence == null ? "—" : `${(result.confidence * 100).toFixed(1)}%`;
  resultDistance.textContent =
    result.score == null || result.acceptanceRadius == null
      ? "—"
      : `${result.score.toFixed(3)} / ${result.acceptanceRadius.toFixed(3)}`;
  resultMargin.textContent =
    result.margin == null
      ? result.accepted == null ? "—" : "Single class"
      : `${result.margin.toFixed(3)} / ${result.requiredMargin.toFixed(3)} required`;
}

function renderCalibration(calibration) {
  if (!calibration) return;
  const state = calibration.state || "needs_data";
  calibrationPanel.className = `panel calibration-panel ${state}`;
  calibrationState.className = `result-badge ${state === "failed" ? "unknown" : state === "passed" || state === "ready" ? "accepted" : "registration"}`;
  calibrationState.textContent = state.replace("_", " ");
  calibrationTitle.textContent = state === "passed" ? "Classifier calibrated" : "Classifier setup";
  calibrationMessage.textContent = calibration.message || "—";
  const done = calibration.evaluatedCandidates || 0;
  const total = calibration.totalCandidates || 0;
  calibrationProgress.textContent = state === "running" && total
    ? `Evaluated ${done.toLocaleString()} of ${total.toLocaleString()} candidates`
    : "";
  const counts = Object.entries(calibration.classCaptureCounts || {});
  calibrationClasses.textContent = counts.length
    ? counts.map(([label, count]) => `${label}: ${count}`).join(", ")
    : "—";
  calibrationUnknowns.textContent = `${calibration.unknownCaptureCount || 0} / 10`;
  const metrics = calibration.metrics;
  calibrationKnown.textContent = metrics ? `${(metrics.knownAccuracy * 100).toFixed(1)}%` : "—";
  calibrationRejection.textContent = metrics ? `${(metrics.unknownRejection * 100).toFixed(1)}%` : "—";
  const parameters = calibration.parameters;
  calibrationParameters.textContent = parameters
    ? `k=${parameters.n_neighbors}, distance=${parameters.distance_threshold.toFixed(4)}, prototype=${parameters.prototype_weight.toFixed(2)}, spread=${parameters.spread_multiplier.toFixed(2)}, margin=${parameters.margin_threshold.toFixed(4)}, temperature=${parameters.probability_temperature.toFixed(4)}`
    : "—";
}

function fitCanvas(canvas) {
  const ratio = window.devicePixelRatio || 1;
  const bounds = canvas.getBoundingClientRect();
  const width = Math.max(1, Math.round(bounds.width * ratio));
  const height = Math.max(1, Math.round(bounds.height * ratio));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  return { width, height, ratio };
}

function drawEnergy() {
  const { width, height, ratio } = fitCanvas(energyCanvas);
  const ctx = energyCanvas.getContext("2d");
  const pad = { top: 18 * ratio, right: 18 * ratio, bottom: 27 * ratio, left: 54 * ratio };
  const chartWidth = width - pad.left - pad.right;
  const chartHeight = height - pad.top - pad.bottom;
  const observedMax = Math.max(
    threshold * 2,
    0.001,
    ...history.map((point) => point.energy),
  );
  const yMax = observedMax * 1.18;
  const y = (value) => pad.top + chartHeight - (value / yMax) * chartHeight;

  ctx.clearRect(0, 0, width, height);
  ctx.strokeStyle = "rgba(164, 196, 186, 0.11)";
  ctx.fillStyle = "#789088";
  ctx.font = `${11 * ratio}px ui-monospace, SFMono-Regular, monospace`;
  ctx.textAlign = "right";
  for (let i = 0; i <= 4; i += 1) {
    const value = (yMax * i) / 4;
    const lineY = y(value);
    ctx.beginPath();
    ctx.moveTo(pad.left, lineY);
    ctx.lineTo(width - pad.right, lineY);
    ctx.stroke();
    ctx.fillText(value.toFixed(3), pad.left - 9 * ratio, lineY + 4 * ratio);
  }

  const thresholdY = y(threshold);
  ctx.strokeStyle = "#f4b95f";
  ctx.lineWidth = 1.5 * ratio;
  ctx.setLineDash([7 * ratio, 5 * ratio]);
  ctx.beginPath();
  ctx.moveTo(pad.left, thresholdY);
  ctx.lineTo(width - pad.right, thresholdY);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = "#f4b95f";
  ctx.textAlign = "left";
  ctx.fillText(`THRESHOLD ${threshold.toFixed(4)}`, pad.left + 8 * ratio, thresholdY - 8 * ratio);

  if (history.length > 1) {
    const firstSequence = history[0].sequence;
    const lastSequenceInHistory = history[history.length - 1].sequence;
    const gradient = ctx.createLinearGradient(0, pad.top, 0, height - pad.bottom);
    gradient.addColorStop(0, "rgba(88, 224, 176, 0.26)");
    gradient.addColorStop(1, "rgba(88, 224, 176, 0.01)");
    const x = (index) => pad.left + (index / (maxPoints - 1)) * chartWidth;

    if (
      latestFeatureWindow &&
      latestFeatureWindow.endSequence >= firstSequence &&
      latestFeatureWindow.startSequence <= lastSequenceInHistory
    ) {
      const sequenceX = (sequence) => {
        const fraction = (sequence - firstSequence) /
          Math.max(1, lastSequenceInHistory - firstSequence);
        const historyIndex = fraction * (history.length - 1);
        return x(historyIndex);
      };
      const windowStart = Math.max(pad.left, sequenceX(latestFeatureWindow.startSequence));
      const windowEnd = Math.min(
        width - pad.right,
        sequenceX(latestFeatureWindow.endSequence),
      );
      const onsetX = Math.min(
        width - pad.right,
        Math.max(pad.left, sequenceX(latestFeatureWindow.onsetSequence)),
      );
      ctx.fillStyle = "rgba(80, 150, 255, 0.16)";
      ctx.fillRect(windowStart, pad.top, Math.max(1, windowEnd - windowStart), chartHeight);
      ctx.strokeStyle = "rgba(108, 174, 255, 0.9)";
      ctx.lineWidth = 1.5 * ratio;
      ctx.setLineDash([4 * ratio, 3 * ratio]);
      ctx.beginPath();
      ctx.moveTo(onsetX, pad.top);
      ctx.lineTo(onsetX, height - pad.bottom);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "#8fc2ff";
      ctx.font = `${11 * ratio}px ui-monospace, SFMono-Regular, monospace`;
      ctx.textAlign = "left";
      ctx.fillText("BEATs FEATURE 580 ms", windowStart + 6 * ratio, pad.top + 14 * ratio);
    }

    ctx.beginPath();
    history.forEach((point, index) => {
      if (index === 0) ctx.moveTo(x(index), y(point.energy));
      else ctx.lineTo(x(index), y(point.energy));
    });
    ctx.lineTo(x(history.length - 1), height - pad.bottom);
    ctx.lineTo(pad.left, height - pad.bottom);
    ctx.closePath();
    ctx.fillStyle = gradient;
    ctx.fill();

    ctx.beginPath();
    history.forEach((point, index) => {
      if (index === 0) ctx.moveTo(x(index), y(point.energy));
      else ctx.lineTo(x(index), y(point.energy));
    });
    ctx.strokeStyle = "#58e0b0";
    ctx.lineWidth = 2 * ratio;
    ctx.stroke();
  }
}

function drawWaveform() {
  const { width, height, ratio } = fitCanvas(waveformCanvas);
  const ctx = waveformCanvas.getContext("2d");
  const center = height / 2;
  ctx.clearRect(0, 0, width, height);
  ctx.strokeStyle = "rgba(164, 196, 186, 0.16)";
  ctx.beginPath();
  ctx.moveTo(0, center);
  ctx.lineTo(width, center);
  ctx.stroke();
  if (!latestWaveform.length) return;
  const maxAmplitude = Math.max(0.05, ...latestWaveform.map(Math.abs));
  ctx.beginPath();
  latestWaveform.forEach((sample, index) => {
    const x = (index / (latestWaveform.length - 1)) * width;
    const y = center - (sample / maxAmplitude) * center * 0.78;
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.strokeStyle = "rgba(142, 238, 205, 0.88)";
  ctx.lineWidth = 1.25 * ratio;
  ctx.stroke();
}

async function poll() {
  try {
    const response = await fetch("/api/audio", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const state = await response.json();
    threshold = state.threshold;
    if (state.sequence !== lastSequence) {
      history.push({ sequence: state.sequence, energy: state.energy });
      if (history.length > maxPoints) history.shift();
      latestWaveform = state.waveform;
      lastSequence = state.sequence;
    }
    latestFeatureWindow = state.featureWindow;
    if (state.resultSequence !== lastResultSequence) {
      renderResult(state.lastResult);
      lastResultSequence = state.resultSequence;
    }
    renderCalibration(state.calibration);
    energyValue.textContent = state.energy.toFixed(4);
    thresholdValue.textContent = state.threshold.toFixed(4);
    thresholdLegend.textContent = state.threshold.toFixed(4);
    peakValue.textContent = state.peak.toFixed(4);
    captureState.textContent = state.capturing ? "Capturing" : "Idle";
    captureState.style.color = state.capturing ? "#f4b95f" : "#e9f4ef";
    sampleRate.textContent = state.sampleRate.toLocaleString();
    connectionStatus.className = state.active ? "status live" : "status";
    connectionStatus.innerHTML = `<span></span>${state.active ? "Live audio" : "Waiting for audio"}`;
  } catch (_error) {
    connectionStatus.className = "status stale";
    connectionStatus.innerHTML = "<span></span>Monitor disconnected";
  }
  drawEnergy();
  drawWaveform();
}

window.addEventListener("resize", () => {
  drawEnergy();
  drawWaveform();
});
setInterval(poll, 50);
poll();
