const energyCanvas = document.querySelector("#energy-chart");
const waveformCanvas = document.querySelector("#waveform-chart");
const energyValue = document.querySelector("#energy-value");
const thresholdValue = document.querySelector("#threshold-value");
const thresholdLegend = document.querySelector("#threshold-legend");
const peakValue = document.querySelector("#peak-value");
const captureState = document.querySelector("#capture-state");
const sampleRate = document.querySelector("#sample-rate");
const connectionStatus = document.querySelector("#connection-status");

const history = [];
const maxPoints = 240;
let latestWaveform = [];
let threshold = 0.015;
let lastSequence = -1;

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
  const observedMax = Math.max(threshold * 2, 0.001, ...history);
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
    const gradient = ctx.createLinearGradient(0, pad.top, 0, height - pad.bottom);
    gradient.addColorStop(0, "rgba(88, 224, 176, 0.26)");
    gradient.addColorStop(1, "rgba(88, 224, 176, 0.01)");
    const x = (index) => pad.left + (index / (maxPoints - 1)) * chartWidth;
    ctx.beginPath();
    history.forEach((value, index) => {
      if (index === 0) ctx.moveTo(x(index), y(value));
      else ctx.lineTo(x(index), y(value));
    });
    ctx.lineTo(x(history.length - 1), height - pad.bottom);
    ctx.lineTo(pad.left, height - pad.bottom);
    ctx.closePath();
    ctx.fillStyle = gradient;
    ctx.fill();

    ctx.beginPath();
    history.forEach((value, index) => {
      if (index === 0) ctx.moveTo(x(index), y(value));
      else ctx.lineTo(x(index), y(value));
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
      history.push(state.energy);
      if (history.length > maxPoints) history.shift();
      latestWaveform = state.waveform;
      lastSequence = state.sequence;
    }
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
