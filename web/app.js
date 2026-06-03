const modelSelect = document.getElementById("model");
const personSelect = document.getElementById("person_id");
const shapeSelect = document.getElementById("shape_id");
const textInput = document.getElementById("text");
const audioInput = document.getElementById("audio");
const submitButton = document.getElementById("submit");
const message = document.getElementById("message");
const statusBox = document.getElementById("status");
const resultTitle = document.getElementById("result-title");
const resultSubtitle = document.getElementById("result-subtitle");
const jobId = document.getElementById("job-id");
const jobStatus = document.getElementById("job-status");
const video = document.getElementById("video");
const placeholder = document.getElementById("placeholder");
const download = document.getElementById("download");

const labels = {
  queued: "排队中",
  running: "生成中",
  completed: "已完成",
  failed: "失败",
  Ready: "就绪",
};

function setStatus(value) {
  statusBox.textContent = labels[value] || value;
  jobStatus.textContent = labels[value] || value;
}

function fillSelect(select, items, getValue, getLabel) {
  select.innerHTML = "";
  items.forEach((item) => {
    const option = document.createElement("option");
    option.value = getValue(item);
    option.textContent = getLabel(item);
    select.appendChild(option);
  });
}

async function loadOptions() {
  const response = await fetch("/api/options");
  const data = await response.json();
  fillSelect(modelSelect, data.models, (item) => item.key, (item) => item.label);
  fillSelect(personSelect, data.person_ids, (item) => item, (item) => item);
  fillSelect(shapeSelect, data.shape_options, (item) => item.key, (item) => item.label);
}

function currentShapeLabel() {
  return shapeSelect.options[shapeSelect.selectedIndex]?.textContent || shapeSelect.value;
}

function showJob(job) {
  jobId.textContent = job.id || "-";
  setStatus(job.status || "Ready");
  resultTitle.textContent = modelSelect.options[modelSelect.selectedIndex]?.textContent || job.model;
  resultSubtitle.textContent = `${job.person_id || personSelect.value} / ${job.shape_id || currentShapeLabel()}`;
  message.textContent = job.status === "failed" ? job.error || "生成失败，请查看终端日志。" : "";

  if (job.video_url) {
    const videoUrl = `${job.video_url}?t=${Date.now()}`;
    video.src = videoUrl;
    download.href = job.video_url;
    video.classList.remove("hidden");
    download.classList.remove("hidden");
    placeholder.classList.add("hidden");
  }
}

async function pollJob(id) {
  const timer = window.setInterval(async () => {
    const response = await fetch(`/api/jobs/${id}`);
    const job = await response.json();
    showJob(job);
    if (job.status === "completed" || job.status === "failed") {
      window.clearInterval(timer);
      submitButton.disabled = false;
    }
  }, 1500);
}

submitButton.addEventListener("click", async () => {
  message.textContent = "";
  const file = audioInput.files[0];
  if (!file) {
    message.textContent = "请先选择语音文件。";
    return;
  }
  if (!textInput.value.trim()) {
    message.textContent = "文本条件不能为空。";
    return;
  }

  const form = new FormData();
  form.append("model", modelSelect.value);
  form.append("text", textInput.value.trim());
  form.append("person_id", personSelect.value);
  form.append("shape_id", shapeSelect.value);
  form.append("audio", file);

  submitButton.disabled = true;
  setStatus("queued");
  const response = await fetch("/api/jobs", { method: "POST", body: form });
  const job = await response.json();
  if (!response.ok) {
    submitButton.disabled = false;
    message.textContent = job.detail || "任务提交失败。";
    return;
  }
  showJob(job);
  pollJob(job.id);
});

loadOptions().catch(() => {
  message.textContent = "后端服务不可用，请确认 run_app.py 已启动。";
});
