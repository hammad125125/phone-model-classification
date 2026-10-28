/**
 * Mobile Phone Model Classification - Client-Side Interaction & Inference Controller
 */

document.addEventListener("DOMContentLoaded", () => {
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("fileInput");
  const previewContainer = document.getElementById("previewContainer");
  const previewImage = document.getElementById("previewImage");
  const resultCard = document.getElementById("resultCard");
  const loadingIndicator = document.getElementById("loadingIndicator");
  const errorAlert = document.getElementById("errorAlert");
  const resetBtn = document.getElementById("resetBtn");
  const sampleThumbs = document.querySelectorAll(".sample-thumb");

  if (!dropzone || !fileInput) return;

  const ALLOWED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp", ".bmp"];
  const MAX_FILE_SIZE = 16 * 1024 * 1024; // 16 MB

  // 1. Drag and Drop events
  ["dragenter", "dragover"].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove("dragover");
    });
  });

  dropzone.addEventListener("drop", (e) => {
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFile(files[0]);
    }
  });

  // 2. Click to upload
  dropzone.addEventListener("click", () => fileInput.click());

  fileInput.addEventListener("change", (e) => {
    if (fileInput.files.length > 0) {
      handleFile(fileInput.files[0]);
    }
  });

  // 3. Sample thumbnail quick-select
  sampleThumbs.forEach(thumb => {
    thumb.addEventListener("click", async () => {
      const imgUrl = thumb.getAttribute("data-src");
      if (!imgUrl) return;

      try {
        showLoading("Fetching sample test image...");
        const response = await fetch(imgUrl);
        const blob = await response.blob();
        const filename = imgUrl.split("/").pop() || "sample.jpg";
        const file = new File([blob], filename, { type: blob.type || "image/jpeg" });
        handleFile(file);
      } catch (err) {
        showError("Failed to load sample image for inference: " + err.message);
      }
    });
  });

  // 4. File handling & validation
  function handleFile(file) {
    clearErrors();
    const fileName = file.name.toLowerCase();
    const isValidExt = ALLOWED_EXTENSIONS.some(ext => fileName.endsWith(ext));

    if (!isValidExt) {
      showError(`Unsupported file format. Please upload an image with extensions: ${ALLOWED_EXTENSIONS.join(", ")}`);
      return;
    }

    if (file.size > MAX_FILE_SIZE) {
      showError(`File is too large (${(file.size / (1024 * 1024)).toFixed(1)}MB). Max size is 16MB.`);
      return;
    }

    // Display image preview
    const reader = new FileReader();
    reader.onload = (e) => {
      previewImage.src = e.target.result;
      previewContainer.style.display = "block";
    };
    reader.readAsDataURL(file);

    // Run inference via API
    runInference(file);
  }

  // 5. REST API Inference Request
  async function runInference(file) {
    showLoading("Running MobileNetV2 Deep Learning Inference...");
    if (resultCard) resultCard.style.display = "none";

    const formData = new FormData();
    formData.append("file", file);
    formData.append("top_k", "3");

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        body: formData
      });

      const data = await response.json();
      hideLoading();

      if (!response.ok || !data.success) {
        if (data.model_status === "untrained") {
          showError(
            `<strong>Model is Untrained:</strong> ${data.error}<br>` +
            `<div style="margin-top: 0.5rem; font-family: monospace; background: #fee2e2; padding: 0.4rem; border-radius: 4px;">` +
            `Run in terminal: <strong>python train.py</strong></div>`
          );
        } else {
          showError(data.error || "Prediction request could not be processed.");
        }
        return;
      }

      displayResults(data);

    } catch (err) {
      hideLoading();
      showError("Connection error while communicating with classification API: " + err.message);
    }
  }

  // 6. Result Rendering
  function displayResults(data) {
    if (!resultCard) return;

    const classEl = document.getElementById("resPredictedClass");
    const confEl = document.getElementById("resConfidence");
    const timeEl = document.getElementById("resInferenceTime");
    const topBreakdownEl = document.getElementById("resTopBreakdown");

    if (classEl) classEl.textContent = data.predicted_class;
    if (confEl) confEl.textContent = `${data.confidence_percentage}% Model Probability`;
    if (timeEl) timeEl.textContent = `Inference latency: ${data.inference_time_ms} ms`;

    // Render top-k bars
    if (topBreakdownEl && data.top_predictions) {
      topBreakdownEl.innerHTML = "";
      data.top_predictions.forEach(item => {
        const itemDiv = document.createElement("div");
        itemDiv.className = "confidence-item";
        itemDiv.innerHTML = `
          <div class="confidence-header">
            <span>${item.class}</span>
            <span>${item.confidence_percentage}%</span>
          </div>
          <div class="progress-track">
            <div class="progress-bar-fill" style="width: ${Math.max(4, item.confidence_percentage)}%;"></div>
          </div>
        `;
        topBreakdownEl.appendChild(itemDiv);
      });
    }

    resultCard.style.display = "block";
    resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  // 7. Reset Action
  if (resetBtn) {
    resetBtn.addEventListener("click", () => {
      fileInput.value = "";
      previewContainer.style.display = "none";
      if (resultCard) resultCard.style.display = "none";
      clearErrors();
    });
  }

  // UI Helper functions
  function showLoading(msg) {
    if (loadingIndicator) {
      loadingIndicator.textContent = msg || "Processing...";
      loadingIndicator.style.display = "block";
    }
  }

  function hideLoading() {
    if (loadingIndicator) loadingIndicator.style.display = "none";
  }

  function showError(msg) {
    if (errorAlert) {
      errorAlert.innerHTML = msg;
      errorAlert.style.display = "block";
    }
  }

  function clearErrors() {
    if (errorAlert) errorAlert.style.display = "none";
  }
});

