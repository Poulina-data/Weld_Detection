/**
 * YOLO Vision Studio - Image Detection Module
 * Handles automatic YOLO inference upon file upload or sample selection,
 * debounced slider updates, side-by-side comparison, and result downloads.
 */

const ImageDetectionModule = (() => {
    // Internal State
    const state = {
        selectedFile: null,
        selectedSampleId: null,
        originalImageSrc: null,
        annotatedImageSrc: null,
        lastResultData: null,
        isProcessing: false,
        isCompareActive: false,
    };

    // Stale response prevention token & debounce timer
    let currentRequestId = 0;
    let sliderDebounceTimer = null;

    // DOM Selectors
    const elements = {
        fileInput: document.getElementById("imageFileInput"),
        uploadDropzone: document.getElementById("uploadDropzone"),
        btnSelectImageBottom: document.getElementById("btnSelectImageBottom"),
        btnBrowseOriginal: document.getElementById("btnBrowseOriginal"),
        btnTopUploadImage: document.getElementById("btnTopUploadImage"),
        
        originalViewer: document.getElementById("originalViewer"),
        originalEmptyState: document.getElementById("originalEmptyState"),
        previewWrapper: document.getElementById("originalPreviewWrapper"),
        originalPreview: document.getElementById("originalImagePreview"),
        originalFileName: document.getElementById("originalFileName"),
        originalDims: document.getElementById("originalDims"),
        originalFileSize: document.getElementById("originalFileSize"),
        uploadStatusBadge: document.getElementById("uploadStatusBadge"),
        
        confSlider: document.getElementById("imageConfSlider"),
        confValue: document.getElementById("imageConfValue"),
        iouSlider: document.getElementById("imageIouSlider"),
        iouValue: document.getElementById("imageIouValue"),
        
        btnReset: document.getElementById("btnResetImage"),
        btnDownload: document.getElementById("btnDownloadAnnotated"),
        btnToggleSplit: document.getElementById("btnToggleSplit"),
        
        resultViewer: document.getElementById("resultViewer"),
        emptyResultState: document.getElementById("emptyResultState"),
        loadingOverlay: document.getElementById("imageLoadingOverlay"),
        loadingMessage: document.getElementById("loadingMessage"),
        annotatedWrapper: document.getElementById("annotatedWrapper"),
        annotatedResult: document.getElementById("annotatedImageResult"),
        detectionStatusBadge: document.getElementById("detectionStatusBadge"),
        
        verdictBanner: document.getElementById("verdictBanner"),
        verdictTitle: document.getElementById("verdictTitle"),
        verdictDesc: document.getElementById("verdictDesc"),
        
        metricsGrid: document.getElementById("metricsGrid"),
        metricTotalObjects: document.getElementById("metricTotalObjects"),
        metricUniqueClasses: document.getElementById("metricUniqueClasses"),
        metricInferenceTime: document.getElementById("metricInferenceTime"),
        metricDefectsCount: document.getElementById("metricDefectsCount"),
        
        tablePanel: document.getElementById("tablePanel"),
        tableBody: document.getElementById("detectionTableBody"),
        samplesList: document.getElementById("samplesList"),
    };

    function init() {
        setupEventListeners();
        setupDragAndDrop();
        setupSliders();
        setupSampleChips();
    }

    function setupEventListeners() {
        // File selection
        if (elements.fileInput) {
            elements.fileInput.addEventListener("change", handleFileSelect);
        }

        // Browse / Select buttons
        if (elements.btnTopUploadImage) {
            elements.btnTopUploadImage.addEventListener("click", () => {
                if (elements.fileInput) elements.fileInput.click();
            });
        }
        if (elements.btnBrowseOriginal) {
            elements.btnBrowseOriginal.addEventListener("click", () => {
                if (elements.fileInput) elements.fileInput.click();
            });
        }
        if (elements.btnSelectImageBottom) {
            elements.btnSelectImageBottom.addEventListener("click", () => {
                if (elements.fileInput) elements.fileInput.click();
            });
        }

        // Control buttons
        if (elements.btnReset) {
            elements.btnReset.addEventListener("click", resetImageMode);
        }
        if (elements.btnDownload) {
            elements.btnDownload.addEventListener("click", downloadAnnotatedImage);
        }
        if (elements.btnToggleSplit) {
            elements.btnToggleSplit.addEventListener("click", toggleCompareView);
        }
    }

    // Debounced slider adjustments
    function setupSliders() {
        if (elements.confSlider) {
            elements.confSlider.addEventListener("input", (e) => {
                elements.confValue.textContent = `${e.target.value}%`;
                triggerDebouncedDetection();
            });
        }

        if (elements.iouSlider) {
            elements.iouSlider.addEventListener("input", (e) => {
                elements.iouValue.textContent = `${e.target.value}%`;
                triggerDebouncedDetection();
            });
        }
    }

    function triggerDebouncedDetection() {
        if (!state.selectedFile && !state.selectedSampleId) return;

        clearTimeout(sliderDebounceTimer);
        sliderDebounceTimer = setTimeout(() => {
            runDetection(true);
        }, 250);
    }

    function setupSampleChips() {
        if (!elements.samplesList) return;
        const chips = elements.samplesList.querySelectorAll(".sample-chip");
        chips.forEach((chip) => {
            chip.addEventListener("click", () => {
                const sampleName = chip.dataset.sample;
                loadSampleImage(sampleName);
            });
        });
    }

    // Drag and Drop Handling on Bottom Upload Zone
    function setupDragAndDrop() {
        const dropzone = elements.uploadDropzone;
        if (!dropzone) return;

        ["dragenter", "dragover"].forEach((eventName) => {
            dropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropzone.classList.add("dragover");
            });
        });

        ["dragleave", "drop"].forEach((eventName) => {
            dropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropzone.classList.remove("dragover");
            });
        });

        dropzone.addEventListener("drop", (e) => {
            const files = e.dataTransfer.files;
            if (files && files.length > 0) {
                processSelectedFile(files[0]);
            }
        });

        // Also allow drag-and-drop onto the source viewer if user drops there
        if (elements.originalViewer) {
            ["dragenter", "dragover"].forEach((eventName) => {
                elements.originalViewer.addEventListener(eventName, (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                });
            });
            elements.originalViewer.addEventListener("drop", (e) => {
                e.preventDefault();
                e.stopPropagation();
                const files = e.dataTransfer.files;
                if (files && files.length > 0) {
                    processSelectedFile(files[0]);
                }
            });
        }
    }

    function handleFileSelect(e) {
        const files = e.target.files;
        if (files && files.length > 0) {
            processSelectedFile(files[0]);
        }
    }

    // Process Selected File & Trigger Automatic Detection
    function processSelectedFile(file) {
        // Validate type
        const allowedTypes = ["image/jpeg", "image/jpg", "image/png", "image/webp"];
        if (!allowedTypes.includes(file.type.toLowerCase()) && !file.name.match(/\.(jpg|jpeg|png|webp)$/i)) {
            App.showToast("Unsupported file type. Please upload a JPG, PNG, or WEBP image.", "error");
            return;
        }

        // Validate size (10 MB limit)
        const maxBytes = 10 * 1024 * 1024;
        if (file.size > maxBytes) {
            App.showToast("File is larger than 10 MB limit.", "error");
            return;
        }

        state.selectedFile = file;
        state.selectedSampleId = null;
        state.isCompareActive = false;

        // Display preview in Original Source immediately
        const reader = new FileReader();
        reader.onload = (e) => {
            state.originalImageSrc = e.target.result;
            elements.originalPreview.src = state.originalImageSrc;

            elements.originalPreview.onload = () => {
                const w = elements.originalPreview.naturalWidth;
                const h = elements.originalPreview.naturalHeight;
                elements.originalDims.textContent = `${w} × ${h} px`;
            };

            if (elements.originalFileName) {
                elements.originalFileName.textContent = file.name;
            }
            elements.originalFileSize.textContent = formatBytes(file.size);
            
            // Switch views
            elements.originalEmptyState.style.display = "none";
            elements.previewWrapper.style.display = "flex";
            elements.uploadStatusBadge.textContent = "Image Loaded";
            elements.btnReset.disabled = false;

            // Immediately trigger automatic YOLO detection!
            runDetection();
        };
        reader.readAsDataURL(file);
    }

    // Load Quick Sample & Trigger Automatic Detection
    async function loadSampleImage(sampleFileName) {
        state.selectedFile = null;
        state.selectedSampleId = sampleFileName;
        state.isCompareActive = false;

        elements.uploadStatusBadge.textContent = "Loading Sample...";

        const sampleUrl = `/static/samples/${sampleFileName}`;
        state.originalImageSrc = sampleUrl;
        elements.originalPreview.src = sampleUrl;

        elements.originalPreview.onload = () => {
            const w = elements.originalPreview.naturalWidth;
            const h = elements.originalPreview.naturalHeight;
            elements.originalDims.textContent = `${w} × ${h} px`;
            if (elements.originalFileName) {
                elements.originalFileName.textContent = sampleFileName;
            }
            elements.originalFileSize.textContent = "Sample Dataset";
            
            elements.originalEmptyState.style.display = "none";
            elements.previewWrapper.style.display = "flex";
            elements.uploadStatusBadge.textContent = "Sample Active";
            elements.btnReset.disabled = false;

            // Automatically run detection on sample
            runDetection();
        };
    }

    // Automatic YOLO Inference
    async function runDetection(isDebounced = false) {
        if (!state.selectedFile && !state.selectedSampleId) return;

        // Increment request ID to invalidate any prior in-flight requests
        const requestId = ++currentRequestId;

        setProcessingState(true);

        const conf = parseFloat(elements.confSlider.value) / 100.0;
        const iou = parseFloat(elements.iouSlider.value) / 100.0;

        try {
            let res;
            if (state.selectedSampleId) {
                // Run on sample endpoint with conf and iou
                res = await fetch("/api/detect-sample", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        sample_id: state.selectedSampleId,
                        conf: conf,
                        iou: iou,
                    }),
                });
            } else {
                // Upload standard multipart form
                const formData = new FormData();
                formData.append("image", state.selectedFile);
                formData.append("conf", conf.toString());
                formData.append("iou", iou.toString());

                res = await fetch("/api/detect", {
                    method: "POST",
                    body: formData,
                });
            }

            const data = await res.json();

            // Check if another request was triggered while this one was in-flight
            if (requestId !== currentRequestId) {
                return; // Discard stale response
            }

            if (!res.ok || !data.success) {
                throw new Error(data.error || `HTTP ${res.status}`);
            }

            // Success: update state and render
            state.lastResultData = data;
            state.annotatedImageSrc = data.image;
            renderDetectionResults(data);

            if (!isDebounced) {
                App.showToast(`Detection complete in ${data.inference_time_ms} ms.`, "success");
            }

        } catch (err) {
            if (requestId !== currentRequestId) return;
            console.error("Detection error:", err);
            App.showToast(`Detection failed: ${err.message}`, "error");
            elements.detectionStatusBadge.textContent = "Inference Failed";
        } finally {
            if (requestId === currentRequestId) {
                setProcessingState(false);
            }
        }
    }

    // Render Detection Results
    function renderDetectionResults(data) {
        elements.emptyResultState.style.display = "none";
        elements.annotatedWrapper.style.display = "flex";
        elements.annotatedResult.src = data.image;

        elements.detectionStatusBadge.textContent = `${data.detection_count} detected (${data.inference_time_ms} ms)`;
        elements.btnDownload.disabled = false;
        elements.btnToggleSplit.disabled = false;

        // Reset compare button state if it was toggled
        elements.btnToggleSplit.classList.remove("btn-primary");
        elements.btnToggleSplit.classList.add("btn-ghost");
        state.isCompareActive = false;

        // Render Verdict
        const verdict = data.verdict;
        elements.verdictBanner.style.display = "flex";
        elements.verdictBanner.className = `verdict-banner verdict-${verdict.badge || 'info'}`;
        elements.verdictTitle.textContent = verdict.label;
        elements.verdictDesc.textContent = verdict.summary || `Top Confidence: ${verdict.confidence || 'N/A'}`;

        // Render Metrics
        elements.metricsGrid.style.display = "grid";
        elements.metricTotalObjects.textContent = data.detection_count;
        elements.metricUniqueClasses.textContent = data.unique_classes_count;
        elements.metricInferenceTime.textContent = `${data.inference_time_ms} ms`;

        // Defect count tally
        let defectTally = 0;
        if (data.detections) {
            data.detections.forEach((d) => {
                if (d.class_id in { 0: 1, 2: 1 } || /defect|bad|crack/i.test(d.class_name)) {
                    defectTally++;
                }
            });
        }
        elements.metricDefectsCount.textContent = defectTally;

        // Render Table
        renderDetectionTable(data.detections);
    }

    function renderDetectionTable(detections) {
        elements.tablePanel.style.display = "flex";
        elements.tableBody.innerHTML = "";

        if (!detections || detections.length === 0) {
            elements.tableBody.innerHTML = `
                <tr>
                    <td colspan="6" style="text-align: center; color: var(--text-muted); padding: 1.5rem;">
                        No objects detected at current confidence threshold.
                    </td>
                </tr>
            `;
            return;
        }

        detections.forEach((det) => {
            const tr = document.createElement("tr");

            const isDefect = det.class_id in { 0: 1, 2: 1 } || /defect|bad|crack/i.test(det.class_name);
            const verdictBadge = isDefect
                ? `<span class="tag-class" style="background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4);">DEFECT</span>`
                : `<span class="tag-class" style="background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4);">PASS</span>`;

            const boxWidth = det.bbox[2] - det.bbox[0];
            const boxHeight = det.bbox[3] - det.bbox[1];

            tr.innerHTML = `
                <td><strong>#${det.id}</strong></td>
                <td>
                    <span class="tag-class" style="background: ${det.color}22; color: ${det.color}; border: 1px solid ${det.color}66;">
                        ${det.class_name}
                    </span>
                </td>
                <td>
                    <div class="conf-cell-wrap">
                        <div class="conf-bar-bg">
                            <div class="conf-bar-fill" style="width: ${det.confidence * 100}%;"></div>
                        </div>
                        <span style="font-family: var(--font-mono); font-weight: 600;">${det.confidence_pct}</span>
                    </div>
                </td>
                <td class="coord-cell">[${det.bbox.join(", ")}]</td>
                <td style="font-family: var(--font-mono);">${boxWidth} × ${boxHeight}</td>
                <td>${verdictBadge}</td>
            `;
            elements.tableBody.appendChild(tr);
        });
    }

    // Toggle Compare View (Switch between original and annotated in right panel)
    function toggleCompareView() {
        if (!state.annotatedImageSrc || !state.originalImageSrc) {
            App.showToast("Run detection first to compare images.", "info");
            return;
        }

        state.isCompareActive = !state.isCompareActive;
        if (state.isCompareActive) {
            elements.annotatedResult.src = state.originalImageSrc;
            elements.btnToggleSplit.classList.add("btn-primary");
            elements.btnToggleSplit.classList.remove("btn-ghost");
            App.showToast("Showing Original Image in Result Viewer.", "info");
        } else {
            elements.annotatedResult.src = state.annotatedImageSrc;
            elements.btnToggleSplit.classList.remove("btn-primary");
            elements.btnToggleSplit.classList.add("btn-ghost");
            App.showToast("Showing Annotated Detections in Result Viewer.", "info");
        }
    }

    // Download Annotated Image
    function downloadAnnotatedImage() {
        if (!state.annotatedImageSrc) {
            App.showToast("No annotated image to download.", "error");
            return;
        }

        const a = document.createElement("a");
        a.href = state.annotatedImageSrc;
        const origName = state.lastResultData?.original_filename || "yolo_detection_result";
        const cleanName = origName.replace(/\.[^/.]+$/, "");
        a.download = `${cleanName}_yolo_detection_result.jpg`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);

        App.showToast("Annotated image downloaded.", "success");
    }

    // UI State Management
    function setProcessingState(isProcessing) {
        state.isProcessing = isProcessing;
        elements.loadingOverlay.style.display = isProcessing ? "flex" : "none";
        if (elements.loadingMessage) {
            elements.loadingMessage.textContent = "Detecting objects...";
        }
        elements.detectionStatusBadge.textContent = isProcessing ? "Detecting objects..." : (state.lastResultData ? `${state.lastResultData.detection_count} detected` : "Ready");
    }

    function clearResultsView() {
        elements.emptyResultState.style.display = "flex";
        elements.annotatedWrapper.style.display = "none";
        elements.annotatedResult.src = "";
        elements.verdictBanner.style.display = "none";
        elements.metricsGrid.style.display = "none";
        elements.tablePanel.style.display = "none";
        elements.detectionStatusBadge.textContent = "Waiting for image";
        elements.btnDownload.disabled = true;
        elements.btnToggleSplit.disabled = true;
        state.annotatedImageSrc = null;
        state.lastResultData = null;
    }

    function resetImageMode() {
        // Invalidate any in-flight requests
        currentRequestId++;
        clearTimeout(sliderDebounceTimer);

        state.selectedFile = null;
        state.selectedSampleId = null;
        state.originalImageSrc = null;
        state.annotatedImageSrc = null;
        state.lastResultData = null;
        state.isCompareActive = false;

        if (elements.fileInput) {
            elements.fileInput.value = "";
        }
        elements.originalEmptyState.style.display = "flex";
        elements.previewWrapper.style.display = "none";
        elements.originalPreview.src = "";
        elements.uploadStatusBadge.textContent = "Awaiting Image";
        elements.btnReset.disabled = true;

        clearResultsView();
        App.showToast("Reset completed.", "info");
    }

    function formatBytes(bytes) {
        if (bytes === 0) return "0 Bytes";
        const k = 1024;
        const sizes = ["Bytes", "KB", "MB"];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + sizes[i];
    }

    return {
        init,
        resetImageMode,
    };
})();

// Initialize on DOM load
document.addEventListener("DOMContentLoaded", () => {
    ImageDetectionModule.init();
});
