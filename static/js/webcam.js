/**
 * YOLO Vision Studio - Real-Time Webcam & Live Feed Module
 * Robust video capture without black-screen freeze, Live Detection toggle,
 * plus dedicated Upload Image, Clear Image, Sample Selector, and Drag & Drop controls.
 */

const WebcamModule = (() => {
    // Internal State
    const state = {
        mediaStream: null,
        isCameraActive: false,
        isLiveDetectionEnabled: true,
        isInFlight: false,         // Single in-flight request lock
        captureIntervalId: null,
        fpsIntervalId: null,
        frameCount: 0,
        currentFps: 0.0,
        totalFramesProcessed: 0,
        targetFps: 8,
        confThreshold: 0.25,
        iouThreshold: 0.45,
        activeImageSource: null,   // null = webcam, or 'upload' / 'sample'
    };

    // Offscreen Canvas for Frame Extraction
    let offscreenCanvas = null;
    let offscreenCtx = null;

    // DOM Elements
    const elements = {
        btnToggleLiveDetection: document.getElementById("btnToggleLiveDetection"),
        liveToggleDot: document.getElementById("liveToggleDot"),
        liveDetectionBtnText: document.getElementById("liveDetectionBtnText"),
        btnCloseCamera: document.getElementById("btnCloseCamera"),
        btnRetryCamera: document.getElementById("btnRetryCamera"),
        
        btnUploadWebcamImage: document.getElementById("btnUploadWebcamImage"),
        webcamFileInput: document.getElementById("webcamFileInput"),
        btnClearWebcamImage: document.getElementById("btnClearWebcamImage"),
        btnResumeLiveCamera: document.getElementById("btnResumeLiveCamera"),
        webcamSamplesList: document.getElementById("webcamSamplesList"),
        
        confSlider: document.getElementById("webcamConfSlider"),
        confValue: document.getElementById("webcamConfValue"),
        fpsSlider: document.getElementById("webcamFpsSlider"),
        fpsValue: document.getElementById("webcamFpsValue"),
        
        videoContainer: document.getElementById("videoContainer"),
        livePreview: document.getElementById("webcamLivePreview"),
        annotatedResult: document.getElementById("webcamAnnotatedResult"),
        offlinePlaceholder: document.getElementById("webcamOfflinePlaceholder"),
        offlineHeading: document.getElementById("webcamOfflineHeading"),
        offlineMsg: document.getElementById("webcamOfflineMsg"),
        
        cameraStatusBadge: document.getElementById("cameraStatusBadge"),
        webcamSourceBadge: document.getElementById("webcamSourceBadge"),
        resolutionBadge: document.getElementById("webcamResolutionBadge"),
        fpsBadge: document.getElementById("webcamFpsBadge"),
        
        hudOverlay: document.getElementById("videoHudOverlay"),
        hudDot: document.getElementById("hudDot"),
        hudStatusText: document.getElementById("hudStatusText"),
        hudLatency: document.getElementById("hudLatency"),
        hudVerdictPill: document.getElementById("hudVerdictPill"),
        
        liveVerdictCard: document.getElementById("liveVerdictCard"),
        liveVerdictTag: document.getElementById("liveVerdictTag"),
        liveVerdictLabel: document.getElementById("liveVerdictLabel"),
        liveVerdictDesc: document.getElementById("liveVerdictDesc"),
        
        liveDetectionsCount: document.getElementById("liveDetectionsCount"),
        liveInferenceTime: document.getElementById("liveInferenceTime"),
        liveTotalProcessed: document.getElementById("liveTotalProcessed"),
        liveEffectiveFps: document.getElementById("liveEffectiveFps"),
        classesTallyList: document.getElementById("classesTallyList"),
    };

    function init() {
        offscreenCanvas = document.createElement("canvas");
        offscreenCtx = offscreenCanvas.getContext("2d", { willReadFrequently: true });

        setupEventListeners();
        setupSliders();
        setupDragAndDrop();
    }

    function setupEventListeners() {
        if (elements.btnToggleLiveDetection) {
            elements.btnToggleLiveDetection.addEventListener("click", toggleLiveDetection);
        }
        if (elements.btnCloseCamera) {
            elements.btnCloseCamera.addEventListener("click", closeCamera);
        }
        if (elements.btnRetryCamera) {
            elements.btnRetryCamera.addEventListener("click", () => startCamera(true));
        }

        // Upload image from file on Live Webcam screen
        if (elements.btnUploadWebcamImage) {
            elements.btnUploadWebcamImage.addEventListener("click", () => {
                if (elements.webcamFileInput) elements.webcamFileInput.click();
            });
        }
        if (elements.webcamFileInput) {
            elements.webcamFileInput.addEventListener("change", handleWebcamFileSelect);
        }

        // Clear Image button
        if (elements.btnClearWebcamImage) {
            elements.btnClearWebcamImage.addEventListener("click", clearImageAndRestore);
        }

        // Resume Live Camera chip
        if (elements.btnResumeLiveCamera) {
            elements.btnResumeLiveCamera.addEventListener("click", resumeLiveCameraStream);
        }

        // Sample chips in Live Webcam
        if (elements.webcamSamplesList) {
            const chips = elements.webcamSamplesList.querySelectorAll(".sample-chip[data-sample]");
            chips.forEach((chip) => {
                chip.addEventListener("click", () => {
                    const sampleName = chip.dataset.sample;
                    loadSampleIntoLiveView(sampleName);
                });
            });
        }
    }

    function setupSliders() {
        if (elements.confSlider) {
            elements.confSlider.addEventListener("input", (e) => {
                const val = e.target.value;
                elements.confValue.textContent = `${val}%`;
                state.confThreshold = parseFloat(val) / 100.0;
                
                // If an uploaded image/sample is active, re-run detection with updated threshold
                if (state.activeImageSource) {
                    reRunActiveImageDetection();
                }
            });
        }

        if (elements.fpsSlider) {
            elements.fpsSlider.addEventListener("input", (e) => {
                const val = parseInt(e.target.value, 10);
                elements.fpsValue.textContent = `${val} FPS`;
                state.targetFps = val;
                
                if (state.isCameraActive && state.isLiveDetectionEnabled && !state.activeImageSource) {
                    restartInferenceLoop();
                }
            });
        }
    }

    // Drag and Drop on the live video stage
    function setupDragAndDrop() {
        const container = elements.videoContainer;
        if (!container) return;

        ["dragenter", "dragover"].forEach((eventName) => {
            container.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                container.classList.add("dragover");
            });
        });

        ["dragleave", "drop"].forEach((eventName) => {
            container.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                container.classList.remove("dragover");
            });
        });

        container.addEventListener("drop", (e) => {
            const files = e.dataTransfer.files;
            if (files && files.length > 0) {
                processWebcamUpload(files[0]);
            }
        });
    }

    // Tab Lifecycle hooks (called by app.js)
    function onTabEnter() {
        if (!state.isCameraActive && !state.activeImageSource) {
            startCamera(true);
        }
    }

    function onTabLeave() {
        if (state.isCameraActive) {
            closeCamera();
        }
    }

    // ==========================================
    // Camera Lifecycle (Fixed for Black Screen)
    // ==========================================
    async function startCamera(autoStartDetection = true) {
        if (state.isCameraActive) return;

        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            setOfflineState("Browser Incompatible", "Your browser does not support webcam capture (getUserMedia API).");
            App.showToast("Webcam capture not supported by browser.", "error");
            return;
        }

        elements.cameraStatusBadge.textContent = "Connecting Camera...";

        try {
            const constraints = {
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: "user",
                },
                audio: false,
            };

            const stream = await navigator.mediaDevices.getUserMedia(constraints);
            state.mediaStream = stream;
            state.isCameraActive = true;
            state.activeImageSource = null;

            const video = elements.livePreview;
            video.muted = true;
            video.defaultMuted = true;
            video.playsInline = true;
            video.autoplay = true;
            video.srcObject = stream;

            // Make video visible in DOM immediately so browser starts decoding frames
            elements.offlinePlaceholder.style.display = "none";
            elements.livePreview.style.display = "block";
            elements.annotatedResult.style.display = "none";
            if (elements.webcamSourceBadge) {
                elements.webcamSourceBadge.style.display = "inline-block";
                elements.webcamSourceBadge.textContent = "Live Camera";
            }
            if (elements.btnResumeLiveCamera) {
                elements.btnResumeLiveCamera.style.display = "none";
            }

            // Start playing video
            video.onloadedmetadata = async () => {
                try {
                    await video.play();
                } catch (playErr) {
                    console.warn("Video play promise error:", playErr);
                }

                const w = video.videoWidth || 640;
                const h = video.videoHeight || 480;

                offscreenCanvas.width = Math.min(w, 640);
                offscreenCanvas.height = Math.round(h * (offscreenCanvas.width / (w || 1)));

                elements.resolutionBadge.textContent = `${w} × ${h}`;
                elements.hudOverlay.style.display = "flex";
                elements.cameraStatusBadge.textContent = "Camera Live";

                startFpsCounter();

                if (autoStartDetection) {
                    state.isLiveDetectionEnabled = true;
                    updateToggleUI(true);
                    restartInferenceLoop();
                    App.showToast("Camera active. Automatic YOLO live detection started.", "success");
                } else {
                    state.isLiveDetectionEnabled = false;
                    updateToggleUI(false);
                    elements.cameraStatusBadge.textContent = "Live detection paused";
                }
            };

        } catch (err) {
            console.error("Camera access error:", err);
            let userMsg = err.message;
            if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
                userMsg = "Camera access was denied. Please allow camera permissions in your browser address bar.";
            } else if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {
                userMsg = "No webcam device was detected. You can upload an image or select a sample above to test.";
            } else if (err.name === "NotReadableError" || err.name === "TrackStartError") {
                userMsg = "Camera is currently used by another application. Close it and retry, or upload an image.";
            }
            setOfflineState("Camera Unavailable", userMsg);
            App.showToast(userMsg, "error", 6000);
        }
    }

    function closeCamera() {
        stopInferenceLoop();

        if (state.mediaStream) {
            state.mediaStream.getTracks().forEach((track) => track.stop());
            state.mediaStream = null;
        }

        state.isCameraActive = false;
        state.isLiveDetectionEnabled = false;
        state.isInFlight = false;
        state.activeImageSource = null;

        if (elements.livePreview) {
            elements.livePreview.pause();
            elements.livePreview.srcObject = null;
        }

        elements.offlinePlaceholder.style.display = "flex";
        elements.livePreview.style.display = "none";
        elements.annotatedResult.style.display = "none";
        elements.hudOverlay.style.display = "none";
        if (elements.webcamSourceBadge) elements.webcamSourceBadge.style.display = "none";
        if (elements.btnResumeLiveCamera) elements.btnResumeLiveCamera.style.display = "none";

        elements.cameraStatusBadge.textContent = "Camera Closed";
        elements.resolutionBadge.textContent = "-- × --";
        elements.fpsBadge.textContent = "0.0 FPS";

        updateToggleUI(false);
        stopFpsCounter();
        resetTelemetry();

        App.showToast("Camera released.", "info");
    }

    // ==========================================
    // Toggle Live Detection
    // ==========================================
    function toggleLiveDetection() {
        if (!state.isCameraActive && !state.activeImageSource) {
            startCamera(true);
            return;
        }

        state.isLiveDetectionEnabled = !state.isLiveDetectionEnabled;

        if (state.isLiveDetectionEnabled) {
            // Live detection ON
            updateToggleUI(true);
            elements.hudStatusText.textContent = "YOLO ACTIVE";
            elements.hudDot.className = "hud-dot live";
            elements.cameraStatusBadge.textContent = state.activeImageSource ? "Image Detection Active" : "Live Detection Active";

            if (!state.activeImageSource) {
                restartInferenceLoop();
            } else {
                reRunActiveImageDetection();
            }
            App.showToast("Live detection enabled.", "success");
        } else {
            // Live detection OFF: show raw camera preview
            stopInferenceLoop();
            state.isInFlight = false;
            updateToggleUI(false);

            // Hide annotated result overlay and show the raw live video!
            if (state.isCameraActive) {
                elements.annotatedResult.style.display = "none";
                elements.livePreview.style.display = "block";
            }

            elements.cameraStatusBadge.textContent = "Live detection paused";
            elements.hudStatusText.textContent = "DETECTION PAUSED";
            elements.hudDot.className = "hud-dot";
            App.showToast("Live detection paused. Unannotated live stream active.", "info");
        }
    }

    function updateToggleUI(isEnabled) {
        if (!elements.btnToggleLiveDetection) return;

        if (isEnabled) {
            elements.btnToggleLiveDetection.classList.add("active");
            if (elements.liveDetectionBtnText) elements.liveDetectionBtnText.textContent = "Live Detection: ON";
        } else {
            elements.btnToggleLiveDetection.classList.remove("active");
            if (elements.liveDetectionBtnText) elements.liveDetectionBtnText.textContent = "Live Detection: OFF";
        }
    }

    function restartInferenceLoop() {
        stopInferenceLoop();
        const intervalMs = Math.max(70, Math.round(1000 / state.targetFps));
        state.captureIntervalId = setInterval(captureAndDetectFrame, intervalMs);
    }

    function stopInferenceLoop() {
        if (state.captureIntervalId) {
            clearInterval(state.captureIntervalId);
            state.captureIntervalId = null;
        }
    }

    // ==========================================
    // Real-Time Frame Pipeline (No Black Freeze)
    // ==========================================
    async function captureAndDetectFrame() {
        if (!state.isCameraActive || !state.isLiveDetectionEnabled || state.isInFlight || state.activeImageSource) {
            return;
        }

        const video = elements.livePreview;
        if (!video || video.readyState < 2 || video.paused || video.ended) {
            return;
        }

        state.isInFlight = true;

        try {
            // Draw current real video frame to offscreen canvas
            offscreenCtx.drawImage(video, 0, 0, offscreenCanvas.width, offscreenCanvas.height);
            const base64Data = offscreenCanvas.toDataURL("image/jpeg", 0.72);

            const tStart = performance.now();
            const response = await fetch("/api/detect-frame", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    image: base64Data,
                    conf: state.confThreshold,
                    iou: state.iouThreshold,
                }),
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            const data = await response.json();
            const latency = Math.round(performance.now() - tStart);

            if (data.success && state.isLiveDetectionEnabled && state.isCameraActive && !state.activeImageSource) {
                // Update annotated frame overlay
                elements.annotatedResult.src = data.image;
                elements.annotatedResult.style.display = "block";

                state.frameCount++;
                state.totalFramesProcessed++;
                elements.liveTotalProcessed.textContent = state.totalFramesProcessed;
                elements.liveInferenceTime.textContent = `${data.inference_time_ms} ms`;
                elements.hudLatency.textContent = `Latency: ${data.inference_time_ms} ms`;
                elements.liveDetectionsCount.textContent = data.detection_count;

                updateLiveVerdict(data.verdict, data.class_counts, data.detection_count);
            }

        } catch (err) {
            console.warn("Webcam inference frame dropped:", err.message);
        } finally {
            state.isInFlight = false;
        }
    }

    // ==========================================
    // Upload & Test Sample in Live Webcam Tab
    // ==========================================
    function handleWebcamFileSelect(e) {
        const files = e.target.files;
        if (files && files.length > 0) {
            processWebcamUpload(files[0]);
        }
    }

    function processWebcamUpload(file) {
        const allowedTypes = ["image/jpeg", "image/jpg", "image/png", "image/webp"];
        if (!allowedTypes.includes(file.type.toLowerCase()) && !file.name.match(/\.(jpg|jpeg|png|webp)$/i)) {
            App.showToast("Unsupported file type. Please upload a JPG, PNG, or WEBP image.", "error");
            return;
        }

        // Pause live camera capture loop while showing uploaded image
        stopInferenceLoop();
        state.activeImageSource = { type: "file", file: file };

        const reader = new FileReader();
        reader.onload = async (e) => {
            const dataUrl = e.target.result;
            runDetectionOnDataUrl(dataUrl, file.name);
        };
        reader.readAsDataURL(file);
    }

    async function loadSampleIntoLiveView(sampleFileName) {
        stopInferenceLoop();
        state.activeImageSource = { type: "sample", id: sampleFileName };

        elements.cameraStatusBadge.textContent = `Testing Sample: ${sampleFileName}`;
        elements.hudStatusText.textContent = "SAMPLE INFERENCE";

        try {
            const res = await fetch("/api/detect-sample", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    sample_id: sampleFileName,
                    conf: state.confThreshold,
                    iou: state.iouThreshold,
                }),
            });

            const data = await res.json();
            if (!data.success) throw new Error(data.error || "Inference failed");

            // Display annotated image
            elements.offlinePlaceholder.style.display = "none";
            elements.livePreview.style.display = "none";
            elements.annotatedResult.src = data.image;
            elements.annotatedResult.style.display = "block";
            elements.hudOverlay.style.display = "flex";

            if (elements.webcamSourceBadge) {
                elements.webcamSourceBadge.style.display = "inline-block";
                elements.webcamSourceBadge.textContent = `Sample: ${sampleFileName}`;
            }
            if (elements.btnResumeLiveCamera && state.isCameraActive) {
                elements.btnResumeLiveCamera.style.display = "inline-flex";
            }

            elements.resolutionBadge.textContent = `${data.original_dims.width} × ${data.original_dims.height}`;
            elements.liveInferenceTime.textContent = `${data.inference_time_ms} ms`;
            elements.hudLatency.textContent = `Latency: ${data.inference_time_ms} ms`;
            elements.liveDetectionsCount.textContent = data.detection_count;

            updateLiveVerdict(data.verdict, data.class_counts, data.detection_count);
            App.showToast(`Loaded sample: ${sampleFileName} in Live View.`, "success");

        } catch (err) {
            App.showToast(`Failed to test sample: ${err.message}`, "error");
        }
    }

    async function runDetectionOnDataUrl(dataUrl, filename) {
        elements.cameraStatusBadge.textContent = `Testing Upload: ${filename}`;
        elements.hudStatusText.textContent = "IMAGE INFERENCE";

        try {
            const res = await fetch("/api/detect-frame", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    image: dataUrl,
                    conf: state.confThreshold,
                    iou: state.iouThreshold,
                }),
            });

            const data = await res.json();
            if (!data.success) throw new Error(data.error || "Inference failed");

            elements.offlinePlaceholder.style.display = "none";
            elements.livePreview.style.display = "none";
            elements.annotatedResult.src = data.image;
            elements.annotatedResult.style.display = "block";
            elements.hudOverlay.style.display = "flex";

            if (elements.webcamSourceBadge) {
                elements.webcamSourceBadge.style.display = "inline-block";
                elements.webcamSourceBadge.textContent = `Uploaded: ${filename}`;
            }
            if (elements.btnResumeLiveCamera && state.isCameraActive) {
                elements.btnResumeLiveCamera.style.display = "inline-flex";
            }

            elements.liveInferenceTime.textContent = `${data.inference_time_ms} ms`;
            elements.hudLatency.textContent = `Latency: ${data.inference_time_ms} ms`;
            elements.liveDetectionsCount.textContent = data.detection_count;

            updateLiveVerdict(data.verdict, data.class_counts, data.detection_count);
            App.showToast(`Uploaded ${filename} detected in Live View.`, "success");

        } catch (err) {
            App.showToast(`Failed to detect image: ${err.message}`, "error");
        }
    }

    function reRunActiveImageDetection() {
        if (!state.activeImageSource) return;
        if (state.activeImageSource.type === "sample") {
            loadSampleIntoLiveView(state.activeImageSource.id);
        } else if (state.activeImageSource.type === "file") {
            processWebcamUpload(state.activeImageSource.file);
        }
    }

    // Clear Image and restore Camera feed
    function clearImageAndRestore() {
        state.activeImageSource = null;
        if (elements.webcamFileInput) elements.webcamFileInput.value = "";
        if (elements.webcamSourceBadge) elements.webcamSourceBadge.style.display = "none";
        if (elements.btnResumeLiveCamera) elements.btnResumeLiveCamera.style.display = "none";

        resetTelemetry();

        if (state.isCameraActive) {
            // Restore smooth camera playback
            elements.offlinePlaceholder.style.display = "none";
            elements.livePreview.style.display = "block";
            elements.annotatedResult.style.display = "none";
            elements.cameraStatusBadge.textContent = "Camera Live";
            if (elements.webcamSourceBadge) {
                elements.webcamSourceBadge.style.display = "inline-block";
                elements.webcamSourceBadge.textContent = "Live Camera";
            }

            if (state.isLiveDetectionEnabled) {
                elements.hudStatusText.textContent = "YOLO ACTIVE";
                elements.hudDot.className = "hud-dot live";
                restartInferenceLoop();
            } else {
                elements.hudStatusText.textContent = "DETECTION PAUSED";
                elements.hudDot.className = "hud-dot";
            }
            App.showToast("Cleared image. Restored live camera stream.", "info");
        } else {
            // Camera not active: restore clean offline placeholder
            elements.offlinePlaceholder.style.display = "flex";
            elements.livePreview.style.display = "none";
            elements.annotatedResult.style.display = "none";
            elements.hudOverlay.style.display = "none";
            elements.cameraStatusBadge.textContent = "Camera Ready";
            App.showToast("Cleared view.", "info");
        }
    }

    function resumeLiveCameraStream() {
        clearImageAndRestore();
    }

    function updateLiveVerdict(verdict, classCounts, count) {
        if (!verdict) return;

        elements.liveVerdictCard.className = `live-verdict-card verdict-${verdict.status || 'idle'}`;
        elements.liveVerdictTag.textContent = (verdict.status || "DETECTING").toUpperCase();
        elements.liveVerdictLabel.textContent = verdict.label || `${count} Objects`;
        elements.liveVerdictDesc.textContent = verdict.summary || `Running at ${state.confThreshold * 100}% confidence.`;
        elements.hudVerdictPill.textContent = (verdict.status || "ACTIVE").toUpperCase();

        renderClassTally(classCounts);
    }

    function renderClassTally(classCounts) {
        if (!classCounts || Object.keys(classCounts).length === 0) {
            elements.classesTallyList.innerHTML = `<div class="class-tally-empty">No objects currently detected in view</div>`;
            return;
        }

        let html = "";
        for (const [clsName, tally] of Object.entries(classCounts)) {
            html += `
                <div class="class-tally-item">
                    <span>${clsName}</span>
                    <span class="class-tally-count">${tally}</span>
                </div>
            `;
        }
        elements.classesTallyList.innerHTML = html;
    }

    function startFpsCounter() {
        state.frameCount = 0;
        state.fpsIntervalId = setInterval(() => {
            state.currentFps = (state.frameCount / 1.0).toFixed(1);
            elements.fpsBadge.textContent = `${state.currentFps} FPS`;
            elements.liveEffectiveFps.textContent = state.currentFps;
            state.frameCount = 0;
        }, 1000);
    }

    function stopFpsCounter() {
        if (state.fpsIntervalId) {
            clearInterval(state.fpsIntervalId);
            state.fpsIntervalId = null;
        }
    }

    function setOfflineState(heading, msg) {
        elements.offlinePlaceholder.style.display = "flex";
        elements.livePreview.style.display = "none";
        elements.annotatedResult.style.display = "none";
        elements.hudOverlay.style.display = "none";
        if (elements.offlineHeading) elements.offlineHeading.textContent = heading;
        if (elements.offlineMsg) elements.offlineMsg.textContent = msg;
        elements.cameraStatusBadge.textContent = heading;
    }

    function resetTelemetry() {
        elements.liveDetectionsCount.textContent = "0";
        elements.liveInferenceTime.textContent = "-- ms";
        elements.liveEffectiveFps.textContent = "0.0";
        elements.classesTallyList.innerHTML = `<div class="class-tally-empty">No objects currently detected in view</div>`;
        elements.liveVerdictCard.className = "live-verdict-card";
        elements.liveVerdictTag.textContent = "IDLE";
        elements.liveVerdictLabel.textContent = "Waiting for Feed";
        elements.liveVerdictDesc.textContent = "Turn on camera or upload an image to analyze real-time weld seam quality.";
    }

    return {
        init,
        startCamera,
        closeCamera,
        toggleLiveDetection,
        clearImageAndRestore,
        onTabEnter,
        onTabLeave,
        isStreaming: () => state.isCameraActive,
    };
})();

// Initialize on DOM load
document.addEventListener("DOMContentLoaded", () => {
    window.WebcamModule = WebcamModule;
    WebcamModule.init();
});
