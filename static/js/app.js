/**
 * YOLO Vision Studio - Main Application Controller
 * Handles tabs, notifications, backend health & telemetry.
 */

const App = (() => {
    // State
    const state = {
        activeTab: "imageMode",
        backendOnline: false,
        modelInfo: null,
    };

    // DOM Elements
    const elements = {
        tabImageBtn: document.getElementById("tabImageBtn"),
        tabWebcamBtn: document.getElementById("tabWebcamBtn"),
        imageMode: document.getElementById("imageMode"),
        webcamMode: document.getElementById("webcamMode"),
        backendStatusPill: document.getElementById("backendStatusPill"),
        backendStatusText: document.getElementById("backendStatusText"),
        activeModelName: document.getElementById("activeModelName"),
        inferenceDevice: document.getElementById("inferenceDevice"),
        toastContainer: document.getElementById("toastContainer"),
        footerUptime: document.getElementById("footerUptime"),
        footerModel: document.getElementById("footerModel"),
    };

    // Initialize Application
    function init() {
        setupTabs();
        checkBackendHealth();
        startStatsPolling();
    }

    // Tab Navigation setup
    function setupTabs() {
        elements.tabImageBtn.addEventListener("click", () => switchTab("imageMode"));
        elements.tabWebcamBtn.addEventListener("click", () => switchTab("webcamMode"));
    }

    function switchTab(targetMode) {
        if (state.activeTab === targetMode) return;

        // If leaving webcam mode, notify webcam module to release camera
        if (state.activeTab === "webcamMode" && window.WebcamModule) {
            window.WebcamModule.onTabLeave();
        }

        // Update Nav buttons
        elements.tabImageBtn.classList.toggle("active", targetMode === "imageMode");
        elements.tabWebcamBtn.classList.toggle("active", targetMode === "webcamMode");

        // Update Panel visibility
        elements.imageMode.classList.toggle("active", targetMode === "imageMode");
        elements.webcamMode.classList.toggle("active", targetMode === "webcamMode");

        state.activeTab = targetMode;

        // If entering webcam mode, automatically start camera & detection
        if (targetMode === "webcamMode" && window.WebcamModule) {
            window.WebcamModule.onTabEnter();
        }
    }

    // Toast Notification System
    function showToast(message, type = "info", duration = 4000) {
        const toast = document.createElement("div");
        toast.className = `toast toast-${type}`;

        let iconSvg = "";
        if (type === "error") {
            iconSvg = `<svg class="toast-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>`;
        } else if (type === "success") {
            iconSvg = `<svg class="toast-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>`;
        } else {
            iconSvg = `<svg class="toast-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>`;
        }

        toast.innerHTML = `${iconSvg}<div class="toast-message">${message}</div>`;
        elements.toastContainer.appendChild(toast);

        setTimeout(() => {
            toast.style.opacity = "0";
            toast.style.transform = "translateY(10px)";
            toast.style.transition = "all 0.3s ease";
            setTimeout(() => toast.remove(), 300);
        }, duration);
    }

    // Backend Health Check
    async function checkBackendHealth() {
        try {
            const res = await fetch("/api/health");
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const data = await res.json();

            state.backendOnline = true;
            elements.backendStatusText.textContent = "Backend Online";
            elements.backendStatusPill.className = "status-pill status-pill-online";

            if (data.model_name) {
                elements.activeModelName.textContent = data.model_name;
                if (elements.footerModel) elements.footerModel.textContent = data.model_name;
            }
            if (data.device) {
                elements.inferenceDevice.textContent = data.device;
            }

            if (!data.model_loaded) {
                showToast(`Warning: YOLO model failed to load: ${data.error}`, "error", 6000);
            }
        } catch (err) {
            state.backendOnline = false;
            elements.backendStatusText.textContent = "Backend Offline";
            elements.backendStatusPill.className = "status-pill status-pill-offline";
            showToast("Cannot reach YOLO Flask backend. Is the server running?", "error");
        }
    }

    // Telemetry Polling (every 5 seconds)
    function startStatsPolling() {
        setInterval(async () => {
            try {
                const res = await fetch("/api/stats");
                if (res.ok) {
                    const stats = await res.json();
                    if (elements.footerUptime && stats.uptime_formatted) {
                        elements.footerUptime.textContent = `Uptime: ${stats.uptime_formatted}`;
                    }
                }
            } catch (e) {
                // Ignore transient network errors
            }
        }, 5000);
    }

    return {
        init,
        showToast,
        getState: () => state,
    };
})();

// Initialize on DOM load
document.addEventListener("DOMContentLoaded", () => {
    App.init();
});
