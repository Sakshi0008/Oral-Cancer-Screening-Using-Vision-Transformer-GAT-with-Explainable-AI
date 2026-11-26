/**
 * ORALSCAN AI - Primary Frontend Application Logic
 * ===============================================
 * Production SPA controller managing REST API communication, authentication state,
 * image uploads, multimodal XAI rendering, and PDF report downloads.
 */

// Application State
const state = {
    currentUser: null,
    token: localStorage.getItem("oralscan_token") || null,
    selectedFile: null,
    currentScreeningId: null,
    currentScreeningData: null,
    authMode: "login" // 'login' or 'register'
};

// API Base Configuration
const API_BASE = "";

// =============================================================================
// Lifecycle & Initialization
// =============================================================================

document.addEventListener("DOMContentLoaded", () => {
    initAuth();
    handleHashNavigation();
    initDropzone();
});

window.addEventListener("hashchange", handleHashNavigation);

function handleHashNavigation() {
    const hash = window.location.hash.replace("#", "") || "landing";
    navigateTo(hash);
}

function navigateTo(viewId) {
    // Hide all view sections
    document.querySelectorAll(".view-section").forEach(sec => sec.classList.remove("active"));
    document.querySelectorAll(".nav-item").forEach(item => item.classList.remove("active"));

    // Activate selected view
    const targetView = document.getElementById(`view-${viewId}`) || document.getElementById("view-landing");
    targetView.classList.add("active");

    const targetNav = document.getElementById(`nav-${viewId}`);
    if (targetNav) targetNav.classList.add("active");

    window.location.hash = viewId;

    // View-specific data fetching
    if (viewId === "dashboard" && state.currentUser) {
        fetchDashboardStats();
    } else if (viewId === "history" && state.currentUser) {
        fetchScreeningHistory();
    }
}

// =============================================================================
// Authentication Management
// =============================================================================

async function initAuth() {
    if (!state.token) {
        updateAuthUI(null);
        return;
    }

    try {
        const res = await fetch(`${API_BASE}/api/auth/me`, {
            headers: { "Authorization": `Bearer ${state.token}` }
        });
        if (res.ok) {
            const data = await res.json();
            state.currentUser = data.user;
            updateAuthUI(data.user);
            if (data.stats) updateDashboardStatsUI(data.stats);
        } else {
            handleLogout();
        }
    } catch (e) {
        console.error("Auth check failed:", e);
        handleLogout();
    }
}

function updateAuthUI(user) {
    const loggedOutDiv = document.getElementById("auth-logged-out");
    const loggedInDiv = document.getElementById("auth-logged-in");
    const navUsername = document.getElementById("nav-username");

    if (user) {
        loggedOutDiv.classList.add("hidden");
        loggedInDiv.classList.remove("hidden");
        navUsername.textContent = user.username;
        document.querySelectorAll(".auth-required").forEach(el => el.classList.remove("hidden"));
    } else {
        loggedOutDiv.classList.remove("hidden");
        loggedInDiv.classList.add("hidden");
        document.querySelectorAll(".auth-required").forEach(el => el.classList.add("hidden"));
    }
}

function openAuthModal(mode = "login") {
    state.authMode = mode;
    switchAuthTab(mode);
    document.getElementById("auth-modal").classList.remove("hidden");
    document.getElementById("auth-error").classList.add("hidden");
}

function closeAuthModal() {
    document.getElementById("auth-modal").classList.add("hidden");
}

function switchAuthTab(mode) {
    state.authMode = mode;
    const tabLogin = document.getElementById("tab-login");
    const tabReg = document.getElementById("tab-register");
    const groupEmail = document.getElementById("group-email");
    const submitBtn = document.getElementById("auth-submit-btn");
    const modalTitle = document.getElementById("auth-modal-title");
    const errorBox = document.getElementById("auth-error");
    errorBox.classList.add("hidden");

    if (mode === "login") {
        tabLogin.classList.add("active");
        tabReg.classList.remove("active");
        groupEmail.classList.add("hidden");
        submitBtn.textContent = "Sign In";
        modalTitle.textContent = "Sign In to ORALSCAN AI";
    } else {
        tabReg.classList.add("active");
        tabLogin.classList.remove("active");
        groupEmail.classList.remove("hidden");
        submitBtn.textContent = "Create Account";
        modalTitle.textContent = "Register for ORALSCAN AI";
    }
}

async function handleAuthSubmit(e) {
    e.preventDefault();
    const errorBox = document.getElementById("auth-error");
    errorBox.classList.add("hidden");

    const username = document.getElementById("auth-username").value.trim();
    const password = document.getElementById("auth-password").value;
    const email = document.getElementById("auth-email").value.trim();

    try {
        let endpoint = `${API_BASE}/api/auth/login`;
        let payload = { username_or_email: username, password: password };

        if (state.authMode === "register") {
            endpoint = `${API_BASE}/api/auth/register`;
            payload = { username: username, email: email, password: password };
        }

        const res = await fetch(endpoint, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        const data = await res.json();
        if (!res.ok) {
            throw new Error(data.detail || "Authentication request failed.");
        }

        state.token = data.token;
        state.currentUser = data.user;
        localStorage.setItem("oralscan_token", data.token);
        updateAuthUI(data.user);
        closeAuthModal();

        // Redirect to screening portal
        navigateTo("screen");
    } catch (err) {
        errorBox.textContent = err.message;
        errorBox.classList.remove("hidden");
    }
}

function handleLogout() {
    if (state.token) {
        fetch(`${API_BASE}/api/auth/logout`, {
            method: "POST",
            headers: { "Authorization": `Bearer ${state.token}` }
        }).catch(() => {});
    }
    state.token = null;
    state.currentUser = null;
    localStorage.removeItem("oralscan_token");
    updateAuthUI(null);
    navigateTo("landing");
}

// =============================================================================
// Image Upload & Dropzone Management
// =============================================================================

function initDropzone() {
    const dropzone = document.getElementById("dropzone");
    if (!dropzone) return;

    ["dragenter", "dragover"].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add("dragover");
        }, false);
    });

    ["dragleave", "drop"].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove("dragover");
        }, false);
    });

    dropzone.addEventListener("drop", (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            setUploadedFile(files[0]);
        }
    });
}

function handleFileSelected(e) {
    if (e.target.files && e.target.files[0]) {
        setUploadedFile(e.target.files[0]);
    }
}

function setUploadedFile(file) {
    if (!file.type.match("image/(jpeg|png)")) {
        alert("Unsupported file format. Please upload a JPEG or PNG image.");
        return;
    }
    if (file.size > 10 * 1024 * 1024) {
        alert("File size exceeds 10MB limit.");
        return;
    }

    state.selectedFile = file;

    const reader = new FileReader();
    reader.onload = (e) => {
        document.getElementById("preview-img").src = e.target.result;
        document.getElementById("dropzone-empty").classList.add("hidden");
        document.getElementById("dropzone-preview").classList.remove("hidden");
        document.getElementById("btn-screen").disabled = false;
    };
    reader.readAsDataURL(file);
}

function resetUpload() {
    state.selectedFile = null;
    document.getElementById("file-input").value = "";
    document.getElementById("preview-img").src = "";
    document.getElementById("dropzone-empty").classList.remove("hidden");
    document.getElementById("dropzone-preview").classList.add("hidden");
    document.getElementById("btn-screen").disabled = true;
}

async function loadSampleImage(imagePath) {
    try {
        // Fetch validation sample and convert to File object
        const res = await fetch(`/${imagePath}`);
        if (!res.ok) throw new Error("Could not load sample image.");
        const blob = await res.blob();
        const file = new File([blob], imagePath.split("/").pop(), { type: "image/jpeg" });
        setUploadedFile(file);
    } catch (e) {
        alert("Sample image could not be loaded directly. You can browse and upload an image from your device.");
    }
}

// =============================================================================
// Screening Execution & XAI Rendering
// =============================================================================

async function executeScreening() {
    if (!state.selectedFile) return;

    // Check authentication
    if (!state.currentUser) {
        openAuthModal("login");
        return;
    }

    const loader = document.getElementById("screening-loader");
    const btnScreen = document.getElementById("btn-screen");
    const progressBar = document.getElementById("progress-bar");
    const loaderStatus = document.getElementById("loader-status");

    loader.classList.remove("hidden");
    btnScreen.disabled = true;

    // Simulated progress pipeline stages
    const stages = [
        { pct: 15, text: "Decoding & Normalizing RGB Image..." },
        { pct: 35, text: "Extracting Vision Transformer Patches (196 nodes)..." },
        { pct: 55, text: "Generating Hybrid Graph Topology (Spatial + Top-8 Cosine)..." },
        { pct: 75, text: "GATv2 Relational Message Passing & Concatenation Fusion..." },
        { pct: 90, text: "Computing Integrated Gradients & Saliency Maps..." }
    ];

    let stageIdx = 0;
    const progressInterval = setInterval(() => {
        if (stageIdx < stages.length) {
            progressBar.style.width = `${stages[stageIdx].pct}%`;
            loaderStatus.textContent = stages[stageIdx].text;
            stageIdx++;
        }
    }, 400);

    const runPerturbation = document.getElementById("perturbation-toggle").checked;
    const formData = new FormData();
    formData.append("file", state.selectedFile);

    try {
        const res = await fetch(`${API_BASE}/api/screen?run_perturbation=${runPerturbation}`, {
            method: "POST",
            headers: {
                "Authorization": `Bearer ${state.token}`
            },
            body: formData
        });

        clearInterval(progressInterval);
        progressBar.style.width = "100%";
        loaderStatus.textContent = "Analysis Complete! Preparing Diagnostic Overlays...";

        const data = await res.json();
        if (!res.ok) {
            throw new Error(data.detail || "Screening inference failed.");
        }

        state.currentScreeningId = data.screening_id;
        state.currentScreeningData = data;

        // Render Results Page
        setTimeout(() => {
            loader.classList.add("hidden");
            btnScreen.disabled = false;
            renderScreeningResults(data);
            navigateTo("result");
        }, 500);

    } catch (err) {
        clearInterval(progressInterval);
        loader.classList.add("hidden");
        btnScreen.disabled = false;
        alert(`Screening Failed: ${err.message}`);
    }
}

function renderScreeningResults(data) {
    const banner = document.getElementById("finding-banner");
    const badge = document.getElementById("finding-badge");
    const title = document.getElementById("finding-title");
    const desc = document.getElementById("finding-desc");
    const scoreVal = document.getElementById("finding-score");

    const isSuspicious = data.pred_label === 1;

    banner.className = `card finding-banner ${isSuspicious ? "suspicious" : "benign"}`;
    badge.className = `badge ${isSuspicious ? "badge-warning" : "badge-accent"}`;
    badge.textContent = isSuspicious ? "SUSPICIOUS PATTERN" : "NO SUSPICIOUS PATTERN";
    title.textContent = data.cautious_result;
    desc.textContent = data.clinical_guidance;
    scoreVal.textContent = `${data.cancer_score.toFixed(1)}%`;
    scoreVal.className = `score-value ${isSuspicious ? "text-danger" : "text-success"}`;

    // Load Visual Overlays
    const sid = data.screening_id;
    document.getElementById("viz-original").src = `${API_BASE}/api/screen/${sid}/image?auth=${state.token}`;
    document.getElementById("viz-rollout").src = `${API_BASE}/api/screen/${sid}/visual/rollout?auth=${state.token}`;
    document.getElementById("viz-graph").src = `${API_BASE}/api/screen/${sid}/visual/graph?auth=${state.token}`;

    // Branch Breakdown
    const fusion = data.fusion_attribution;
    const vitShare = fusion.vit_share_pct;
    const gatShare = fusion.gat_share_pct;

    document.getElementById("branch-vit").style.width = `${vitShare}%`;
    document.getElementById("branch-vit-text").textContent = `ViT: ${vitShare.toFixed(1)}%`;
    document.getElementById("branch-gat").style.width = `${gatShare}%`;
    document.getElementById("branch-gat-text").textContent = `GAT: ${gatShare.toFixed(1)}%`;

    document.getElementById("branch-vit-val").textContent = `${vitShare.toFixed(1)}%`;
    document.getElementById("branch-gat-val").textContent = `${gatShare.toFixed(1)}%`;
    document.getElementById("convergence-note").textContent = 
        `Completeness Delta: |sum(IG) - Delta_f| = ${fusion.ig_absolute_error.toFixed(4)} logit units (Relative Error: ${fusion.ig_relative_error_pct.toFixed(1)}%) | Reference Baseline: [0.05]^384`;

    // Narrative Body
    document.getElementById("narrative-body").textContent = data.narrative_explanation;
}

// =============================================================================
// PDF Download
// =============================================================================

function downloadCurrentPDF() {
    if (!state.currentScreeningId) return;
    downloadScreeningPDF(state.currentScreeningId);
}

function downloadScreeningPDF(screeningId) {
    if (!state.token) {
        alert("Please sign in to download screening reports.");
        return;
    }
    const url = `${API_BASE}/api/screen/${screeningId}/pdf`;
    
    // Fetch with authentication token and trigger download
    fetch(url, {
        headers: { "Authorization": `Bearer ${state.token}` }
    })
    .then(response => {
        if (!response.ok) throw new Error("PDF generation failed or access denied.");
        return response.blob();
    })
    .then(blob => {
        const blobUrl = window.URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = blobUrl;
        a.download = `oralscan_report_${screeningId.substring(0, 8)}.pdf`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        window.URL.revokeObjectURL(blobUrl);
    })
    .catch(err => {
        alert(`Error downloading report: ${err.message}`);
    });
}

// =============================================================================
// Dashboard & History Management
// =============================================================================

async function fetchDashboardStats() {
    if (!state.token) return;
    try {
        const res = await fetch(`${API_BASE}/api/auth/me`, {
            headers: { "Authorization": `Bearer ${state.token}` }
        });
        if (res.ok) {
            const data = await res.json();
            if (data.stats) updateDashboardStatsUI(data.stats);
        }
    } catch (e) {
        console.error("Failed to load dashboard stats:", e);
    }
}

function updateDashboardStatsUI(stats) {
    document.getElementById("dash-total").textContent = stats.total_screenings;
    document.getElementById("dash-suspicious").textContent = stats.suspicious_count;
    document.getElementById("dash-benign").textContent = stats.benign_count;
}

async function fetchScreeningHistory() {
    if (!state.token) return;
    const tbody = document.getElementById("history-table-body");
    tbody.innerHTML = `<tr><td colspan="6" class="text-center py-4">Loading screening records...</td></tr>`;

    try {
        const res = await fetch(`${API_BASE}/api/history`, {
            headers: { "Authorization": `Bearer ${state.token}` }
        });
        if (!res.ok) throw new Error("Failed to load history.");
        const records = await res.json();

        if (records.length === 0) {
            tbody.innerHTML = `<tr><td colspan="6" class="text-center py-4">No screening records found.</td></tr>`;
            return;
        }

        tbody.innerHTML = records.map(r => `
            <tr>
                <td>${r.created_at || "Recent"}</td>
                <td><strong>${escapeHtml(r.original_filename)}</strong></td>
                <td>${r.cancer_score.toFixed(1)}%</td>
                <td>
                    <span class="badge ${r.pred_label === 1 ? 'badge-warning' : 'badge-accent'}">
                        ${r.pred_label === 1 ? 'Suspicious' : 'No Suspicion'}
                    </span>
                </td>
                <td>ViT: ${r.vit_share_pct.toFixed(0)}% / GAT: ${r.gat_share_pct.toFixed(0)}%</td>
                <td>
                    <button class="btn btn-sm btn-outline" onclick="downloadScreeningPDF('${r.id}')">PDF</button>
                    <button class="btn btn-sm btn-secondary" onclick="deleteHistoryRecord('${r.id}')">Delete</button>
                </td>
            </tr>
        `).join("");
    } catch (e) {
        tbody.innerHTML = `<tr><td colspan="6" class="text-center py-4 text-danger">Error loading history: ${e.message}</td></tr>`;
    }
}

async function deleteHistoryRecord(screeningId) {
    if (!confirm("Are you sure you want to permanently delete this screening record?")) return;

    try {
        const res = await fetch(`${API_BASE}/api/history/${screeningId}`, {
            method: "DELETE",
            headers: { "Authorization": `Bearer ${state.token}` }
        });
        if (!res.ok) throw new Error("Failed to delete record.");
        fetchScreeningHistory();
        fetchDashboardStats();
    } catch (e) {
        alert(e.message);
    }
}

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}
