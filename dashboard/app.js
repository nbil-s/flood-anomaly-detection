import { initializeApp } from
    "https://www.gstatic.com/firebasejs/11.10.0/firebase-app.js";

import {
    sendSignInLinkToEmail,
    isSignInWithEmailLink,
    signInWithEmailLink,
    getAuth,
    GoogleAuthProvider,
    signInWithPopup,
    onAuthStateChanged,
    signOut
} from "https://www.gstatic.com/firebasejs/11.10.0/firebase-auth.js";

import { firebaseConfig } from "./firebase-config.js";

const firebaseApp = initializeApp(firebaseConfig);
const auth = getAuth(firebaseApp);
const googleProvider = new GoogleAuthProvider();

/* =========================================================
   FLOODWATCH — DASHBOARD APPLICATION
   ========================================================= */

const API_URL = "http://127.0.0.1:8000/dashboard-data";
const REFRESH_INTERVAL = 300000;

const DEMO_MODE = true;

let waterChart = null;
let environmentChart = null;
let refreshTimer = null;
let isLoading = false;

let latestDashboardData = null;

/* ---------- DOM HELPERS ---------- */

function element(id) {
    return document.getElementById(id);
}

function setText(id, value) {
    const node = element(id);

    if (node) {
        node.textContent = value ?? "--";
    }
}

function showElement(id, show = true) {
    const node = element(id);

    if (node) {
        node.classList.toggle("hidden", !show);
    }
}

function formatNumber(value, decimals = 2) {
    if (value === null || value === undefined || value === "") {
        return "--";
    }

    const number = Number(value);

    if (!Number.isFinite(number)) {
        return "--";
    }

    return number.toLocaleString(undefined, {
        maximumFractionDigits: decimals
    });
}

function formatTimestamp(value) {
    if (!value) {
        return "Unknown time";
    }

    const date = new Date(value);

    if (Number.isNaN(date.getTime())) {
        return String(value);
    }

    return date.toLocaleString(undefined, {
        year: "numeric",
        month: "short",
        day: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit"
    });
}

function getTimestamp(reading) {
    return (
        reading?.timestamp ??
        reading?.created_at ??
        reading?.createdAt ??
        reading?.time ??
        null
    );
}

function getReadingValue(reading, key) {
    if (!reading) {
        return null;
    }

    const aliases = {
        water_level: ["water_level", "waterLevel", "waterlevel"],
        rainfall: ["rainfall", "rain", "rainfall_level"],
        temperature: ["temperature", "temp"],
        humidity: ["humidity", "relative_humidity"]
    };

    for (const candidate of aliases[key] ?? [key]) {
        if (
            reading[candidate] !== undefined &&
            reading[candidate] !== null
        ) {
            return reading[candidate];
        }
    }

    return null;
}

async function handleGoogleSignIn() {
    const loginMessage = element("loginMessage");
    const loginButton = element("googleSignInButton");

    try {
        if (loginButton) loginButton.disabled = true;
        if (loginMessage) loginMessage.textContent = "Signing in...";

        await signInWithPopup(auth, googleProvider);

    } catch (error) {
        console.error("Google sign-in error:", error);

        if (loginMessage) {
            loginMessage.textContent =
                error.message || "Google sign-in failed.";
        }

    } finally {
        if (loginButton) loginButton.disabled = false;
    }
}

async function handleSignOut() {
    try {
        await signOut(auth);
    } catch (error) {
        console.error("Sign-out error:", error);
    }
}

function handleAuthentication(user) {
    const isSignedIn = Boolean(user);

    showElement("loginView", !isSignedIn);
    showElement("dashboardView", isSignedIn);

    if (!isSignedIn) {
        updateViewerProfile(null);

        if (refreshTimer) {
            clearInterval(refreshTimer);
            refreshTimer = null;
        }

        return;
    }

    updateViewerProfile(user);

    initializeTheme();
    initCharts();
    initializeControls();

    loadBackendDashboard();

    if (refreshTimer) {
        clearInterval(refreshTimer);
    }

    refreshTimer = setInterval(() => {
        loadBackendDashboard();
    }, REFRESH_INTERVAL);
}


/* ---------- CONNECTION STATUS ---------- */

function setConnection(status, message) {
    const dot = element("connectionDot");

    setText("connectionStatus", message);

    if (!dot) {
        return;
    }

    dot.classList.remove("connecting", "disconnected");

    if (status === "connected") {
        dot.style.background = "";
    } else if (status === "connecting") {
        dot.classList.add("connecting");
    } else {
        dot.classList.add("disconnected");
    }
}

function initializeAuthenticationControls() {
    const loginButton = element("googleSignInButton");
    const logoutButton = element("signOutButton");

    if (loginButton) {
        loginButton.addEventListener("click", handleGoogleSignIn);
    }

    if (logoutButton) {
        logoutButton.addEventListener("click", handleSignOut);
    }
}

const emailActionCodeSettings = {
    url: `${window.location.origin}${window.location.pathname}`,
    handleCodeInApp: true
};

// Send the passwordless sign-in link.
async function handleEmailSignIn() {
    const emailInput = document.getElementById("emailInput");
    const button = document.getElementById("emailSignInButton");
    const message = document.getElementById("emailLoginMessage");

    const email = emailInput.value.trim();

    if (!email) {
        message.textContent = "Enter your email address.";
        return;
    }

    if (!emailInput.checkValidity()) {
        message.textContent = "Enter a valid email address.";
        return;
    }

    try {
        button.disabled = true;
        message.textContent = "Sending sign-in link...";

        await sendSignInLinkToEmail(
            auth,
            email,
            emailActionCodeSettings
        );

        // Store the email locally for same-device completion.
        window.localStorage.setItem("emailForSignIn", email);

        message.textContent =
            `A sign-in link has been sent to ${email}. Check your inbox and spam folder.`;

    } catch (error) {
        console.error("Email sign-in error:", error);

        message.textContent =
            "Could not send the sign-in link. " + error.message;

    } finally {
        button.disabled = false;
    }
}


// Complete sign-in after the user clicks the email link.
async function completeEmailSignIn() {
    const emailInput = document.getElementById("emailConfirmInput");
    const message = document.getElementById("emailLoginMessage");
    const button = document.getElementById("completeEmailSignInButton");

    const email = emailInput.value.trim();

    if (!email || !emailInput.checkValidity()) {
        message.textContent = "Enter the email address that received the link.";
        return;
    }

    try {
        button.disabled = true;
        message.textContent = "Verifying your email link...";

        const result = await signInWithEmailLink(
            auth,
            email,
            window.location.href
        );

        window.localStorage.removeItem("emailForSignIn");

        // Remove the authentication code from the browser URL.
        window.history.replaceState(
            {},
            document.title,
            window.location.pathname
        );

        console.log("Email sign-in successful:", result.user.email);

        message.textContent = "Email verified. Signing you in...";

    } catch (error) {
        console.error("Email link completion error:", error);

        message.textContent =
            "Sign-in failed. The link may have expired or the email may not match.";

        button.disabled = false;
    }
}


// Check whether the current page URL contains an email sign-in link.
async function checkForEmailSignInLink() {
    if (!isSignInWithEmailLink(auth, window.location.href)) {
        return;
    }

    const savedEmail = window.localStorage.getItem("emailForSignIn");

    const confirmSection = document.getElementById("emailConfirmSection");
    const confirmInput = document.getElementById("emailConfirmInput");
    const message = document.getElementById("emailLoginMessage");

    if (savedEmail) {
        confirmInput.value = savedEmail;
        await completeEmailSignIn();
    } else {
        // Link opened on a different device or browser.
        confirmSection.classList.remove("hidden");
        message.textContent =
            "Enter the email address that received this sign-in link.";
    }
}


/* ---------- VIEWER PROFILE ---------- */

function updateViewerProfile(user = null) {
    const nameNode = element("profileName");
    const emailNode = element("profileEmail");
    const initialsNode = element("profileInitials");
    const photoNode = element("profilePhoto");

    if (!user) {
        setText("profileName", "Demo Viewer");
        setText("profileEmail", "Local demonstration");
        setText(
            "signedInUser",
            "Local demonstration — FastAPI backend data"
        );

        if (initialsNode) {
            initialsNode.textContent = "DV";
            initialsNode.classList.remove("hidden");
        }

        if (photoNode) {
            photoNode.classList.add("hidden");
            photoNode.removeAttribute("src");
        }

        return;
    }

    const name = user.displayName || user.name || "Dashboard Viewer";
    const email = user.email || "Authenticated user";

    if (nameNode) {
        nameNode.textContent = name;
    }

    if (emailNode) {
        emailNode.textContent = email;
    }

    setText("signedInUser", `Signed in as ${name}`);

 const initials = name
    .trim()
    .split(/\s+/)
    .slice(0, 2)
    .map(part => part[0])
    .join("")
    .toUpperCase();

    if (initialsNode) {
        initialsNode.textContent = initials || "DV";
        initialsNode.classList.remove("hidden");
    }

    if (photoNode && user.photoURL) {
        photoNode.src = user.photoURL;
        photoNode.alt = `${name} profile photo`;
        photoNode.classList.remove("hidden");

        if (initialsNode) {
            initialsNode.classList.add("hidden");
        }

        photoNode.onerror = () => {
    photoNode.classList.add("hidden");

    if (initialsNode) {
        initialsNode.classList.remove("hidden");
    }
};
    } else if (photoNode) {
        photoNode.classList.add("hidden");
        photoNode.removeAttribute("src");
    }
}

/* ---------- DETECTION RESULT ---------- */

function getDetection(reading, data) {
    // Support common field names used by different API versions.
    return (
        reading?.detection ??
        reading?.anomaly_detection ??
        reading?.anomaly_result ??
        reading?.model_result ??
        reading?.analysis ??
        data?.detection ??
        data?.anomaly_detection ??
        data?.model_result ??
        null
    );
}

function isAnomaly(detection) {
    if (!detection) {
        return null;
    }

    const value =
        detection.anomaly_detected ??
        detection.is_anomaly ??
        detection.anomaly ??
        detection.detected;

    if (typeof value === "boolean") {
        return value;
    }

    if (typeof value === "number") {
        return value !== 0;
    }

    if (typeof value === "string") {
        const normalized = value.trim().toLowerCase();

        if (["true", "yes", "anomaly", "abnormal", "detected"].includes(normalized)) {
            return true;
        }

        if (["false", "no", "normal", "none", "not detected"].includes(normalized)) {
            return false;
        }
    }

    return null;
}

function updateDetection(reading, data) {
    const detection = getDetection(reading, data);
    const badge = element("detectionBadge");

    if (!detection) {
        if (badge) {
            badge.className = "status-badge status-pending";
            badge.textContent = "Awaiting analysis";
        }

        setText(
            "detectionDescription",
            "No completed anomaly analysis is available for the latest reading."
        );

        setText("reconstructionError", "--");
        setText("modelThreshold", "--");

        showElement("alertBanner", false);
        return;
    }

    const anomaly = isAnomaly(detection);

    const interpretation =
        detection.interpretation ??
        detection.status ??
        detection.message ??
        "";

    const error =
        detection.reconstruction_error ??
        detection.reconstructionError ??
        detection.error ??
        null;

    const threshold =
        detection.threshold ??
        detection.anomaly_threshold ??
        detection.anomalyThreshold ??
        null;

    setText("reconstructionError", formatNumber(error, 6));
    setText("modelThreshold", formatNumber(threshold, 6));

    if (badge) {
        badge.classList.remove(
            "status-pending",
            "status-normal",
            "status-anomaly"
        );

        if (anomaly === true) {
            badge.classList.add("status-anomaly");
            badge.textContent = "Anomaly detected";
        } else if (anomaly === false) {
            badge.classList.add("status-normal");
            badge.textContent = "Normal pattern";
        } else {
            badge.classList.add("status-pending");
            badge.textContent = "Result unavailable";
        }
    }

    if (interpretation) {
        setText("detectionDescription", interpretation);
    } else if (anomaly === true) {
        setText(
            "detectionDescription",
            "The model identified a sensor pattern that differs from its learned normal patterns."
        );
    } else if (anomaly === false) {
        setText(
            "detectionDescription",
            "The latest completed model analysis did not identify an anomaly."
        );
    } else {
        setText(
            "detectionDescription",
            "The backend returned a detection result, but its anomaly status could not be interpreted."
        );
    }

    showElement("alertBanner", anomaly === true);

    if (anomaly === true) {
        setText("alertTitle", "Potential flood-related anomaly");
        setText(
            "alertMessage",
            interpretation ||
            "The model detected an unusual sensor pattern. Verify the readings and local conditions."
        );
    }
}

/* ---------- LATEST SENSOR READING ---------- */

function updateLatestReading(reading, data) {
    if (!reading) {
        setText("waterLevel", "--");
        setText("rainfall", "--");
        setText("temperature", "--");
        setText("humidity", "--");
        setText("lastUpdated", "No readings available");

        updateDetection(null, data);
        return;
    }

    setText(
        "waterLevel",
        formatNumber(getReadingValue(reading, "water_level"))
    );

    setText(
        "rainfall",
        formatNumber(getReadingValue(reading, "rainfall"))
    );

    setText(
        "temperature",
        formatNumber(getReadingValue(reading, "temperature"), 1)
    );

    setText(
        "humidity",
        formatNumber(getReadingValue(reading, "humidity"), 1)
    );

    const timestamp = getTimestamp(reading);

    setText(
        "lastUpdated",
        timestamp ? `Latest reading: ${formatTimestamp(timestamp)}` : "Latest reading loaded"
    );

    updateDetection(reading, data);
}

/* ---------- CHART CONFIGURATION ---------- */

function getChartTheme() {
    const styles = getComputedStyle(document.documentElement);

    return {
        text: styles.getPropertyValue("--text-secondary").trim() || "#68758c",
        grid: styles.getPropertyValue("--border").trim() || "#e4eaf3",
        surface: styles.getPropertyValue("--surface").trim() || "#ffffff"
    };
}

function buildChartOptions() {
    const theme = getChartTheme();

    return {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
            intersect: false,
            mode: "index"
        },
        plugins: {
            legend: {
                labels: {
                    color: theme.text,
                    usePointStyle: true,
                    pointStyle: "circle",
                    boxWidth: 7,
                    padding: 18,
                    font: {
                        size: 10
                    }
                }
            },
            tooltip: {
                backgroundColor: theme.surface,
                titleColor: theme.text,
                bodyColor: theme.text,
                borderColor: theme.grid,
                borderWidth: 1,
                padding: 10
            }
        },
        scales: {
            x: {
                ticks: {
                    color: theme.text,
                    maxTicksLimit: 7,
                    maxRotation: 0,
                    font: {
                        size: 9
                    }
                },
                grid: {
                    display: false
                },
                border: {
                    display: false
                }
            },
            y: {
                ticks: {
                    color: theme.text,
                    font: {
                        size: 9
                    }
                },
                grid: {
                    color: theme.grid
                },
                border: {
                    display: false
                }
            }
        }
    };
}

function initCharts() {
    if (typeof Chart === "undefined") {
        console.error("Chart.js has not loaded.");
        return;
    }

    const waterCanvas = element("waterChart");
    const environmentCanvas = element("environmentChart");

    if (!waterCanvas || !environmentCanvas) {
        console.error("One or more chart canvases are missing from index.html.");
        return;
    }

    if (waterChart) {
    waterChart.destroy();
    waterChart = null;
}

if (environmentChart) {
    environmentChart.destroy();
    environmentChart = null;
}

    const commonOptions = buildChartOptions();

    waterChart = new Chart(waterCanvas, {
        type: "line",
        data: {
            labels: [],
            datasets: [
                {
                    label: "Water level",
                    data: [],
                    borderColor: "#5d8fee",
                    backgroundColor: "rgba(93, 143, 238, 0.12)",
                    fill: true,
                    borderWidth: 2,
                    tension: 0.32,
                    pointRadius: 2,
                    pointHoverRadius: 5,
                    spanGaps: true
                }
            ]
        },
        options: {
            ...commonOptions,
            plugins: {
                ...commonOptions.plugins,
                legend: {
                    display: false
                }
            }
        }
    });

    environmentChart = new Chart(environmentCanvas, {
        type: "line",
        data: {
            labels: [],
            datasets: [
                {
                    label: "Rainfall",
                    data: [],
                    borderColor: "#22b8cf",
                    backgroundColor: "rgba(34, 184, 207, 0.08)",
                    borderWidth: 2,
                    tension: 0.32,
                    pointRadius: 2,
                    spanGaps: true
                },
                {
                    label: "Temperature (°C)",
                    data: [],
                    borderColor: "#e59b38",
                    backgroundColor: "rgba(229, 155, 56, 0.08)",
                    borderWidth: 2,
                    tension: 0.32,
                    pointRadius: 2,
                    spanGaps: true
                },
                {
                    label: "Humidity (%)",
                    data: [],
                    borderColor: "#8877ed",
                    backgroundColor: "rgba(136, 119, 237, 0.08)",
                    borderWidth: 2,
                    tension: 0.32,
                    pointRadius: 2,
                    spanGaps: true
                }
            ]
        },
        options: commonOptions
    });
}

function getChartLabel(reading) {
    const timestamp = getTimestamp(reading);

    if (!timestamp) {
        return "Unknown";
    }

    const date = new Date(timestamp);

    if (Number.isNaN(date.getTime())) {
        return String(timestamp);
    }

    return date.toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit"
    });
}

function updateCharts(readings) {
    if (!waterChart || !environmentChart) {
        return;
    }

    const orderedReadings = [...readings].sort((a, b) => {
        const first = new Date(getTimestamp(a) || 0).getTime();
        const second = new Date(getTimestamp(b) || 0).getTime();

        return first - second;
    });

    const labels = orderedReadings.map(getChartLabel);

    waterChart.data.labels = labels;
    waterChart.data.datasets[0].data = orderedReadings.map(reading => {
        const value = getReadingValue(reading, "water_level");
        return value == null ? null : Number(value);
    });

    environmentChart.data.labels = labels;
    environmentChart.data.datasets[0].data = orderedReadings.map(reading => {
        const value = getReadingValue(reading, "rainfall");
        return value == null ? null : Number(value);
    });

    environmentChart.data.datasets[1].data = orderedReadings.map(reading => {
        const value = getReadingValue(reading, "temperature");
        return value == null ? null : Number(value);
    });

    environmentChart.data.datasets[2].data = orderedReadings.map(reading => {
        const value = getReadingValue(reading, "humidity");
        return value == null ? null : Number(value);
    });

    waterChart.update();
    environmentChart.update();
}

/* ---------- READINGS TABLE ---------- */

function createCell(value) {
    const cell = document.createElement("td");
    cell.textContent = value;
    return cell;
}

function createDetectionCell(reading, data) {
    const cell = document.createElement("td");
    const badge = document.createElement("span");

    const detection = getDetection(reading, data);
    const anomaly = isAnomaly(detection);

    badge.classList.add("detection-label");

    if (anomaly === true) {
        badge.classList.add("anomaly");
        badge.textContent = "Anomaly";
    } else if (anomaly === false) {
        badge.classList.add("normal");
        badge.textContent = "Normal";
    } else {
        badge.classList.add("pending");
        badge.textContent = "Included in sequence";
    }

    cell.appendChild(badge);

    return cell;
}

function updateTable(readings, data) {
    const tbody = element("readingsTable");

    if (!tbody) {
        return;
    }

    tbody.replaceChildren();

    const orderedReadings = [...readings].sort((a, b) => {
        const first = new Date(getTimestamp(a) || 0).getTime();
        const second = new Date(getTimestamp(b) || 0).getTime();

        return second - first;
    });

    const visibleReadings = orderedReadings.slice(0, 240);

    setText("tableRecordCount", visibleReadings.length);
    setText("readingCount", readings.length);

    if (visibleReadings.length === 0) {
        const row = document.createElement("tr");
        const cell = document.createElement("td");

        cell.colSpan = 6;
        cell.className = "empty-cell";
        cell.textContent = "No sensor readings have been returned by the backend.";

        row.appendChild(cell);
        tbody.appendChild(row);

        return;
    }

    for (const reading of visibleReadings) {
        const row = document.createElement("tr");

        row.appendChild(
            createCell(formatTimestamp(getTimestamp(reading)))
        );

        row.appendChild(
            createCell(formatNumber(getReadingValue(reading, "water_level")))
        );

        row.appendChild(
            createCell(formatNumber(getReadingValue(reading, "rainfall")))
        );

        row.appendChild(
            createCell(formatNumber(getReadingValue(reading, "temperature"), 1))
        );

        row.appendChild(
            createCell(formatNumber(getReadingValue(reading, "humidity"), 1))
        );

        row.appendChild(createDetectionCell(reading, data));

        tbody.appendChild(row);
    }
}

/* ---------- BACKEND DATA NORMALIZATION ---------- */

function normalizeDashboardResponse(data) {
    if (!data || typeof data !== "object") {
        throw new Error("The backend returned an invalid response.");
    }

    const readings = Array.isArray(data.readings)
        ? data.readings
        : [];

    const latestReading =
        data.latest_reading ??
        data.latestReading ??
        readings.reduce((latest, reading) => {
            if (!latest) {
                return reading;
            }

            const currentTime = new Date(
                getTimestamp(reading) || 0
            ).getTime();

            const latestTime = new Date(
                getTimestamp(latest) || 0
            ).getTime();

           return currentTime > latestTime ? reading : latest;
        }, null);

    return {
        ...data,
        readings,
        latest_reading: latestReading,
        count: data.count ?? readings.length
    };
}

/* ---------- LOAD DASHBOARD DATA ---------- */


async function loadBackendDashboard() {
    if (isLoading) {
        return;
    }

    isLoading = true;
    setConnection("connecting", "Connecting...");

    const refreshButton = element("refreshButton");

    if (refreshButton) {
        refreshButton.disabled = true;
        refreshButton.style.opacity = "0.7";
    }

    try {
        // Ensure a Firebase user is signed in.
        const user = auth.currentUser;

        if (!user) {
            throw new Error(
                "You must sign in before accessing dashboard data."
            );
        }

        // Retrieve the Firebase ID token.
        const idToken = await user.getIdToken();

        // Send the token to FastAPI for verification.
       const response = await fetch(`${API_URL}?limit=240`, {
            method: "GET",
            headers: {
                "Accept": "application/json",
                "Authorization": `Bearer ${idToken}`
            },
            cache: "no-store"
        });

        if (!response.ok) {
            if (response.status === 401) {
                throw new Error(
                    "Authentication failed. Please sign in again."
                );
            }

            throw new Error(
                `Backend returned HTTP ${response.status}.`
            );
        }

        const responseData = await response.json();
        const data = normalizeDashboardResponse(responseData);

        latestDashboardData = data;

        setConnection("connected", "Backend connected");

        updateLatestReading(data.latest_reading, data);
        updateCharts(data.readings);
        updateTable(data.readings, data);

    } catch (error) {
        console.error("Unable to load dashboard data:", error);

        setConnection("disconnected", "Backend unavailable");

        setText(
            "lastUpdated",
            "Unable to refresh — check authentication or backend"
        );

        const tbody = element("readingsTable");

        if (tbody && !latestDashboardData) {
            tbody.replaceChildren();

            const row = document.createElement("tr");
            const cell = document.createElement("td");

            cell.colSpan = 6;
            cell.className = "empty-cell";
            cell.textContent =
                "Could not retrieve readings. Check your sign-in and backend connection.";

            row.appendChild(cell);
            tbody.appendChild(row);
        }

    } finally {
        isLoading = false;

        if (refreshButton) {
            refreshButton.disabled = false;
            refreshButton.style.opacity = "";
        }
    }
}

/* ---------- THEME ---------- */

function applyTheme(theme) {
    const root = document.documentElement;

    let resolvedTheme = theme;

    if (theme === "system") {
        resolvedTheme = window.matchMedia("(prefers-color-scheme: dark)").matches
            ? "dark"
            : "light";
    }

    root.dataset.theme = resolvedTheme;

    if (element("themeSelect")) {
        element("themeSelect").value = theme;
    }

    try {
        localStorage.setItem("floodwatch-theme", theme);
    } catch (error) {
        console.warn("Theme preference could not be saved.", error);
    }

    // Rebuild chart options so the labels and grid match the theme.
    if (waterChart && environmentChart) {
        const waterData = waterChart.data;
        const environmentData = environmentChart.data;

        initCharts();

        waterChart.data = waterData;
        environmentChart.data = environmentData;

        waterChart.update();
        environmentChart.update();
    }
}

function initializeTheme() {
    let savedTheme = "system";

    try {
        savedTheme = localStorage.getItem("floodwatch-theme") || "system";
    } catch (error) {
        console.warn("Could not read saved theme.", error);
    }

    if (!["light", "dark", "system"].includes(savedTheme)) {
        savedTheme = "system";
    }

    applyTheme(savedTheme);

    const selector = element("themeSelect");

    if (selector) {
        selector.addEventListener("change", event => {
            applyTheme(event.target.value);
        });
    }

    const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");

    mediaQuery.addEventListener?.("change", () => {
        const currentTheme = element("themeSelect")?.value;

        if (currentTheme === "system") {
            applyTheme("system");
        }
    });
}

/* ---------- BUTTONS AND NAVIGATION ---------- */

function initializeControls() {
    const refreshButton = element("refreshButton");

    if (refreshButton) {
        refreshButton.addEventListener("click", () => {
            loadBackendDashboard();
        });
    }
}

/* ---------- APPLICATION STARTUP ---------- */

document.addEventListener("DOMContentLoaded", () => {
    console.log("FloodWatch dashboard starting...");

    initializeAuthenticationControls();

    onAuthStateChanged(auth, user => {
        handleAuthentication(user);
    });

    console.log("Firebase authentication initialized.");

    document
    .getElementById("emailSignInButton")
    .addEventListener("click", handleEmailSignIn);

document
    .getElementById("completeEmailSignInButton")
    .addEventListener("click", completeEmailSignIn);

checkForEmailSignInLink();
});