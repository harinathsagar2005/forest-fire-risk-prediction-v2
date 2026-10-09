/**
 * Forest Fire Risk Prediction Web Application
 * Client-Side Application Logic
 * Vardhaman College of Engineering (Autonomous) | Team 27
 */

let mapInstance = null;
let mapMarkers = [];
let comparisonChart = null;
let featureImportanceChart = null;
let cachedGisZones = [];

// Initialize application on DOM ready
document.addEventListener('DOMContentLoaded', () => {
    initApp();
});

async function initApp() {
    // 1. Trigger initial prediction with default values
    await runPrediction();

    // 2. Initialize Charts
    initCharts();

    // 3. Load Dataset Sample
    loadDatasetSample();
}

// -------------------------------------------------------------
// TAB NAVIGATION
// -------------------------------------------------------------
function switchTab(tabId) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.add('hidden');
    });

    // Reset button states
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active', 'bg-slate-800', 'text-white');
        btn.classList.add('text-slate-400');
    });

    // Show target tab
    const targetTab = document.getElementById(tabId);
    if (targetTab) {
        targetTab.classList.remove('hidden');
    }

    // Activate corresponding button
    const targetBtn = document.getElementById(`btn-${tabId}`);
    if (targetBtn) {
        targetBtn.classList.add('active', 'bg-slate-800', 'text-white');
        targetBtn.classList.remove('text-slate-400');
    }

    // Lazy load / re-render GIS map when navigating to GIS tab
    if (tabId === 'tab-gis') {
        setTimeout(() => {
            if (!mapInstance) {
                initGisMap();
            } else {
                mapInstance.invalidateSize();
            }
        }, 150);
    }
}

// -------------------------------------------------------------
// INPUT VALUE SYNC & PRESETS
// -------------------------------------------------------------
function updateVal(param, unit) {
    const input = document.getElementById(`input-${param}`);
    const display = document.getElementById(`val-${param}`);
    if (input && display) {
        display.innerText = `${input.value} ${unit}`.trim();
    }
}

function applyPreset(scenario) {
    const presets = {
        extreme_summer: {
            temp: 41.5, rh: 16, rain: 0.0, wind: 34.0, ndvi: 0.18, slope: 24.0, dist: 550, pressure: 1008, elev: 720
        },
        monsoon: {
            temp: 23.0, rh: 84, rain: 22.0, wind: 9.0, ndvi: 0.74, slope: 8.0, dist: 2800, pressure: 1015, elev: 420
        },
        moderate_spring: {
            temp: 28.0, rh: 48, rain: 0.0, wind: 15.0, ndvi: 0.45, slope: 12.0, dist: 1600, pressure: 1012, elev: 580
        },
        mountain_ridge: {
            temp: 34.0, rh: 26, rain: 0.0, wind: 31.0, ndvi: 0.26, slope: 32.0, dist: 950, pressure: 994, elev: 1520
        }
    };

    const p = presets[scenario];
    if (!p) return;

    document.getElementById('input-temp').value = p.temp;
    updateVal('temp', '°C');

    document.getElementById('input-rh').value = p.rh;
    updateVal('rh', '%');

    document.getElementById('input-rain').value = p.rain;
    updateVal('rain', 'mm');

    document.getElementById('input-wind').value = p.wind;
    updateVal('wind', 'km/h');

    document.getElementById('input-ndvi').value = p.ndvi;
    updateVal('ndvi', '');

    document.getElementById('input-slope').value = p.slope;
    updateVal('slope', '°');

    document.getElementById('input-dist').value = p.dist;
    updateVal('dist', 'm');

    document.getElementById('input-pressure').value = p.pressure;
    document.getElementById('input-elev').value = p.elev;

    runPrediction();
}

// -------------------------------------------------------------
// PREDICTION EXECUTION
// -------------------------------------------------------------
async function runPrediction() {
    const btn = document.getElementById('btn-predict');
    if (btn) {
        btn.disabled = true;
        btn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> <span>Processing Model Inference...</span>`;
    }

    const payload = {
        temperature: parseFloat(document.getElementById('input-temp').value),
        relative_humidity: parseFloat(document.getElementById('input-rh').value),
        rainfall: parseFloat(document.getElementById('input-rain').value),
        wind_speed: parseFloat(document.getElementById('input-wind').value),
        atmospheric_pressure: parseFloat(document.getElementById('input-pressure').value),
        ndvi: parseFloat(document.getElementById('input-ndvi').value),
        elevation: parseFloat(document.getElementById('input-elev').value),
        slope: parseFloat(document.getElementById('input-slope').value),
        distance_to_road: parseFloat(document.getElementById('input-dist').value),
        model: document.getElementById('model-select').value
    };

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await response.json();

        if (data.status === 'success') {
            renderPredictionResults(data);
        } else {
            alert('Prediction error: ' + (data.message || 'Unknown error'));
        }
    } catch (err) {
        console.error('Inference error:', err);
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = `<i class="fa-solid fa-bolt-lightning text-yellow-300"></i> <span>Calculate Forest Fire Risk Index</span>`;
        }
    }
}

function renderPredictionResults(data) {
    const pred = data.prediction;
    const model = data.chosen_model;
    const allModels = data.all_models;
    const contributions = data.feature_contributions;
    const sop = data.disaster_management;

    // Model Badge
    document.getElementById('result-model-badge').innerText = model;

    // Status Pill
    const pill = document.getElementById('result-status-pill');
    pill.innerHTML = `<i class="fa-solid fa-shield-cat mr-1.5"></i> ${pred.class_name}`;
    
    // Risk % and Bar
    const riskPctEl = document.getElementById('result-risk-pct');
    riskPctEl.innerText = pred.risk_percentage.toFixed(1);

    const barEl = document.getElementById('result-risk-bar');
    barEl.style.width = `${Math.min(100, Math.max(5, pred.risk_percentage))}%`;

    // Colors depending on risk class
    if (pred.class_name === 'High Risk') {
        pill.className = 'inline-flex items-center px-4 py-1.5 rounded-full text-xs font-black uppercase tracking-wider bg-red-950/90 text-red-400 border border-red-700 animate-pulse mb-3';
        riskPctEl.className = 'text-5xl font-black tracking-tight text-red-500 font-mono';
        barEl.className = 'h-full rounded-full bg-red-500 transition-all duration-500';
    } else if (pred.class_name === 'Medium Risk') {
        pill.className = 'inline-flex items-center px-4 py-1.5 rounded-full text-xs font-black uppercase tracking-wider bg-amber-950/90 text-amber-400 border border-amber-700 mb-3';
        riskPctEl.className = 'text-5xl font-black tracking-tight text-amber-400 font-mono';
        barEl.className = 'h-full rounded-full bg-amber-500 transition-all duration-500';
    } else {
        pill.className = 'inline-flex items-center px-4 py-1.5 rounded-full text-xs font-black uppercase tracking-wider bg-emerald-950/90 text-emerald-400 border border-emerald-700 mb-3';
        riskPctEl.className = 'text-5xl font-black tracking-tight text-emerald-400 font-mono';
        barEl.className = 'h-full rounded-full bg-emerald-500 transition-all duration-500';
    }

    // Probabilities
    document.getElementById('prob-low').innerText = `${pred.probabilities['Low Risk']}%`;
    document.getElementById('prob-bar-low').style.width = `${pred.probabilities['Low Risk']}%`;

    document.getElementById('prob-med').innerText = `${pred.probabilities['Medium Risk']}%`;
    document.getElementById('prob-bar-med').style.width = `${pred.probabilities['Medium Risk']}%`;

    document.getElementById('prob-high').innerText = `${pred.probabilities['High Risk']}%`;
    document.getElementById('prob-bar-high').style.width = `${pred.probabilities['High Risk']}%`;

    // Multi-Model Consensus
    const rfEl = document.getElementById('consensus-rf');
    rfEl.innerText = allModels['Random Forest'].prediction;
    rfEl.className = `font-bold ${getColorClass(allModels['Random Forest'].prediction)}`;

    const svmEl = document.getElementById('consensus-svm');
    svmEl.innerText = allModels['Support Vector Machine'].prediction;
    svmEl.className = `font-bold ${getColorClass(allModels['Support Vector Machine'].prediction)}`;

    const lrEl = document.getElementById('consensus-lr');
    lrEl.innerText = allModels['Logistic Regression'].prediction;
    lrEl.className = `font-bold ${getColorClass(allModels['Logistic Regression'].prediction)}`;

    // SOP Content
    const sopContainer = document.getElementById('sop-content');
    sopContainer.innerHTML = sop.actions.map(action => `
        <div class="flex items-start gap-2">
            <i class="fa-solid fa-circle-check ${getIconColorClass(pred.class_name)} mt-0.5 text-[10px]"></i>
            <span>${action}</span>
        </div>
    `).join('');

    // Feature Contribution Grid (Local Explainability)
    const contribGrid = document.getElementById('feature-contributions-grid');
    contribGrid.innerHTML = contributions.slice(0, 8).map(item => {
        const isRiskPush = item.impact > 0;
        const color = isRiskPush ? 'text-red-400' : 'text-emerald-400';
        const sign = isRiskPush ? '+' : '';
        const bg = isRiskPush ? 'bg-red-950/30 border-red-900/40' : 'bg-emerald-950/30 border-emerald-900/40';
        return `
            <div class="p-3 rounded-xl border ${bg} flex flex-col justify-between">
                <div>
                    <span class="text-slate-400 block text-[11px] font-medium">${item.feature}</span>
                    <span class="text-slate-200 font-mono font-bold">${item.val} ${item.unit}</span>
                </div>
                <div class="mt-2 pt-1 border-t border-slate-800/80 flex items-center justify-between text-[11px]">
                    <span class="text-slate-500">Risk Delta:</span>
                    <span class="font-mono font-bold ${color}">${sign}${item.impact.toFixed(1)} pts</span>
                </div>
            </div>
        `;
    }).join('');
}

function getColorClass(riskLevel) {
    if (riskLevel === 'High Risk') return 'text-red-400';
    if (riskLevel === 'Medium Risk') return 'text-amber-400';
    return 'text-emerald-400';
}

function getIconColorClass(riskLevel) {
    if (riskLevel === 'High Risk') return 'text-red-400';
    if (riskLevel === 'Medium Risk') return 'text-amber-400';
    return 'text-emerald-400';
}

// -------------------------------------------------------------
// GIS LEAFLET MAP IMPLEMENTATION
// -------------------------------------------------------------
async function initGisMap() {
    const mapContainer = document.getElementById('gis-map');
    if (!mapContainer) return;

    // Center on Telangana / Nallamala Forest Reserves
    mapInstance = L.map('gis-map').setView([16.35, 78.85], 8);

    // Layer 1: OpenStreetMap Standard
    const osmLayer = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        maxZoom: 18,
        attribution: '&copy; OpenStreetMap contributors'
    });

    // Layer 2: ESRI World Imagery (Satellite)
    const satLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
        maxZoom: 18,
        attribution: 'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS'
    });

    // Layer 3: OpenTopoMap
    const topoLayer = L.tileLayer('https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png', {
        maxZoom: 17,
        attribution: '&copy; OpenTopoMap'
    });

    // Add satellite layer by default for rich forest inspection
    satLayer.addTo(mapInstance);

    // Layer control
    const baseMaps = {
        "Satellite Imagery": satLayer,
        "Topographic Map": topoLayer,
        "Street Map": osmLayer
    };
    L.control.layers(baseMaps).addTo(mapInstance);

    // Load zones from API
    await loadGisZones();
}

async function loadGisZones() {
    try {
        const res = await fetch('/api/gis-data');
        const data = await res.json();
        cachedGisZones = data.zones || [];
        renderMapMarkers(cachedGisZones);
        renderGisZoneList(cachedGisZones);
    } catch (err) {
        console.error('Failed to load GIS data:', err);
    }
}

function renderMapMarkers(zones) {
    // Clear existing markers
    mapMarkers.forEach(m => mapInstance.removeLayer(m));
    mapMarkers = [];

    zones.forEach(zone => {
        let color = '#10b981'; // Green
        let pulseClass = 'low-pulse';
        if (zone.risk_level === 'High Risk') {
            color = '#ef4444'; // Red
            pulseClass = 'high-pulse';
        } else if (zone.risk_level === 'Medium Risk') {
            color = '#f59e0b'; // Amber
            pulseClass = 'med-pulse';
        }

        // Circular risk radius marker
        const circle = L.circle([zone.lat, zone.lng], {
            color: color,
            fillColor: color,
            fillOpacity: 0.35,
            radius: 8500
        }).addTo(mapInstance);

        // Core marker
        const marker = L.circleMarker([zone.lat, zone.lng], {
            radius: 9,
            fillColor: color,
            color: '#ffffff',
            weight: 2,
            opacity: 1,
            fillOpacity: 0.95
        }).addTo(mapInstance);

        // Rich Popup Window
        const popupContent = `
            <div class="p-2 min-w-[210px] text-slate-900 font-sans">
                <div class="border-b border-slate-200 pb-1.5 mb-1.5">
                    <span class="text-[10px] font-bold uppercase text-slate-500">${zone.id} &bull; ${zone.region}</span>
                    <h4 class="font-bold text-xs text-slate-900">${zone.name}</h4>
                </div>
                <div class="flex items-center justify-between mb-2">
                    <span class="text-xs font-semibold px-2 py-0.5 rounded text-white" style="background-color: ${color}">
                        ${zone.risk_level} (${zone.risk_score}%)
                    </span>
                    <span class="text-[10px] font-bold text-red-700">${zone.status}</span>
                </div>
                <div class="grid grid-cols-2 gap-1 text-[11px] text-slate-700 bg-slate-100 p-1.5 rounded mb-2">
                    <div>Temp: <strong>${zone.temperature}°C</strong></div>
                    <div>Humidity: <strong>${zone.humidity}%</strong></div>
                    <div>Wind: <strong>${zone.wind_speed} km/h</strong></div>
                    <div>NDVI: <strong>${zone.ndvi}</strong></div>
                    <div>Slope: <strong>${zone.slope}°</strong></div>
                    <div>Rain: <strong>${zone.rainfall} mm</strong></div>
                </div>
                <p class="text-[10px] text-slate-600 italic">
                    <strong>Action:</strong> ${zone.recommended_action}
                </p>
            </div>
        `;
        marker.bindPopup(popupContent);
        circle.bindPopup(popupContent);

        mapMarkers.push(marker);
        mapMarkers.push(circle);
    });
}

function renderGisZoneList(zones) {
    const list = document.getElementById('gis-zone-list');
    if (!list) return;

    list.innerHTML = zones.map(zone => {
        let badgeColor = 'bg-emerald-950 text-emerald-400 border-emerald-800';
        if (zone.risk_level === 'High Risk') badgeColor = 'bg-red-950 text-red-400 border-red-800';
        else if (zone.risk_level === 'Medium Risk') badgeColor = 'bg-amber-950 text-amber-400 border-amber-800';

        return `
            <div onclick="flyToZone(${zone.lat}, ${zone.lng})" class="p-3 bg-slate-900/70 hover:bg-slate-850 rounded-xl border border-slate-800/80 cursor-pointer transition flex items-center justify-between">
                <div>
                    <div class="flex items-center gap-1.5">
                        <span class="text-[10px] font-bold text-slate-500">${zone.id}</span>
                        <h4 class="text-xs font-bold text-slate-200">${zone.name}</h4>
                    </div>
                    <p class="text-[10px] text-slate-400 mt-0.5">
                        <i class="fa-solid fa-temperature-half text-orange-400 mr-1"></i>${zone.temperature}°C &bull;
                        <i class="fa-solid fa-wind text-teal-400 mr-1"></i>${zone.wind_speed} km/h &bull;
                        <i class="fa-solid fa-seedling text-emerald-400 mr-1"></i>NDVI ${zone.ndvi}
                    </p>
                </div>
                <div class="text-right">
                    <span class="text-[10px] px-2 py-0.5 rounded border font-bold ${badgeColor}">
                        ${zone.risk_level}
                    </span>
                    <span class="block text-[10px] font-mono font-bold text-slate-400 mt-1">${zone.risk_score}%</span>
                </div>
            </div>
        `;
    }).join('');
}

function flyToZone(lat, lng) {
    if (mapInstance) {
        mapInstance.flyTo([lat, lng], 11, { duration: 1.5 });
    }
}

function resetMapCenter() {
    if (mapInstance) {
        mapInstance.flyTo([16.35, 78.85], 8, { duration: 1.2 });
    }
}

function filterMapZones(level) {
    // Update button states
    document.querySelectorAll('.map-filter-btn').forEach(btn => {
        btn.classList.remove('bg-slate-800', 'text-white');
    });

    if (level === 'all') {
        renderMapMarkers(cachedGisZones);
        renderGisZoneList(cachedGisZones);
    } else {
        const filtered = cachedGisZones.filter(z => z.risk_level === level);
        renderMapMarkers(filtered);
        renderGisZoneList(filtered);
    }
}

function triggerSimulatedEmergency() {
    alert("⚠️ SIMULATED EMERGENCY ALERT:\nHigh-velocity dry foehn winds (38 km/h) & extreme temperature spike (+4.5°C) detected in Amrabad Tiger Reserve and Bandipur Forest Edge.\n\nAutomated Siren Active: Rapid response firefighting teams mobilized.");
    // Temporarily elevate Amrabad zone
    if (cachedGisZones.length > 0) {
        flyToZone(cachedGisZones[0].lat, cachedGisZones[0].lng);
    }
}

// -------------------------------------------------------------
// CHARTS INITIALIZATION (Chart.js)
// -------------------------------------------------------------
async function initCharts() {
    try {
        const res = await fetch('/api/compare');
        const data = await res.json();
        const comp = data.comparison;
        const featImp = data.feature_importance;

        // 1. Comparison Bar Chart
        const ctxComp = document.getElementById('chart-comparison');
        if (ctxComp && comp) {
            const labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score'];
            const rfVals = [comp['Random Forest'].accuracy, comp['Random Forest'].precision, comp['Random Forest'].recall, comp['Random Forest'].f1_score];
            const svmVals = [comp['Support Vector Machine'].accuracy, comp['Support Vector Machine'].precision, comp['Support Vector Machine'].recall, comp['Support Vector Machine'].f1_score];
            const lrVals = [comp['Logistic Regression'].accuracy, comp['Logistic Regression'].precision, comp['Logistic Regression'].recall, comp['Logistic Regression'].f1_score];

            comparisonChart = new Chart(ctxComp, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [
                        {
                            label: 'Random Forest (Best)',
                            data: rfVals,
                            backgroundColor: '#10b981',
                            borderRadius: 6
                        },
                        {
                            label: 'Support Vector Machine',
                            data: svmVals,
                            backgroundColor: '#a855f7',
                            borderRadius: 6
                        },
                        {
                            label: 'Logistic Regression',
                            data: lrVals,
                            backgroundColor: '#3b82f6',
                            borderRadius: 6
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { labels: { color: '#cbd5e1', font: { size: 11, weight: 'bold' } } }
                    },
                    scales: {
                        y: {
                            min: 60,
                            max: 100,
                            ticks: { color: '#94a3b8', font: { family: 'monospace' } },
                            grid: { color: '#334155' }
                        },
                        x: {
                            ticks: { color: '#cbd5e1', font: { weight: 'bold' } },
                            grid: { display: false }
                        }
                    }
                }
            });
        }

        // 2. Feature Importance Horizontal Bar Chart
        const ctxFeat = document.getElementById('chart-feature-importance');
        if (ctxFeat && featImp) {
            const featLabels = featImp.map(f => f.feature.replace('_', ' ').toUpperCase());
            const featVals = featImp.map(f => f.importance);

            featureImportanceChart = new Chart(ctxFeat, {
                type: 'bar',
                data: {
                    labels: featLabels,
                    datasets: [{
                        label: 'Relative Importance (%)',
                        data: featVals,
                        backgroundColor: '#ea580c',
                        borderRadius: 6
                    }]
                },
                options: {
                    indexAxis: 'y',
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false }
                    },
                    scales: {
                        x: {
                            max: Math.max(...featVals) + 5,
                            ticks: { color: '#94a3b8', font: { family: 'monospace' } },
                            grid: { color: '#334155' }
                        },
                        y: {
                            ticks: { color: '#cbd5e1', font: { size: 10, weight: 'bold' } },
                            grid: { display: false }
                        }
                    }
                }
            });
        }
    } catch (err) {
        console.error('Failed to init charts:', err);
    }
}

// -------------------------------------------------------------
// HISTORICAL DATASET TABLE LOADER
// -------------------------------------------------------------
async function loadDatasetSample() {
    try {
        const res = await fetch('/api/historical-sample');
        const data = await res.json();
        const tbody = document.getElementById('dataset-sample-tbody');
        if (tbody && data.sample) {
            tbody.innerHTML = data.sample.slice(0, 15).map(row => {
                let badgeClass = 'text-emerald-400 bg-emerald-950 border-emerald-800';
                let label = 'Low Risk';
                if (row.risk_category === 2) {
                    badgeClass = 'text-red-400 bg-red-950 border-red-800';
                    label = 'High Risk';
                } else if (row.risk_category === 1) {
                    badgeClass = 'text-amber-400 bg-amber-950 border-amber-800';
                    label = 'Medium Risk';
                }

                return `
                    <tr class="hover:bg-slate-900/60 transition">
                        <td class="py-2 px-3 text-slate-200 font-bold">${row.temperature}°C</td>
                        <td class="py-2 px-3 text-slate-300">${row.relative_humidity}%</td>
                        <td class="py-2 px-3 text-slate-300">${row.rainfall}</td>
                        <td class="py-2 px-3 text-slate-300">${row.wind_speed}</td>
                        <td class="py-2 px-3 text-slate-300">${row.ndvi}</td>
                        <td class="py-2 px-3 text-slate-300">${row.slope}°</td>
                        <td class="py-2 px-3 text-slate-300">${row.elevation}m</td>
                        <td class="py-2 px-3 text-slate-300">${row.distance_to_road}m</td>
                        <td class="py-2 px-3">
                            <span class="px-2 py-0.5 rounded text-[10px] font-bold border ${badgeClass}">
                                ${label}
                            </span>
                        </td>
                    </tr>
                `;
            }).join('');
        }
    } catch (err) {
        console.error('Failed to load dataset sample:', err);
    }
}

// -------------------------------------------------------------
// LLM WILDFIRE COPILOT & DISASTER ADVISORY
// -------------------------------------------------------------
let currentLlmReportMarkdown = "";

function toggleApiKeyInput() {
    const drawer = document.getElementById('api-key-drawer');
    if (drawer) {
        drawer.classList.toggle('hidden');
        const storedKey = localStorage.getItem('vce_gemini_api_key');
        if (storedKey) {
            document.getElementById('gemini-api-key-input').value = storedKey;
        }
    }
}

function saveApiKey() {
    const val = document.getElementById('gemini-api-key-input').value.trim();
    if (val) {
        localStorage.setItem('vce_gemini_api_key', val);
        document.getElementById('api-key-status-label').innerText = "Gemini API Key: Active";
        alert("Google Gemini API Key saved locally!");
    } else {
        localStorage.removeItem('vce_gemini_api_key');
        document.getElementById('api-key-status-label').innerText = "Gemini API Key: Optional";
        alert("Using built-in Expert LLM Engine.");
    }
    toggleApiKeyInput();
}

function openLlmCopilotWithCurrentState() {
    switchTab('tab-llm');
    const select = document.getElementById('llm-sector-select');
    if (select) {
        select.value = 'CURRENT_INPUTS';
    }
    generateLlmIncidentReport();
}

async function generateLlmIncidentReport() {
    const reportBox = document.getElementById('llm-report-content');
    const badge = document.getElementById('llm-source-badge');
    const sectorVal = document.getElementById('llm-sector-select').value;
    const apiKey = localStorage.getItem('vce_gemini_api_key') || null;

    if (!reportBox) return;

    reportBox.innerHTML = `
        <div class="text-center py-16 text-purple-400 space-y-3">
            <i class="fa-solid fa-brain fa-bounce text-4xl"></i>
            <p class="font-bold text-sm">Synthesizing LLM Wildfire Incident Assessment...</p>
            <p class="text-xs text-slate-500">Evaluating thermodynamic fuel moisture, flame vectors & resource deployment</p>
        </div>
    `;

    // Construct context based on selection
    let context = {};
    if (sectorVal === 'CURRENT_INPUTS' || cachedGisZones.length === 0) {
        const activeClass = document.getElementById('result-class-name') ? document.getElementById('result-class-name').innerText : 'High Risk';
        const activeScore = document.getElementById('result-risk-pct') ? parseFloat(document.getElementById('result-risk-pct').innerText) : 92.4;
        const activeModel = document.getElementById('model-select') ? document.getElementById('model-select').value : 'Random Forest (95.0% Accuracy)';

        context = {
            temperature: parseFloat(document.getElementById('input-temp').value),
            relative_humidity: parseFloat(document.getElementById('input-rh').value),
            rainfall: parseFloat(document.getElementById('input-rain').value),
            wind_speed: parseFloat(document.getElementById('input-wind').value),
            ndvi: parseFloat(document.getElementById('input-ndvi').value),
            slope: parseFloat(document.getElementById('input-slope').value),
            distance_to_road: parseFloat(document.getElementById('input-dist').value),
            risk_level: activeClass,
            risk_score: activeScore,
            model_name: activeModel,
            location: 'Current Monitored Sector (Predictor Telemetry)',
            api_key: apiKey
        };
    } else {
        const found = cachedGisZones.find(z => z.id === sectorVal) || cachedGisZones[0];
        context = {
            temperature: found.temperature,
            relative_humidity: found.humidity,
            rainfall: found.rainfall,
            wind_speed: found.wind_speed,
            ndvi: found.ndvi,
            slope: found.slope,
            distance_to_road: found.distance_to_road,
            risk_level: found.risk_level,
            risk_score: found.risk_score,
            model_name: 'Random Forest (95.0% Accuracy)',
            location: `${found.name} (${found.region})`,
            api_key: apiKey
        };
    }

    try {
        const res = await fetch('/api/llm/incident-report', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(context)
        });
        const data = await res.json();

        if (data.status === 'success') {
            currentLlmReportMarkdown = data.report_markdown;
            if (badge) {
                badge.innerText = data.source;
            }
            reportBox.innerHTML = renderMarkdown(data.report_markdown);
        } else {
            reportBox.innerHTML = `<div class="p-4 text-red-400 bg-red-950/40 rounded-xl">Error generating report: ${data.message}</div>`;
        }
    } catch (err) {
        reportBox.innerHTML = `<div class="p-4 text-red-400 bg-red-950/40 rounded-xl">Connection error: ${err.message}</div>`;
    }
}

function copyLlmReport() {
    if (!currentLlmReportMarkdown) {
        alert("Please generate a report first.");
        return;
    }
    navigator.clipboard.writeText(currentLlmReportMarkdown).then(() => {
        alert("Executive Incident Report copied to clipboard!");
    });
}

// -------------------------------------------------------------
// INTERACTIVE CONVERSATIONAL COPILOT (CHATBOT)
// -------------------------------------------------------------
async function handleChatSubmit(e) {
    if (e) e.preventDefault();
    const input = document.getElementById('llm-chat-input');
    const stream = document.getElementById('llm-chat-stream');
    const sendBtn = document.getElementById('btn-chat-send');
    if (!input || !stream) return;

    const message = input.value.trim();
    if (!message) return;

    // Append User Message
    input.value = '';
    stream.innerHTML += `
        <div class="flex items-start justify-end gap-2.5">
            <div class="bg-purple-600 p-3 rounded-2xl rounded-tr-sm text-white leading-relaxed max-w-[85%]">
                ${escapeHtml(message)}
            </div>
            <div class="w-6 h-6 rounded-md bg-purple-700 flex items-center justify-center text-white text-[10px] shrink-0 mt-0.5">
                <i class="fa-solid fa-user"></i>
            </div>
        </div>
    `;
    stream.scrollTop = stream.scrollHeight;

    // Loading indicator
    const loadingId = 'loading-' + Date.now();
    stream.innerHTML += `
        <div id="${loadingId}" class="flex items-start gap-2.5">
            <div class="w-6 h-6 rounded-md bg-purple-900/60 border border-purple-700/60 flex items-center justify-center text-purple-300 text-[10px] shrink-0 mt-0.5">
                <i class="fa-solid fa-robot"></i>
            </div>
            <div class="bg-slate-900 p-3 rounded-2xl rounded-tl-sm border border-slate-800 text-slate-400">
                <i class="fa-solid fa-spinner fa-spin mr-1.5"></i> Analyzing telemetry & incident protocols...
            </div>
        </div>
    `;
    stream.scrollTop = stream.scrollHeight;

    if (sendBtn) sendBtn.disabled = true;

    // Gather active telemetry context
    const apiKey = localStorage.getItem('vce_gemini_api_key') || null;
    const context = {
        temperature: parseFloat(document.getElementById('input-temp') ? document.getElementById('input-temp').value : 35),
        relative_humidity: parseFloat(document.getElementById('input-rh') ? document.getElementById('input-rh').value : 25),
        wind_speed: parseFloat(document.getElementById('input-wind') ? document.getElementById('input-wind').value : 25),
        rainfall: parseFloat(document.getElementById('input-rain') ? document.getElementById('input-rain').value : 0),
        ndvi: parseFloat(document.getElementById('input-ndvi') ? document.getElementById('input-ndvi').value : 0.25),
        slope: parseFloat(document.getElementById('input-slope') ? document.getElementById('input-slope').value : 18),
        risk_level: document.getElementById('result-class-name') ? document.getElementById('result-class-name').innerText : 'High Risk'
    };

    try {
        const res = await fetch('/api/llm/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: message, context: context, api_key: apiKey })
        });
        const data = await res.json();

        const loadEl = document.getElementById(loadingId);
        if (loadEl) loadEl.remove();

        if (data.status === 'success') {
            stream.innerHTML += `
                <div class="flex items-start gap-2.5">
                    <div class="w-6 h-6 rounded-md bg-purple-900/60 border border-purple-700/60 flex items-center justify-center text-purple-300 text-[10px] shrink-0 mt-0.5">
                        <i class="fa-solid fa-robot"></i>
                    </div>
                    <div class="bg-slate-900 p-3.5 rounded-2xl rounded-tl-sm border border-slate-800 text-slate-200 leading-relaxed max-w-[90%]">
                        <div class="flex items-center justify-between text-[10px] text-purple-400 font-bold mb-1 border-b border-slate-800/80 pb-1">
                            <span>AI Wildfire Copilot</span>
                            <span class="text-[9px] text-slate-500 font-normal">${data.source}</span>
                        </div>
                        <div class="prose prose-invert text-xs space-y-1">
                            ${renderMarkdown(data.reply)}
                        </div>
                    </div>
                </div>
            `;
        } else {
            stream.innerHTML += `<div class="p-2 text-red-400 text-xs">Error: ${data.message}</div>`;
        }
    } catch (err) {
        const loadEl = document.getElementById(loadingId);
        if (loadEl) loadEl.remove();
        stream.innerHTML += `<div class="p-2 text-red-400 text-xs">Connection error: ${err.message}</div>`;
    } finally {
        if (sendBtn) sendBtn.disabled = false;
        stream.scrollTop = stream.scrollHeight;
    }
}

function sendQuickPrompt(promptText) {
    const input = document.getElementById('llm-chat-input');
    if (input) {
        input.value = promptText;
        handleChatSubmit();
    }
}

function clearChat() {
    const stream = document.getElementById('llm-chat-stream');
    if (stream) {
        stream.innerHTML = `
            <div class="flex items-start gap-2.5">
                <div class="w-6 h-6 rounded-md bg-purple-900/60 border border-purple-700/60 flex items-center justify-center text-purple-300 text-[10px] shrink-0 mt-0.5">
                    <i class="fa-solid fa-robot"></i>
                </div>
                <div class="bg-slate-900 p-3 rounded-2xl rounded-tl-sm border border-slate-800 text-slate-200 leading-relaxed">
                    <p class="font-semibold text-purple-300 text-[11px] mb-1">AI Wildfire Copilot</p>
                    Greetings! I am your AI Wildfire Disaster Copilot for Team 27's Forest Fire Risk Prediction System. You can ask me technical questions about our ML models (Random Forest, SVM, LR), feature explainability (TreeSHAP), topographical dynamics, or tactical disaster mitigation protocols.
                </div>
            </div>
        `;
    }
}

// Simple lightweight markdown parser for clean chat and report rendering
function renderMarkdown(md) {
    if (!md) return '';
    let html = escapeHtml(md);

    // Headers
    html = html.replace(/^### (.*$)/gim, '<h4 class="font-bold text-white text-sm mt-3 mb-1 text-purple-300 border-b border-slate-800 pb-1">$1</h4>');
    html = html.replace(/^#### (.*$)/gim, '<h5 class="font-bold text-amber-300 text-xs mt-2 mb-1">$1</h5>');

    // Bold & italic
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong class="font-bold text-white">$1</strong>');
    html = html.replace(/\*(.*?)\*/g, '<em class="italic text-slate-300">$1</em>');

    // Blockquotes
    html = html.replace(/^\> (.*$)/gim, '<blockquote class="border-l-2 border-purple-500 bg-purple-950/30 p-2 my-1 rounded text-purple-200 text-xs italic">$1</blockquote>');

    // Bullet lists
    html = html.replace(/^\* (.*$)/gim, '<li class="ml-4 list-disc text-slate-300">$1</li>');
    html = html.replace(/^- (.*$)/gim, '<li class="ml-4 list-disc text-slate-300">$1</li>');

    // Newlines
    html = html.replace(/\n\n/g, '<div class="h-2"></div>');
    html = html.replace(/\n/g, '<br>');

    return html;
}

function escapeHtml(str) {
    return str
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

