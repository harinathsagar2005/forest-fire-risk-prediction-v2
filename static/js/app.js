/**
 * Wildfire Risk Intelligence System
 * Scientific Client Logic & Data Visualization
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
    // 1. Initial prediction with baseline defaults
    await runPrediction();

    // 2. Initialize Scientific Charts
    initCharts();

    // 3. Load Dataset Sample
    loadDatasetSample();
}

// -------------------------------------------------------------
// TAB NAVIGATION
// -------------------------------------------------------------
function switchTab(tabId) {
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.add('hidden');
    });

    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });

    const targetTab = document.getElementById(tabId);
    if (targetTab) {
        targetTab.classList.remove('hidden');
    }

    const targetBtn = document.getElementById(`btn-${tabId}`);
    if (targetBtn) {
        targetBtn.classList.add('active');
    }

    // Leaflet map size recalculation
    if (tabId === 'tab-gis') {
        setTimeout(() => {
            if (!mapInstance) {
                initGisMap();
            } else {
                mapInstance.invalidateSize();
            }
        }, 120);
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
        btn.innerHTML = `<span>Calculating...</span>`;
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
            btn.innerHTML = `<span>Calculate Fire Risk Prediction</span>`;
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
    pill.innerText = pred.class_name;

    // Risk Score
    const riskPctEl = document.getElementById('result-risk-pct');
    riskPctEl.innerText = pred.risk_percentage.toFixed(1);

    const barEl = document.getElementById('result-risk-bar');
    barEl.style.width = `${Math.min(100, Math.max(4, pred.risk_percentage))}%`;

    // Minimal semantic styling
    if (pred.class_name === 'High Risk') {
        pill.className = 'inline-flex items-center px-3 py-1 rounded text-xs font-bold uppercase tracking-wide bg-red-100 text-red-700 border border-red-200 mb-2';
        riskPctEl.className = 'text-4xl font-bold tracking-tight text-red-600 font-mono';
        barEl.className = 'h-full rounded-full bg-red-600 transition-all duration-300';
    } else if (pred.class_name === 'Medium Risk') {
        pill.className = 'inline-flex items-center px-3 py-1 rounded text-xs font-bold uppercase tracking-wide bg-amber-100 text-amber-700 border border-amber-200 mb-2';
        riskPctEl.className = 'text-4xl font-bold tracking-tight text-amber-600 font-mono';
        barEl.className = 'h-full rounded-full bg-amber-500 transition-all duration-300';
    } else {
        pill.className = 'inline-flex items-center px-3 py-1 rounded text-xs font-bold uppercase tracking-wide bg-emerald-100 text-emerald-700 border border-emerald-200 mb-2';
        riskPctEl.className = 'text-4xl font-bold tracking-tight text-emerald-600 font-mono';
        barEl.className = 'h-full rounded-full bg-emerald-600 transition-all duration-300';
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
    rfEl.className = `font-bold ${getModelTextColor(allModels['Random Forest'].prediction)}`;

    const svmEl = document.getElementById('consensus-svm');
    svmEl.innerText = allModels['Support Vector Machine'].prediction;
    svmEl.className = `font-bold ${getModelTextColor(allModels['Support Vector Machine'].prediction)}`;

    const lrEl = document.getElementById('consensus-lr');
    lrEl.innerText = allModels['Logistic Regression'].prediction;
    lrEl.className = `font-bold ${getModelTextColor(allModels['Logistic Regression'].prediction)}`;

    // Standard Operating Protocols
    const sopContainer = document.getElementById('sop-content');
    sopContainer.innerHTML = sop.actions.map((action, idx) => `
        <div class="flex items-start gap-2">
            <span class="font-bold text-slate-800">${idx + 1}.</span>
            <span>${action}</span>
        </div>
    `).join('');

    // Feature Contribution Grid (TreeSHAP Approximation)
    const contribGrid = document.getElementById('feature-contributions-grid');
    contribGrid.innerHTML = contributions.slice(0, 8).map(item => {
        const isPositive = item.impact > 0;
        const color = isPositive ? 'text-red-600' : 'text-emerald-600';
        const sign = isPositive ? '+' : '';
        return `
            <div class="p-2.5 rounded bg-slate-50 border border-slate-200 flex flex-col justify-between">
                <div>
                    <span class="text-slate-500 block text-[11px] font-medium">${item.feature}</span>
                    <span class="text-slate-800 font-mono font-bold">${item.val} ${item.unit}</span>
                </div>
                <div class="mt-2 pt-1 border-t border-slate-200 flex items-center justify-between text-[11px]">
                    <span class="text-slate-400">Delta:</span>
                    <span class="font-mono font-bold ${color}">${sign}${item.impact.toFixed(1)} pts</span>
                </div>
            </div>
        `;
    }).join('');
}

function getModelTextColor(level) {
    if (level === 'High Risk') return 'text-red-600';
    if (level === 'Medium Risk') return 'text-amber-600';
    return 'text-emerald-600';
}

// -------------------------------------------------------------
// GIS LEAFLET MAP
// -------------------------------------------------------------
async function initGisMap() {
    const mapContainer = document.getElementById('gis-map');
    if (!mapContainer) return;

    // Center on Telangana Forest Reserve (Amrabad / Nallamala)
    mapInstance = L.map('gis-map').setView([16.35, 78.85], 8);

    // Standard Street Layer
    const osmLayer = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        maxZoom: 18,
        attribution: '&copy; OpenStreetMap'
    });

    // Topographic Layer
    const topoLayer = L.tileLayer('https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png', {
        maxZoom: 17,
        attribution: '&copy; OpenTopoMap'
    });

    // Satellite Layer
    const satLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
        maxZoom: 18,
        attribution: '&copy; Esri World Imagery'
    });

    osmLayer.addTo(mapInstance);

    const baseMaps = {
        "Street Map": osmLayer,
        "Satellite Imagery": satLayer,
        "Topographic": topoLayer
    };
    L.control.layers(baseMaps).addTo(mapInstance);

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
    mapMarkers.forEach(m => mapInstance.removeLayer(m));
    mapMarkers = [];

    zones.forEach(zone => {
        let color = '#16a34a'; // Green
        if (zone.risk_level === 'High Risk') color = '#dc2626'; // Red
        else if (zone.risk_level === 'Medium Risk') color = '#d97706'; // Amber

        const circle = L.circle([zone.lat, zone.lng], {
            color: color,
            fillColor: color,
            fillOpacity: 0.2,
            radius: 7500,
            weight: 1.5
        }).addTo(mapInstance);

        const marker = L.circleMarker([zone.lat, zone.lng], {
            radius: 7,
            fillColor: color,
            color: '#ffffff',
            weight: 2,
            opacity: 1,
            fillOpacity: 0.95
        }).addTo(mapInstance);

        const popupContent = `
            <div class="p-2.5 min-w-[200px] text-slate-800 font-sans text-xs">
                <div class="border-b border-slate-200 pb-1 mb-1.5">
                    <span class="text-[10px] font-bold text-slate-400 uppercase">${zone.id} &bull; ${zone.region}</span>
                    <h4 class="font-bold text-slate-900">${zone.name}</h4>
                </div>
                <div class="flex items-center justify-between mb-2">
                    <span class="font-bold text-white px-1.5 py-0.5 rounded text-[10px]" style="background-color: ${color}">
                        ${zone.risk_level} (${zone.risk_score}%)
                    </span>
                    <span class="text-[10px] font-semibold text-slate-600">${zone.status}</span>
                </div>
                <div class="grid grid-cols-2 gap-1 text-[11px] bg-slate-50 p-1.5 rounded border border-slate-100 mb-1.5">
                    <div>Temp: <strong>${zone.temperature}°C</strong></div>
                    <div>Humidity: <strong>${zone.humidity}%</strong></div>
                    <div>Wind: <strong>${zone.wind_speed} km/h</strong></div>
                    <div>NDVI: <strong>${zone.ndvi}</strong></div>
                </div>
                <p class="text-[10px] text-slate-500">
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
        let badgeClass = 'text-emerald-700 bg-emerald-50 border-emerald-200';
        if (zone.risk_level === 'High Risk') badgeClass = 'text-red-700 bg-red-50 border-red-200';
        else if (zone.risk_level === 'Medium Risk') badgeClass = 'text-amber-700 bg-amber-50 border-amber-200';

        return `
            <div onclick="flyToZone(${zone.lat}, ${zone.lng})" class="p-2.5 bg-slate-50 hover:bg-slate-100 rounded border border-slate-200 cursor-pointer transition flex items-center justify-between">
                <div>
                    <div class="flex items-center gap-1.5">
                        <span class="text-[10px] font-bold text-slate-400">${zone.id}</span>
                        <h4 class="text-xs font-semibold text-slate-800">${zone.name}</h4>
                    </div>
                    <p class="text-[10px] text-slate-500 mt-0.5">
                        ${zone.temperature}°C &bull; ${zone.wind_speed} km/h &bull; NDVI ${zone.ndvi}
                    </p>
                </div>
                <div class="text-right">
                    <span class="text-[10px] px-1.5 py-0.5 rounded border font-bold ${badgeClass}">
                        ${zone.risk_level}
                    </span>
                    <span class="block text-[10px] font-mono text-slate-500 mt-0.5">${zone.risk_score}%</span>
                </div>
            </div>
        `;
    }).join('');
}

function flyToZone(lat, lng) {
    if (mapInstance) {
        mapInstance.flyTo([lat, lng], 11, { duration: 1.2 });
    }
}

function resetMapCenter() {
    if (mapInstance) {
        mapInstance.flyTo([16.35, 78.85], 8, { duration: 1.0 });
    }
}

function triggerSimulatedEmergency() {
    alert("SIMULATED ADVISORY:\nHigh-velocity dry winds (38 km/h) & temperature surge detected in Sector Alpha (Amrabad Reserve).\nAutomated SOP protocols initialized.");
    if (cachedGisZones.length > 0) {
        flyToZone(cachedGisZones[0].lat, cachedGisZones[0].lng);
    }
}

// -------------------------------------------------------------
// CHARTS INITIALIZATION (Chart.js Minimalist)
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
                            label: 'Random Forest (95.0%)',
                            data: rfVals,
                            backgroundColor: '#0f172a',
                            borderRadius: 3
                        },
                        {
                            label: 'Support Vector Machine (86.5%)',
                            data: '#64748b',
                            borderRadius: 3
                        },
                        {
                            label: 'Logistic Regression (79.5%)',
                            data: '#94a3b8',
                            borderRadius: 3
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { labels: { color: '#334155', font: { size: 11, family: 'Inter' } } }
                    },
                    scales: {
                        y: {
                            min: 60,
                            max: 100,
                            ticks: { color: '#64748b', font: { family: 'monospace', size: 10 } },
                            grid: { color: '#f1f5f9' }
                        },
                        x: {
                            ticks: { color: '#334155', font: { weight: '600', size: 11 } },
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
                        backgroundColor: '#0f172a',
                        borderRadius: 3
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
                            ticks: { color: '#64748b', font: { family: 'monospace', size: 10 } },
                            grid: { color: '#f1f5f9' }
                        },
                        y: {
                            ticks: { color: '#334155', font: { size: 10, weight: '500' } },
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
                let badgeClass = 'text-emerald-700 bg-emerald-50 border-emerald-200';
                let label = 'Low Risk';
                if (row.risk_category === 2) {
                    badgeClass = 'text-red-700 bg-red-50 border-red-200';
                    label = 'High Risk';
                } else if (row.risk_category === 1) {
                    badgeClass = 'text-amber-700 bg-amber-50 border-amber-200';
                    label = 'Medium Risk';
                }

                return `
                    <tr class="hover:bg-slate-50 transition">
                        <td class="py-2 px-3 text-slate-800 font-medium">${row.temperature}°C</td>
                        <td class="py-2 px-3 text-slate-600">${row.relative_humidity}%</td>
                        <td class="py-2 px-3 text-slate-600">${row.rainfall}</td>
                        <td class="py-2 px-3 text-slate-600">${row.wind_speed}</td>
                        <td class="py-2 px-3 text-slate-600">${row.ndvi}</td>
                        <td class="py-2 px-3 text-slate-600">${row.slope}°</td>
                        <td class="py-2 px-3 text-slate-600">${row.elevation}m</td>
                        <td class="py-2 px-3 text-slate-600">${row.distance_to_road}m</td>
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
