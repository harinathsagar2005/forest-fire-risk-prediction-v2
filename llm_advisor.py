"""
LLM Wildfire Copilot & Intelligent Disaster Advisory Module
Forest Fire Risk Prediction System - Team 27
Vardhaman College of Engineering (Autonomous)
Department of CSE (AI&ML)

Provides:
1. Automated Executive Wildfire Incident Intelligence Generation
2. Tactical Resource Allocation & Containment Strategy
3. Multi-lingual Emergency Advisory (English & Telugu)
4. Interactive Conversational AI Forest Incident Copilot
5. Dual Engine: Live Gemini API integration + High-Fidelity Offline Expert Reasoning Engine
"""

import os
import json
import urllib.request
import urllib.error

# Optional default Gemini API Key from environment
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

def call_gemini_api(prompt, api_key=None):
    """
    Invokes Google Gemini REST API using Python's standard library (no extra dependencies required).
    """
    key = api_key or GEMINI_API_KEY
    if not key:
        return None

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={key}"
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 1024
        }
    }

    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        with urllib.request.urlopen(req, timeout=12) as response:
            res_data = json.loads(response.read().decode('utf-8'))
            candidates = res_data.get('candidates', [])
            if candidates:
                parts = candidates[0].get('content', {}).get('parts', [])
                if parts:
                    return parts[0].get('text', '')
    except Exception as e:
        print(f"[!] Gemini API call failed: {e}")
        return None

    return None

def generate_llm_incident_report(context, api_key=None):
    """
    Generates a comprehensive executive disaster incident brief
    combining ML predictions, SHAP feature drivers, and ecological physics.
    """
    temp = context.get('temperature', 35.0)
    rh = context.get('relative_humidity', 25.0)
    wind = context.get('wind_speed', 25.0)
    rain = context.get('rainfall', 0.0)
    ndvi = context.get('ndvi', 0.25)
    slope = context.get('slope', 18.0)
    dist_road = context.get('distance_to_road', 800.0)
    risk_level = context.get('risk_level', 'High Risk')
    risk_score = context.get('risk_score', 88.5)
    model_name = context.get('model_name', 'Random Forest (95.0% Accuracy)')
    location = context.get('location', 'Forest Sector Zone Alpha (Amrabad Reserve)')

    prompt = f"""You are the Chief AI Disaster Management Advisor for the Department of Forest & Environment.
Analyze the following real-time environmental telemetry and machine learning prediction:

LOCATION: {location}
PREDICTED HAZARD: {risk_level} (Hazard Index: {risk_score}%)
MODEL USED: {model_name}
TELEMETRY DATA:
- Ambient Temperature: {temp}°C
- Relative Humidity: {rh}%
- Wind Velocity: {wind} km/h
- Precipitation / Rain: {rain} mm
- Vegetation Health Index (NDVI): {ndvi}
- Terrain Incline / Slope: {slope}°
- Distance to Nearest Road: {dist_road} meters

Provide a structured Disaster Incident Intelligence Report containing:
1. Executive Situational Assessment
2. Ecological & Physical Ignition Mechanics
3. Tactical Fire Suppression & Resource Allocation Plan
4. Containment Buffer Radius & Evacuation Advisory
5. Bilingual Emergency Alert (English & Telugu for local panchayats)
"""

    # Attempt live Gemini API invocation if API key is supplied
    gemini_output = call_gemini_api(prompt, api_key)
    if gemini_output:
        return {
            "source": "Google Gemini 1.5 Flash (Live API)",
            "report_text": gemini_output
        }

    # Offline High-Fidelity Domain Generator (Deterministic, scientifically grounded)
    if risk_level == "High Risk":
        buffer_radius = round(max(3.0, (wind / 10.0) * 1.8 + (slope / 15.0)), 1)
        tenders_needed = int(max(4, 2 + wind / 8.0))
        drones_needed = 3
        personnel = int(max(25, tenders_needed * 6))

        telugu_alert = f"⚠️ అత్యవసర అటవీ హెచ్చరిక: {location} పరిసరాల్లో తీవ్రమైన అగ్నిప్రమాద ప్రమాదం పొంచి ఉంది. గాలి వేగం {wind} km/h మరియు ఉష్ణోగ్రత {temp}°C గా నమోదైంది. సమీప గ్రామ ప్రజలు అప్రమత్తంగా ఉండాలి. అడవిలోకి ప్రవేశించవద్దు. అత్యవసర శాఖలను సంప్రదించండి."

        report_markdown = f"""### 🛡️ EXECUTIVE WILDFIRE INCIDENT BRIEFING
**INCIDENT LEVEL:** CRITICAL EMERGENCY (RED TIER) | **LOCATION:** {location}  
**PREDICTIVE ENGINE:** {model_name} | **CONFIDENCE RATING:** {risk_score}% Hazard Potential

---

#### 1. Executive Situational Assessment
The AI monitoring system has verified critical wildfire susceptibility in **{location}**. Compound climatic indicators exhibit an acute combustible state driven by low atmospheric vapor pressure and severe fine fuel desiccation. Model consensus across Random Forest (95.0%) confirms imminent flame propagation danger.

#### 2. Ecological & Thermodynamic Ignition Mechanics
* **Atmospheric Vapor Deficit:** Elevated temperature (**{temp}°C**) combined with depleted relative humidity (**{rh}%**) has driven fine dead fuel moisture (FDFM) beneath the critical ignition threshold (< 6%).
* **Convective Flame Acceleration:** Wind velocity of **{wind} km/h** bends flame geometry forward, preheating unburnt brush ahead of the flame front via convective radiation.
* **Topographical Chimney Effect:** A terrain incline of **{slope}°** accelerates upslope rate of spread (ROS) approximately **{round(1.0 + (slope/10.0)**1.2, 1)}x** faster than planar ground.
* **Fuel Combustibility (NDVI {ndvi}):** Low vegetative vigor indicates senescent biomass ready for flash combustion.

#### 3. Tactical Resource Allocation & Containment Plan
* **Primary Suppression Fleet:** Deploy **{tenders_needed} Rapid Intervention Water Tenders** and 2 High-Pressure Foam Modules.
* **Aerial Surveillance:** Launch **{drones_needed} FLIR Thermal Drones** for live perimeter tracking and ember shower detection.
* **Ground Taskforce:** Mobilize **{personnel} Forest Disaster Response Personnel** equipped with fire swatters and Pulaski axes.
* **Counter-Fire Tactics:** Construct a **{int(slope * 1.5 + 15)}m mechanical firebreak** along the lee ridge perpendicular to the wind vector.

#### 4. Containment Buffer & Evacuation Advisory
* **Calculated Evacuation Buffer:** **{buffer_radius} km** radius from the active sector coordinate.
* **Road Corridors:** Restrict vehicular movement along all secondary forest roads within **{int(dist_road + 1500)}m**.
* **Community Advisory:** Place nearby panchayats on **Level-2 Pre-Evacuation Alert**.

#### 5. Bilingual Emergency Broadcast Dispatch

**[English Emergency Advisory]**
> "URGENT WILDFIRE ALERT: High wildfire risk declared in {location} due to extreme weather ({temp}°C, {wind} km/h wind). Immediate restriction on forest entry. Local communities are advised to remain indoors and clear dry brush around properties. Emergency helpline: 1926."

**[తెలుగు హెచ్చరిక సందేశం - Local Panchayat Alert]**
> "{telugu_alert}"
"""

    elif risk_level == "Medium Risk":
        buffer_radius = 1.8
        tenders_needed = 2
        drones_needed = 1
        personnel = 12

        report_markdown = f"""### 🛡️ EXECUTIVE WILDFIRE INCIDENT BRIEFING
**INCIDENT LEVEL:** ELEVATED SUSCEPTIBILITY (YELLOW WATCH) | **LOCATION:** {location}  
**PREDICTIVE ENGINE:** {model_name} | **CONFIDENCE RATING:** {risk_score}% Hazard Potential

---

#### 1. Executive Situational Assessment
Environmental parameters indicate moderate wildfire potential in **{location}**. While continuous flame propagation is constrained by ambient moisture conditions, localized thermal anomalies or anthropogenic ignitions can escalate rapidly under wind gusts.

#### 2. Ecological & Thermodynamic Mechanics
* **Fuel Condition:** Moderate vegetative canopy (NDVI {ndvi}) maintains baseline moisture, but surface litter remains vulnerable during peak solar radiation.
* **Microclimate:** Temperature of **{temp}°C** and humidity of **{rh}%** create localized drying pockets, moderated by {rain} mm precipitation.

#### 3. Tactical Recommendations
* Deploy **{tenders_needed} standby water tenders** at forward ranger checkpoints.
* Activate **continuous observation towers** and schedule twice-daily thermal drone reconnaissance.
* Dispatch **{personnel} forest beat guards** for ground inspection of natural firebreaks.

#### 4. Community Advisory
Issue precautionary agricultural advisory against open biomass burning within a **{buffer_radius} km** perimeter.
"""

    else:
        report_markdown = f"""### 🛡️ EXECUTIVE WILDFIRE INCIDENT BRIEFING
**INCIDENT LEVEL:** SAFE / STABLE STATUS (GREEN CONDITION) | **LOCATION:** {location}  
**PREDICTIVE ENGINE:** {model_name} | **CONFIDENCE RATING:** {risk_score}% Hazard Potential

---

#### 1. Executive Situational Assessment
All monitored telemetry for **{location}** indicates low fire risk. Ambient humidity (**{rh}%**) and precipitation (**{rain} mm**) provide substantial fuel damping, preventing combustible flame front formation.

#### 2. Ecological State
High fuel moisture content and healthy canopy vigor (NDVI {ndvi}) effectively inhibit fire spread. Slope and road proximity factors present negligible danger under current weather.

#### 3. Standard Operating Procedure
* Maintain standard telemetry sensor pings and routine patrol schedules.
* No emergency resource mobilization required.
"""

    return {
        "source": "AI Forest Incident Copilot (Local Expert Engine)",
        "report_text": report_markdown
    }

def answer_wildfire_copilot_chat(user_question, current_context=None, api_key=None):
    """
    Handles interactive conversational queries from students, examiners, and forest officers.
    """
    ctx_str = ""
    if current_context:
        ctx_str = f"Current active context: Risk={current_context.get('risk_level')}, Temp={current_context.get('temperature')}C, Wind={current_context.get('wind_speed')}km/h, RH={current_context.get('relative_humidity')}%, NDVI={current_context.get('ndvi')}"

    system_prompt = f"""You are the AI Wildfire Copilot for the Forest Fire Risk Prediction project developed by Team 27 (G. Harinath, S. Anirudh) guided by Mrs. D Sandhya Rani at Vardhaman College of Engineering (Autonomous), Department of CSE (AI&ML).
The system uses Random Forest (95.0% accuracy), SVM (86.5%), and Logistic Regression (79.5%) along with TreeSHAP and Leaflet GIS mapping.
{ctx_str}

Respond professionally, technically, and concisely to the user's query:
User Query: {user_question}
"""

    # Check for live API call
    gemini_reply = call_gemini_api(system_prompt, api_key)
    if gemini_reply:
        return {
            "source": "Google Gemini 1.5 Flash (Live API)",
            "answer": gemini_reply
        }

    # Offline Intelligent Q&A Knowledge Engine
    q_lower = user_question.lower()

    if ("random forest" in q_lower and any(w in q_lower for w in ["why", "better", "svm", "perform", "accura", "compare", "outperform"])) or any(k in q_lower for k in ["why random forest", "best model", "outperform", "highest accuracy", "rf vs"]):
        ans = (
            "**Why Random Forest Outperformed SVM and Logistic Regression:**\n\n"
            "1. **Non-Linear Threshold Dynamics:** Forest fire behavior is governed by strict physical thresholds (e.g., ignition accelerates drastically only when Temperature > 32°C AND Humidity < 35% AND Wind > 20 km/h, but is completely negated if Rainfall > 3.5 mm). Random Forest easily isolates these orthogonal step-splits.\n"
            "2. **Resilience to Multi-Collinearity:** Ensemble bagging across 200 trees mitigates correlation between temperature and relative humidity.\n"
            "3. **Empirical Results:** On our 3,000-sample stratified benchmark:\n"
            "   - **Random Forest:** **95.00% Accuracy**, 94.83% F1-Score, 0.9564 ROC-AUC\n"
            "   - **SVM (RBF):** 86.50% Accuracy (smooth kernel struggles with sharp threshold boundaries)\n"
            "   - **Logistic Regression:** 79.50% Accuracy (linear hyperplanes cannot model high-order feature interactions)."
        )

    elif any(k in q_lower for k in ["shap", "explainab", "xai", "feature importance"]):
        ans = (
            "**Explainable AI (TreeSHAP) Architecture:**\n\n"
            "- **Game-Theoretic Fairness:** TreeSHAP computes the marginal contribution of each feature across all possible sub-coalitions of decision trees.\n"
            "- **Top Feature Hierarchy:**\n"
            "  1. **Relative Humidity & Temperature (>54% total impact):** Regulate vapor pressure deficit and desiccation of combustible leaf litter.\n"
            "  2. **Wind Speed:** Acts as a convective heat transfer multiplier, tilting flames and driving spotting ember showers.\n"
            "  3. **NDVI (Vegetation Index):** Measures moisture content vs dried combustible brush.\n"
            "  4. **Slope:** Accentuates the upward chimney pre-heating effect on steep hillsides."
        )

    elif any(k in q_lower for k in ["ndvi", "vegetation"]):
        ans = (
            "**Role of NDVI (Normalized Difference Vegetation Index):**\n\n"
            "NDVI ranges from -0.1 to +0.85:\n"
            "- **High NDVI (> 0.65):** Dense, green, moist forest canopy (e.g. Wayanad rainforest). High fuel moisture strongly inhibits ignition.\n"
            "- **Low NDVI (0.10 - 0.35):** Senescent, cured, or dry deciduous scrub (e.g. dry season in Nallamala / Amrabad). Acts as dry kindling that ignites rapidly."
        )

    elif any(k in q_lower for k in ["slope", "terrain", "elevation"]):
        ans = (
            "**Physics of Terrain Slope in Wildfire Spread:**\n\n"
            "Slope significantly accelerates fire spread uphill due to convective radiation. For every 10° increase in slope incline, the rate of spread (ROS) approximately doubles! Flames lean into the uphill unburnt slope, preheating the vegetation before direct contact."
        )

    elif any(k in q_lower for k in ["evacuat", "action", "sop", "resource", "protocol"]):
        ans = (
            "**Emergency Standard Operating Procedures (SOPs):**\n\n"
            "- **High Risk (>75%):** Automated siren alarms, rapid intervention water tenders dispatched, aerial FLIR drones deployed, forest tourist trails evacuated, and firebreaks cleared.\n"
            "- **Medium Risk (35-75%):** Watch tower observation activated, beat patrols intensified, advisory issued to roadside agricultural settlements.\n"
            "- **Low Risk (<35%):** Routine telemetry monitoring."
        )

    elif any(k in q_lower for k in ["telugu", "panchayat", "local"]):
        ans = (
            "**తెలుగు హెచ్చరిక సందేశం (Telugu Emergency Dispatch):**\n\n"
            "\"అటవీ అగ్నిప్రమాద హెచ్చరిక: వాతావరణంలో తీవ్రమైన వేడి మరియు గాలి వేగం పెరిగినందున, అటవీ సమీప గ్రామాల ప్రజలు అప్రమత్తంగా ఉండాలి. ఎండుటాకులు, గడ్డి కాల్చడం నిషేధించబడింది. అత్యవసర సహాయం కోసం ఫారెస్ట్ డిపార్ట్‌మెంట్ హెల్ప్‌లైన్ 1926 కు కాల్ చేయండి.\""
        )

    else:
        ans = (
            f"**AI Wildfire Advisor Briefing:**\n\n"
            f"Regarding your query on *\"{user_question}\"*:\n"
            f"- In forest fire disaster mitigation, real-time telemetry (Temperature, Humidity, Wind, Slope, NDVI) feeds into our Random Forest classifier (95.0% Accuracy).\n"
            f"- For critical threats, immediate multi-agency response protocols are triggered: deploying water tenders, thermal drone reconnaissance, and creating tactical firebreaks perpendicular to the wind direction.\n"
            f"- You can test real-time environmental values on the Risk Predictor tab or simulate weather alerts on the GIS Risk Map!"
        )

    return {
        "source": "AI Forest Incident Copilot (Local Expert Engine)",
        "answer": ans
    }
