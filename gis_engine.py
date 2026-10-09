"""
GIS Risk Mapping and Spatial Analytics Module
Forest Fire Risk Prediction - Team 27
Vardhaman College of Engineering (Autonomous)
"""

import json
import numpy as np

def get_gis_forest_zones():
    """
    Returns spatial GIS zones, sensor stations, and risk susceptibility coordinates
    covering key forest regions (Telangana Amrabad Tiger Reserve, Nallamala, Kawal, Western Ghats).
    """
    zones = [
        {
            "id": "ZONE-01",
            "name": "Amrabad Tiger Reserve - Sector Alpha",
            "region": "Telangana / Nallamala Range",
            "lat": 16.3820,
            "lng": 78.8350,
            "elevation": 640,
            "slope": 22.4,
            "ndvi": 0.28,
            "temperature": 38.5,
            "humidity": 22,
            "wind_speed": 28.5,
            "rainfall": 0.0,
            "distance_to_road": 650,
            "risk_level": "High Risk",
            "risk_score": 92.4,
            "status": "CRITICAL ALERT",
            "recommended_action": "Deploy rapid response water tender and activate aerial drone patrol."
        },
        {
            "id": "ZONE-02",
            "name": "Nallamala Dense Canopy - Sector Bravo",
            "region": "Telangana / Srisailam Border",
            "lat": 16.0745,
            "lng": 78.8710,
            "elevation": 720,
            "slope": 18.2,
            "ndvi": 0.42,
            "temperature": 33.2,
            "humidity": 38,
            "wind_speed": 18.0,
            "rainfall": 0.0,
            "distance_to_road": 1800,
            "risk_level": "Medium Risk",
            "risk_score": 64.8,
            "status": "WATCH ZONE",
            "recommended_action": "Continuous lookout tower monitoring; inspect fire break lines."
        },
        {
            "id": "ZONE-03",
            "name": "Kawal Tiger Reserve - Buffer Range",
            "region": "Telangana (Mancherial)",
            "lat": 19.1250,
            "lng": 78.9650,
            "elevation": 310,
            "slope": 7.5,
            "ndvi": 0.65,
            "temperature": 27.8,
            "humidity": 68,
            "wind_speed": 11.2,
            "rainfall": 12.4,
            "distance_to_road": 3200,
            "risk_level": "Low Risk",
            "risk_score": 18.3,
            "status": "SAFE / NORMAL",
            "recommended_action": "Standard routine sensor health ping; minimal fire spread risk."
        },
        {
            "id": "ZONE-04",
            "name": "Eturnagaram Wildlife Sanctuary",
            "region": "Telangana (Mulugu)",
            "lat": 18.3360,
            "lng": 80.4320,
            "elevation": 240,
            "slope": 14.1,
            "ndvi": 0.35,
            "temperature": 35.8,
            "humidity": 28,
            "wind_speed": 22.4,
            "rainfall": 0.2,
            "distance_to_road": 1100,
            "risk_level": "High Risk",
            "risk_score": 84.7,
            "status": "ELEVATED ALERT",
            "recommended_action": "Pre-position fire suppression crews and notify forest beat officers."
        },
        {
            "id": "ZONE-05",
            "name": "Pakhal Lake Wildlife Corridor",
            "region": "Telangana (Warangal)",
            "lat": 17.9620,
            "lng": 79.9810,
            "elevation": 290,
            "slope": 9.8,
            "ndvi": 0.58,
            "temperature": 29.4,
            "humidity": 52,
            "wind_speed": 14.0,
            "rainfall": 2.8,
            "distance_to_road": 2100,
            "risk_level": "Medium Risk",
            "risk_score": 45.2,
            "status": "MODERATE MONITORING",
            "recommended_action": "Maintain routine satellite thermal anomaly tracking."
        },
        {
            "id": "ZONE-06",
            "name": "Bandipur Forest Edge - Western Ghats",
            "region": "Karnataka / Tamil Nadu Junction",
            "lat": 11.6664,
            "lng": 76.6291,
            "elevation": 1020,
            "slope": 26.5,
            "ndvi": 0.22,
            "temperature": 37.2,
            "humidity": 21,
            "wind_speed": 31.0,
            "rainfall": 0.0,
            "distance_to_road": 450,
            "risk_level": "High Risk",
            "risk_score": 95.8,
            "status": "CRITICAL ALERT",
            "recommended_action": "Trigger emergency fire alarm; mobilize ground counter-fire squads."
        },
        {
            "id": "ZONE-07",
            "name": "Wayanad Evergreen Biosphere",
            "region": "Kerala Rainforest Zone",
            "lat": 11.6854,
            "lng": 76.1320,
            "elevation": 890,
            "slope": 16.0,
            "ndvi": 0.78,
            "temperature": 24.1,
            "humidity": 79,
            "wind_speed": 8.5,
            "rainfall": 28.5,
            "distance_to_road": 3800,
            "risk_level": "Low Risk",
            "risk_score": 11.2,
            "status": "SAFE / NORMAL",
            "recommended_action": "No intervention needed; high fuel moisture inhibits combustion."
        },
        {
            "id": "ZONE-08",
            "name": "Ananthagiri Hills Forest Reserve",
            "region": "Telangana (Vikarabad)",
            "lat": 17.3110,
            "lng": 77.8630,
            "elevation": 670,
            "slope": 12.3,
            "ndvi": 0.48,
            "temperature": 31.5,
            "humidity": 44,
            "wind_speed": 16.5,
            "rainfall": 0.0,
            "distance_to_road": 1300,
            "risk_level": "Medium Risk",
            "risk_score": 53.6,
            "status": "WATCH ZONE",
            "recommended_action": "Periodic thermal monitoring and forest patrol check-in."
        }
    ]
    return zones

if __name__ == '__main__':
    data = get_gis_forest_zones()
    print(f"Loaded {len(data)} GIS forest monitoring zones.")
