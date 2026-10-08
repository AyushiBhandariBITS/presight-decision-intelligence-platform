import numpy as np
import pandas as pd

def generate_abu_dhabi_chronological_data(year=2026, seed=42):
    np.random.seed(seed)
    
    # 1. SETUP CHRONOLOGICAL SPACE-TIME MATRIX
    # Generate every single hour for a full 365-day year
    date_range = pd.date_range(start=f"{year}-01-01 00:00:00", end=f"{year}-12-31 23:00:00", freq='H')
    
    # Define Abu Dhabi specific geographic corridors
    corridors = {
        'Sheikh Zayed Bridge (Inbound)': {'type': 'Bridge_Inbound', 'base_risk': 0.4},
        'Maqta Bridge (Inbound)': {'type': 'Bridge_Inbound', 'base_risk': 0.5},
        'E11 Highway (Abu Dhabi-Dubai Main Arterial)': {'type': 'Intercity_Highway', 'base_risk': 0.3},
        'E22 Highway (Abu Dhabi-Al Ain Arterial)': {'type': 'Intercity_Highway', 'base_risk': 0.35},
        'Hamdan Street Grid (Downtown Island)': {'type': 'Island_Grid', 'base_risk': 0.6}
    }
    
    rows = []
    
    # 2. GENERATE DATA CHRONOLOGICALLY HOUR-BY-HOUR
    for ts in date_range:
        month = ts.month
        day_of_week = ts.dayofweek # 0 = Monday, 6 = Sunday
        hour = ts.hour
        # Cyclic features for Time of Day
        time_of_day_sin = np.sin(2 * np.pi * hour / 24)
        time_of_day_cos = np.cos(2 * np.pi * hour / 24)
        
        # --- CLIMATIC RULES ---
        # Rule 1: Winter Morning Fog (Dec, Jan, Feb between 4 AM - 9 AM)
        is_winter_morning = (month in [12, 1, 2]) and (4 <= hour <= 9)
        if is_winter_morning:
            # High probability of severe fog
            weather_severity = np.random.choice([np.random.uniform(7.5, 10.0), np.random.uniform(0.0, 2.0)], p=[0.4, 0.6])
        else:
            # Rule 3: Shamal Sandstorm Shocks (Occasional random atmospheric dust)
            if np.random.rand() < 0.02: 
                weather_severity = np.random.uniform(5.0, 8.0)
            else:
                weather_severity = np.random.beta(a=1, b=8) * 10 # Default clear desert skies
                
        # Rule 2: Summer Pavement Extreme Heat (June to Sept, peaks mid-afternoon)
        if month in [6, 7, 8, 9]:
            base_temp = 38 + 10 * np.sin(2 * np.pi * (hour - 6) / 24) # Peak heat at ~3-4 PM
            pavement_temp = base_temp + np.random.normal(5, 1.5)
        else:
            base_temp = 20 + 8 * np.sin(2 * np.pi * (hour - 6) / 24)
            pavement_temp = base_temp + np.random.normal(2, 1.0)
            
        # Loop through each corridor to capture geographic differences at this exact hour
        for corridor_name, properties in corridors.items():
            corr_type = properties['type']
            historical_base_rate = properties['base_risk'] + np.random.normal(0, 0.02)
            historical_base_rate = np.clip(historical_base_rate, 0.05, 0.95)
            
            # --- SOCIO-ECONOMIC RUNTIME RULES ---
            # Rule 6: Heavy Vehicle Ratio (KIZAD freight routes)
            if corr_type == 'Intercity_Highway':
                # Freight moves heavily late at night due to daytime inner-city bans
                heavy_ratio = 0.25 if (hour >= 22 or hour <= 5) else 0.08
            elif corr_type == 'Bridge_Inbound':
                heavy_ratio = 0.12 if (hour >= 22 or hour <= 5) else 0.02
            else: # Downtown Island Grid
                heavy_ratio = np.random.uniform(0.01, 0.03) # Strictly restricted
            heavy_ratio += np.random.uniform(0.0, 0.04)
            
            # Rule 4 & 5: UAE 4.5-Day Work Week Traffic Flow Profiling
            is_rush_hour = False
            # Monday - Thursday (Full working days)
            if day_of_week in [0, 1, 2, 3]:
                if (7 <= hour <= 9) or (16 <= hour <= 18):
                    is_rush_hour = True
            # Friday (Half day ending at noon)
            elif day_of_week == 4:
                if (7 <= hour <= 9) or (11 <= hour <= 13):
                    is_rush_hour = True
                    
            # Rule 5: Inter-Emirate Super Commutes start earlier on long highways
            is_highway_early_rush = (corr_type == 'Intercity_Highway') and (day_of_week <= 4) and (6 <= hour <= 7)
            
            # --- MEDIATORS GENERATION: TRAFFIC DENSITY ---
            # Rule 4: Bridge Bottlenecks capacity ceiling
            if is_rush_hour:
                density_mean = 120 if corr_type == 'Bridge_Inbound' else 95
            elif is_highway_early_rush:
                density_mean = 110
            elif day_of_week >= 5 and (18 <= hour <= 22): # Weekend evening leisure rush (Yas Island/Malls)
                density_mean = 80
            else: # Off-peak
                density_mean = 30 + 10 * time_of_day_sin
                
            # Weather slows down and packs vehicles together
            density_mean += (weather_severity * 2.5)
            traffic_density = np.random.normal(density_mean, 6)
            
            # Hard physical capping limits based on infrastructure geometry
            if corr_type == 'Bridge_Inbound':
                traffic_density = np.clip(traffic_density, 10, 140) # Hard bottle-neck ceiling
            elif corr_type == 'Intercity_Highway':
                traffic_density = np.clip(traffic_density, 5, 110) # Absorbs density better via multiple lanes
            else:
                traffic_density = np.clip(traffic_density, 5, 90)

            # --- TREATMENTS & INTERVENTIONS (Causal Bias Injection) ---
            # Rule 1 & Advanced Feature Engineering: Confounder Bias Injection
            # Variable Speed Signs dropped automatically if weather severity exceeds 7 (Smart Grid)
            speed_drop_logit = 3.5 * (weather_severity - 7.0) + np.random.normal(0, 0.8)
            speed_drop_prob = 1 / (1 + np.exp(-speed_drop_logit))
            speed_limit_drop_applied = np.random.binomial(1, speed_drop_prob)
            
            # Transit Flow Sync: Prioritized on the Island Grid during daytime peak commutes
            transit_logit = 2.0 * time_of_day_cos if (corr_type == 'Island_Grid' and is_rush_hour) else -2.5
            transit_sync_prob = 1 / (1 + np.exp(transit_logit + np.random.normal(0, 0.2)))
            transit_flow_sync_active = np.random.binomial(1, transit_sync_prob)
            
            # Nav Detour Alerts: Triggered during extreme congestion levels or heavy fog closures
            detour_logit = 0.08 * (traffic_density - 100) + 0.5 * weather_severity - 2.0
            detour_prob = 1 / (1 + np.exp(-detour_logit))
            nav_detour_triggered = np.random.binomial(1, detour_prob)
            
            # Apply intervention modifiers to traffic mechanics
            if nav_detour_triggered == 1:
                traffic_density = max(5, traffic_density - np.random.uniform(15, 25))
            
            # --- MEDIATORS GENERATION: SPEED VARIANCE ---
            # Rule 2, 4, 6: Factors driving raw variance
            variance_mean = 8 + (0.8 * weather_severity) + (pavement_temp * 0.05)
            
            if corr_type == 'Intercity_Highway':
                # Rule 4: At 140 km/h, structural variance is naturally higher due to truck/car speed differentials
                variance_mean += 6.0 + (heavy_ratio * 25)
            else:
                variance_mean += 2.0 + (heavy_ratio * 10)
                
            # Interventions actively suppress variance (especially dropping speed limit down to uniform 80 km/h)
            variance_modifier = -7.5 * speed_limit_drop_applied - 1.5 * transit_flow_sync_active
            average_speed_variance = variance_mean + variance_modifier + np.random.normal(0, 1.2)
            average_speed_variance = np.clip(average_speed_variance, 1.5, 45.0)

            # --- OUTCOME GENERATION: ACCIDENT RISK SCORE ---
            # Baseline infrastructural engineering risk
            risk = 15 + (45 * historical_base_rate)
            
            # Causal structural updates from mechanisms
            risk += (0.15 * traffic_density) + (1.4 * average_speed_variance)
            
            # Confounder environmental direct penalties
            risk += (2.2 * weather_severity) + (0.12 * max(0, pavement_temp - 55)) # Tire blowout multiplier
            
            # Advanced Feature Engineering: Interaction Scaling Rule
            # Variable speed limit drops drastically lower risk in hazardous conditions, but have neutral/minor impact on clear days
            interaction_effect = -2.8 * (weather_severity * speed_limit_drop_applied)
            risk += interaction_effect
            
            # Other residual policy treatments
            risk += -4.0 * nav_detour_triggered
            
            # Output generation with continuous mapping limits
            accident_risk_score = risk + np.random.normal(0, 2.5)
            accident_risk_score = np.clip(accident_risk_score, 0.0, 100.0)
            
            # Collect metrics row by row
            rows.append({
                'Timestamp': ts,
                'Corridor_Name': corridor_name,
                'Corridor_Type': corr_type,
                'Month': month,
                'Day_of_Week': day_of_week,
                'Hour': hour,
                'Pavement_Temperature_C': np.round(pavement_temp, 1),
                'Heavy_Vehicle_Ratio': np.round(heavy_ratio, 3),
                
                # REQ 1: Context & Confounders
                'Weather_Severity_Index': np.round(weather_severity, 2),
                'Time_of_Day_Sin': np.round(time_of_day_sin, 4),
                'Time_of_Day_Cos': np.round(time_of_day_cos, 4),
                'Historical_Accident_Base_Rate': np.round(historical_base_rate, 4),
                
                # REQ 2: Interventions & Treatments
                'Speed_Limit_Drop_Applied': speed_limit_drop_applied,
                'Transit_Flow_Sync_Active': transit_flow_sync_active,
                'Nav_Detour_Triggered': nav_detour_triggered,
                
                # REQ 3: Mediators
                'Traffic_Density_Veh_Km': np.round(traffic_density, 2),
                'Average_Speed_Variance': np.round(average_speed_variance, 2),
                # REQ 4: Outcome Target
                'Accident_Risk_Score': np.round(accident_risk_score, 2)
                })
                # Convert matrix array to structured pandas DataFrame
    df = pd.DataFrame(rows)
                # Ensure explicit chronological execution sorting
    df = df.sort_values(by=['Timestamp', 'Corridor_Name']).reset_index(drop=True)
    return df
            #    --- EXECUTE SIMULATION GENERATOR ---
if __name__ == '__main__':
    print("Initializing chronological Abu Dhabi realistic simulation...")
    traffic_dataset = generate_abu_dhabi_chronological_data(year=2026)
    # Save directly out to local storage
    output_filename = "abu_dhabi_traffic_operations_2026.csv"
    traffic_dataset.to_csv(output_filename, index=False)
    print(f"Generation successful! Saved data file to: {output_filename}")
    print(f"Total Rows Generated: {len(traffic_dataset)} (8,760 hours * 5 distinct corridors)")
    print("\nSample Slice of Chronological Outputs:")
    print(traffic_dataset.head(10))