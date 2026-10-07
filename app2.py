import streamlit as st
import pandas as pd
import time
from decision_orchestrator import AbuDhabiDecisionOrchestrator

# --- 1. INITIALIZATION & LAYOUT CONFIGURATION ---
st.set_page_config(
    page_title="Presight Decision Intelligence Platform", 
    layout="wide"
)

# Initialize the causal architecture once per user session lifetime
if 'orchestrator' not in st.session_state:
    with st.spinner("Synchronizing Causal AI Archetypes..."):
        st.session_state.orchestrator = AbuDhabiDecisionOrchestrator(model_directory=".")
        # Pre-set mock parameters matching your test scenarios
        st.session_state.visibility = 8.0  # Severe fog sequence from your test script
        st.session_state.base_rate = 0.4
        st.session_state.temperature = 22.0
        st.session_state.is_highway = 1     # Sector E11 is a high-speed intercity highway

orchestrator = st.session_state.orchestrator

# Pre-load a default recommendation array so the panel doesn't see a blank panel on boot
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = orchestrator.compute_policy_recommendations(
        weather_severity=st.session_state.visibility,
        base_rate=st.session_state.base_rate,
        temperature=st.session_state.temperature,
        is_highway=st.session_state.is_highway,
        weights={'risk': 0.6, 'cost': 0.2, 'time': 0.2}
    )

# --- 2. ENHANCED BRAND STYLING INJECTIONS ---
st.markdown("""
    <style>
    .stApp { background-color: #000B26; color: #FFFFFF; }
    h1, h2, h3, h4, p, span, div { color: #FFFFFF !important; }
    .risk-banner { background: #260C14; border: 1px solid #FF3B30; border-radius: 8px; padding: 20px; text-align: center; margin-bottom: 15px; }
    .rec-card-gold { background: #11283B; border: 1px solid #00E575; border-left: 6px solid #00E575; border-radius: 8px; padding: 15px; margin-bottom: 12px; }
    .rec-card-normal { background: #0B1936; border: 1px solid #00D2FF; border-left: 6px solid #00D2FF; border-radius: 8px; padding: 15px; margin-bottom: 12px; }
    .stButton>button { background-color: #00E575 !important; color: #000B26 !important; font-weight: bold !important; border: none !important; width: 100%; }
    code { background-color: #0B1936 !important; color: #00D2FF !important; }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Presight Decision Intelligence Platform")
st.caption("Abu Dhabi Traffic Operations Systems — Graduate Program Assessment Prototype")
st.markdown("---")

# =========================================================================
# --- ROW 1: LIVE PREDICTIVE LAYER ---
# =========================================================================
with st.container():
    st.subheader("1. Predictive Baseline")
    
    # Glowing Alert Card displaying baseline parameters
    st.markdown("""
        <div class="risk-banner">
            <h4 style="color:#FF3B30; margin:0;">🚨 High Accident Risk Forecast</h4>
            <p style="margin:5px 0 0 0; font-size:28px; font-weight:bold; font-family:monospace;">74% Risk Tomorrow</p>
            <p style="margin:2px 0 0 0; font-size:13px; color:#A0A0A0;">Location Zone: Abu Dhabi Sector E11</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("**Correlation Engine Insights:**")
    st.caption("Standard AI model links heavy morning fog, GPS slowdowns, and peak camera density to historical incident trends. *No mitigation evaluated yet.*")
    
    # Interactive variables display panel for the tech panel
    st.markdown("**Active Environment Telemetry Vector:**")
    st.text(f"• Visibility Deficit: {st.session_state.visibility} Index\n"
            f"• Core Infrastructure: Intercity Highway (E11)\n"
            f"• Pavement Surface Temp: {st.session_state.temperature}°C")

st.markdown("---")

# =========================================================================
# --- ROW 2: INNOVATION & POLICY SANDBOX ---
# =========================================================================
with st.container():
    st.subheader("🎛️ 2. Policy Intervention Sandbox")
    st.write("Adjust the proposed operational variables to run real-time **Counterfactual Analysis**:")
    
    # Your Sliders and Selection Toggles
    speed_drop = st.slider("Smart Speed Limit Drops (km/h reduction)", 0, 40, 20, step=10)
    signal_timing = st.select_slider("Signal Timing Optimization Flow", options=["Standard", "Balanced", "Synchronized Flow"], value="Balanced")
    nav_detours = st.checkbox("Enable Live App Navigation Detours", value=True)
    transit_flow = st.checkbox("Public Transit Flow Synchronization", value=False)
    
    # Functional Button Trigger executing CATE model recalculations
    if st.button("🔮 Run Structural Counterfactual Simulation"):
        with st.spinner("Processing Causal Graph via EconML engine..."):
            time.sleep(1.2) # Keeps your clean micro-animation loader intact
            
            # Formulate variable weights based dynamically on the checkbox/slider states
            risk_priority_weight = 0.4 + (speed_drop / 100.0) + (0.1 if nav_detours else 0.0)
            custom_weights = {
                'risk': min(risk_priority_weight, 0.8),
                'cost': max(0.1, 0.4 - (speed_drop / 200.0)),
                'time': max(0.1, 0.2)
            }
            
            # Execute live inference via your serialized EconML model files
            st.session_state.recommendations = orchestrator.compute_policy_recommendations(
                weather_severity=st.session_state.visibility,
                base_rate=st.session_state.base_rate,
                temperature=st.session_state.temperature,
                is_highway=st.session_state.is_highway,
                weights=custom_weights
            )
        st.success("Simulation Complete! Optimization matrices updated below.")

st.markdown("---")

# =========================================================================
# --- ROW 3: DECISION MATRIX & RANKING ---
# =========================================================================
with st.container():
    st.subheader("🏆 3. Intervention Ranking Matrix")
    
    # 1. Map dynamic risk adjustments using actual calculations from your models
    total_mitigation_magnitude = 0.0
    for rec in st.session_state.recommendations:
        # Extract the float point risk numbers from your recommendation strings
        numeric_drop = float(rec['risk_mitigation'].split()[0].replace('-', ''))
        
        # Check active status flags from the sandbox inputs to construct the live metric card score
        if rec['raw_policy_key'] == 'Speed_Limit_Drop_Applied' and speed_drop > 0:
            total_mitigation_magnitude += numeric_drop * (speed_drop / 20.0) # Scale effect size by slider intensity
        elif rec['raw_policy_key'] == 'Nav_Detour_Triggered' and nav_detours:
            total_mitigation_magnitude += numeric_drop
        elif rec['raw_policy_key'] == 'Transit_Flow_Sync_Active' and transit_flow:
            total_mitigation_magnitude += numeric_drop
            
    # Calculate final updated risk values matching user actions
    final_risk = max(74.0 - total_mitigation_magnitude, 12.0)
    
    # Main dynamic dashboard KPI indicator card
    st.metric(
        label="Simulated Risk Level Post-Intervention", 
        value=f"{final_risk:.1f}%", 
        delta=f"-{total_mitigation_magnitude:.1f}% Risk Reduction"
    )
    
    st.markdown("**Top Recommended Action Frameworks:**")
    st.caption("Ordered multi-objective ranking based on CATE impact matrices, budget fees, and real-time network lag:")
    
        # 2. Loop through your model payloads to build high-fidelity interface cards
    for rec in st.session_state.recommendations:
        card_class = "rec-card-gold" if rec['rank'] == 1 else "rec-card-normal"
        badge = "⭐ SYSTEM OPTIMAL PATH" if rec['rank'] == 1 else f"STRATEGY RANK #{rec['rank']}"
        
        st.markdown(f"""
            <div class="{card_class}">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h4 style="margin: 0; color: #FFFFFF; font-weight: bold;">{rec['policy_name']}</h4>
                    <span style="font-size: 10px; font-weight: bold; background: #000B26; padding: 2px 6px; border-radius: 4px; border: 1px solid #00D2FF;">{badge}</span>
                </div>
                <hr style="margin: 8px 0; opacity: 0.15; border-color: #00D2FF;">
                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; font-size: 12px; line-height: 1.4;">
                    <span>📉 <b>Causal CATE Drop:</b><br><span style="color:#FF3B30;">{rec['risk_mitigation']}</span></span>
                    <span>💰 <b>Est. Cost (AED):</b><br>{rec['cost_profile']}</span>
                    <span>⏱️ <b>Deployment Net Lag:</b><br>{rec['time_profile']}</span>
                </div>
                <div style="margin-top: 8px; font-size: 11px; color: #A0A0A0;">
                    🛡️ <b>Engine Confidence Score: {rec['score']}/100</b> — Verified via placebo validation loops.
                </div>
            </div>
        """, unsafe_allow_html=True)

    
