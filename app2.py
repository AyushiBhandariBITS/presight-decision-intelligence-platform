import streamlit as st
import time
import pandas as pd

st.set_page_config(layout="wide")
st.title("Decision Casual Intelligence Demo")

# -----------------------------
# SCENARIOS + INTERVENTIONS
# -----------------------------

scenarios = {
    "Urban Traffic Safety": {
        "description": "Analyzing accident risk based on lighting, speed, congestion, and weather.",
        "base_risk": 84,
        "interventions": {
            "Improve Lighting": {
                "delta": -22,
                "pros": [
                    "Better visibility in fog and rain\n",
                    "- Reduces accident probability by ~10-22%\n"
                ],
                "cons": [
                    "Increases energy cost by ~2%\n",
                    "- Requires maintenance budget\n"
                ]
            },
            "Reduce Speed Limit": {
                "delta": -18,
                "pros": [
                    "Reduces high-speed collisions\n",
                    "- Improves pedestrian safety\n"
                ],
                "cons": [
                    "May increase travel time\n",
                    "- Requires enforcement\n"
                ]
            },
            "Increase Bus Frequency": {
                "delta": -13,
                "pros": [
                    "Reduces private vehicle usage\n",
                    "- Lowers congestion by ~8%\n"
                ],
                "cons": [
                    "Increases fuel usage by ~5%\n",
                    "- Adds operational cost\n"
                ]
            }
        },
        "best_intervention":"Improve Lighting",
        "best_score":62,
        "factors": ["Poor Lighting", "High Speed", "Rain", "Congestion"]
    },

    "Hospital Capacity Management Issue": {
        "description": "Analyzing ICU overload risk based on admissions, staffing, and discharge delays.",
        "base_risk": 78,
        "interventions": {
            "Add Night-Shift Staff": {
                "delta": -21,
                "pros": [
                    "Reduces patient wait time\n",
                    "- Improves ICU throughput\n"
                ],
                "cons": [
                    "Higher staffing cost\n",
                    "- Training required\n"
                ]
            },
            "Accelerate Discharge Process": {
                "delta": -17,
                "pros": [
                    "Frees up beds faster\n",
                    "- Reduces ICU bottlenecks\n"
                ],
                "cons": [
                    "Requires coordination\n",
                    "- Risk of premature discharge\n"
                ]
            },
            "Deploy Mobile ICU Units": {
                "delta": -12,
                "pros": [
                    "Adds emergency capacity\n",
                    "- Useful during surges\n"
                ],
                "cons": [
                    "Very high cost\n",
                    "- Logistical complexity\n"
                ]
            }
        },
        "best_intervention":"Add Night-Shift Staff",
        "best_score":67,
        "factors": ["High Admissions", "Low Staffing", "ICU Overload", "Delayed Discharges"]
    },

    "Energy Load Balancing": {
        "description": "Analyzing grid overload risk based on demand peaks, reserve margins, and weather.",
        "base_risk": 81,
        "interventions": {
            "Demand Response Program": {
                "delta": -23,
                "pros": [
                    "Flattens peak demand\n",
                    "- Reduces overload risk significantly\n"
                ],
                "cons": [
                    "Requires customer participation\n",
                    "- May reduce comfort levels\n"
                ]
            },
            "Activate Backup Generators": {
                "delta": -16,
                "pros": [
                    "Instant reserve capacity\n",
                    "- Stabilizes grid during peaks\n"
                ],
                "cons": [
                    "High fuel cost\n",
                    "- Environmental impact\n"
                ]
            },
            "Increase Solar Utilization": {
                "delta": -11,
                "pros": [
                    "Reduces reliance on fossil fuels\n",
                    "- Adds daytime capacity\n"
                ],
                "cons": [
                    "Weather dependent\n",
                    "- Requires storage systems\n"
                ]
            }
        },
        "best_intervention":"Demand Response Program",
        "best_score":58,
        "factors": ["Peak Demand", "Low Reserve", "Heat Wave", "Grid Constraints"]
    },

    "Water Distribution": {
        "description": "Analyzing water shortage risk based on consumption, leakage, and supply variability.",
        "base_risk": 74,
        "interventions": {
            "Fix Pipeline Leaks": {
                "delta": -20,
                "pros": [
                    "Reduces water loss by ~15%\n",
                    "- Improves pressure stability\n"
                ],
                "cons": [
                    "Requires field crews\n",
                    "- High repair cost\n"
                ]
            },
            "Smart Metering": {
                "delta": -14,
                "pros": [
                    "Detects abnormal usage\n",
                    "- Improves demand forecasting\n"
                ],
                "cons": [
                    "Installation cost\n",
                    "- Requires customer adoption\n"
                ]
            },
            "Increase Reservoir Release": {
                "delta": -10,
                "pros": [
                    "Instant supply boost\n",
                    "- Reduces shortage risk\n"
                ],
                "cons": [
                    "Depletes reserves\n",
                    "- Not sustainable long-term\n"
                ]
            }
        },
        "best_intervention":"Fix Pipeline Leaks",
        "best_score":54,
        "factors": ["High Consumption", "Leakage", "Low Reservoir Levels", "Supply Variability"]
    },

    "Public Safety Deployment": {
        "description": "Analyzing emergency response delay risk based on call volume, unit distance, and traffic.",
        "base_risk": 76,
        "interventions": {
            "Reposition Patrol Units": {
                "delta": -19,
                "pros": [
                    "Reduces response time\n",
                    "- Improves coverage\n"
                ],
                "cons": [
                    "May leave other areas exposed\n",
                    "- Requires real-time coordination\n"
                ]
            },
            "Increase Patrol Vehicles": {
                "delta": -15,
                "pros": [
                    "More units available\n",
                    "- Better peak-time coverage\n"
                ],
                "cons": [
                    "High cost\n",
                    "- Requires staffing\n"
                ]
            },
            "Traffic Signal Priority": {
                "delta": -11,
                "pros": [
                    "Faster emergency movement\n",
                    "- Reduces delay by ~6%\n"
                ],
                "cons": [
                    "Requires city-wide integration\n",
                    "- May disrupt normal traffic\n"
                ]
            }
        },
        "best_intervention":"Reposition Patrol Units",
        "best_score":57,
        "factors": ["High Call Volume", "Unit Distance", "Traffic Congestion", "Dispatch Delays"]
    }
}

# -----------------------------
# SIDEBAR
# -----------------------------

scenario = st.sidebar.selectbox("Select Scenario", list(scenarios.keys()))
intervention = st.sidebar.selectbox("Select Intervention", list(scenarios[scenario]["interventions"].keys()))
run = st.sidebar.button("Run Analysis")

# -----------------------------
# MAIN PAGE
# -----------------------------

st.write(f"### Scenario: {scenario}")
st.write(scenarios[scenario]["description"])

if run:
    st.header("Running Decision Intelligence Pipeline...")
    progress = st.progress(0)

    # 1. Load Operational Data
    with st.spinner("Loading operational data..."):
        time.sleep(1.8)
        progress.progress(10)
    st.success("Operational data loaded.")

    # 2. Existing AI Prediction
    with st.spinner("Running prediction model..."):
        time.sleep(1.8)
        progress.progress(25)

    base_risk = scenarios[scenario]["base_risk"]
    st.metric(f"{scenario} Risk Tomorrow", f"{base_risk}%")

    # 3. Structural Causal Model
    with st.spinner("Building structural causal model..."):
        time.sleep(1.5)
        progress.progress(45)

    causal_data = pd.DataFrame({
        "Factor": scenarios[scenario]["factors"],
        "Impact": [22, 18, 12, 9]
    })
    st.bar_chart(causal_data.set_index("Factor"))

    # 4. Counterfactual Simulation
    with st.spinner("Running counterfactual simulation..."):
        time.sleep(1.5)
        progress.progress(65)

    delta = scenarios[scenario]["interventions"][intervention]["delta"]
    new_risk = base_risk + delta

    st.metric(f"Risk After Intervention ({intervention})", f"{new_risk}%")

    # Explanation of WHY the percentage changed
    st.subheader("Why this percentage?")
    st.write(f"""
The new risk value of **{new_risk}%** is calculated by applying the causal impact  
(**{delta} percentage points**) of **{intervention}** to the baseline risk (**{base_risk}%**).
""")
    st.info("""This causal impact is derived from:
- how strongly the intervention influences the top causal factors  
- historical observational data  
- simulation of alternative outcomes  
- counterfactual reasoning (what would happen if we changed X?)  

In real deployment, these values come from DoWhy/EconML models trained on real operational data.
""")

    # 5. Intervention Ranking
    with st.spinner("Optimizing intervention ranking..."):
        time.sleep(1.5)
        progress.progress(85)

    ranking = pd.DataFrame({
        "Intervention": list(scenarios[scenario]["interventions"].keys()),
        "Impact": [f"{scenarios[scenario]['interventions'][i]['delta']}%" for i in scenarios[scenario]["interventions"]],
        "Pros": ["; ".join(scenarios[scenario]["interventions"][i]["pros"]) for i in scenarios[scenario]["interventions"]],
        "Cons": ["; ".join(scenarios[scenario]["interventions"][i]["cons"]) for i in scenarios[scenario]["interventions"]],
    })
    st.subheader("Intervention Ranking")
    st.table(ranking)

    # 6. Explanation Layer
    with st.spinner("Generating explanation..."):
        time.sleep(1.2)
        progress.progress(100)

    st.subheader("Causal Explanation")
    st.write(f"""
**Scenario:** {scenario}  
**Intervention:** {intervention}  

**Pros:**  
- {chr(10).join(scenarios[scenario]["interventions"][intervention]["pros"])}

**Cons:**  
- {chr(10).join(scenarios[scenario]["interventions"][intervention]["cons"])}
""")
    st.info("""This intervention modifies key causal factors, which is why it produces the  observed reduction in risk. The structural causal model quantifies this impact and the simulation engine validates it.
""")

    # 7. Final Recommendation
    st.success(f"Recommended Intervention: {scenarios[scenario]["best_intervention"]} (Confidence: {scenarios[scenario]["best_score"]})")
