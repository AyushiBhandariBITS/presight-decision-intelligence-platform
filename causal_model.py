import numpy as np
import pandas as pd
import dowhy
import joblib
from dowhy import CausalModel
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from econml.dr import LinearDRLearner
#MONKEY PATCH
import networkx as nx
import networkx.algorithms.d_separation as d_sep
# Map directly to the function inside the module rather than the module itself
if hasattr(d_sep, "d_separated"):
    nx.algorithms.d_separated = d_sep.d_separated
elif hasattr(d_sep, "is_d_separator"):
    nx.algorithms.d_separated = d_sep.is_d_separator
# =============================================================================
# 0. PREPARE THE SIMULATION DATA
# =============================================================================
# Assuming 'generate_abu_dhabi_chronological_data()' was run and saved to local env
try:
    df = pd.read_csv("abu_dhabi_traffic_operations_2026.csv")
except FileNotFoundError:
    # Fallback to importing your data variable directly if running in the same script session
    raise FileNotFoundError("Please ensure 'abu_dhabi_traffic_operations_2026.csv' has been generated and is present.")

# Convert string categorical values to numerical markers for the estimation models
df['Is_Highway'] = (df['Corridor_Type'] == 'Intercity_Highway').astype(int)

# =============================================================================
# 1. DEFINE THE STRUCTURAL CAUSAL MODEL (SCM) GRAPH
# =============================================================================
# Explicitly mapping out Confounders (W) -> Treatments (T) -> Mediators (M) -> Outcome (Y)
# The critical injection vector is: Weather -> Speed_Limit_Drop_Applied

scm_graph = """
digraph {
    Weather_Severity_Index [label="Weather Severity Confounder"];
    Historical_Accident_Base_Rate [label="Infrastructure Baseline Confounder"];
    Pavement_Temperature_C [label="Climatic Confounder"];
    Is_Highway [label="Geographic Confounder"];
    Speed_Limit_Drop_Applied [label="Treatment Variable"];
    Traffic_Density_Veh_Km [label="Traffic Volume Mediator"];
    Average_Speed_Variance [label="Speed Disparity Mediator"];
    Accident_Risk_Score [label="Target Outcome"];
    Weather_Severity_Index -> Speed_Limit_Drop_Applied;
    Historical_Accident_Base_Rate -> Speed_Limit_Drop_Applied;
    Weather_Severity_Index -> Traffic_Density_Veh_Km;
    Weather_Severity_Index -> Average_Speed_Variance;
    Weather_Severity_Index -> Accident_Risk_Score;
    Pavement_Temperature_C -> Average_Speed_Variance;
    Pavement_Temperature_C -> Accident_Risk_Score;
    Historical_Accident_Base_Rate -> Accident_Risk_Score;
    Is_Highway -> Average_Speed_Variance;
    Is_Highway -> Traffic_Density_Veh_Km;
    Speed_Limit_Drop_Applied -> Traffic_Density_Veh_Km;
    Speed_Limit_Drop_Applied -> Average_Speed_Variance;
    Speed_Limit_Drop_Applied -> Accident_Risk_Score;
    Traffic_Density_Veh_Km -> Accident_Risk_Score;
    Average_Speed_Variance -> Accident_Risk_Score;
}
"""

# =============================================================================
# 2. PASS TO DOWHY & IDENTIFY EFFECT (BACKDOOR CRITERION)
# =============================================================================
print("--- STAGE 1: Initializing Structural Causal Model & Identification ---")
model = CausalModel(
    data=df,
    treatment='Speed_Limit_Drop_Applied',
    outcome='Accident_Risk_Score',
    graph=scm_graph.replace("\n", " ")
)

# Identify the causal effect using backdoor verification rules
identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
print("\n[DoWhy Mathematical Proof Success]")
print(identified_estimand)

# =============================================================================
# 3. ESTIMATION STEP VIA ECONML (DOUBLE ROBUST CATE LEARNER)
# =============================================================================
print("\n--- STAGE 2: Estimating Conditional Average Treatment Effects (CATE) ---")

# Define target baseline confounders/effect modifiers (X) to evaluate CATE over
# This allows us to observe how treatment efficacy changes strictly based on specific baseline conditions
confounder_cols = ['Weather_Severity_Index', 'Historical_Accident_Base_Rate', 'Pavement_Temperature_C', 'Is_Highway']

# Initialize the Propensity (Treatment Assignment) and Outcome Model components
# Double Robust structure protects against functional form misspecifications
propensity_model = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42)
outcome_model = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42)

# Pass the identified structural rules into EconML's LinearDRLearner framework via DoWhy
# This calculates treatment effect as a flexible linear function of the baseline attributes
estimate = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.econml.dr.LinearDRLearner",
    method_params={
        "init_params": {
            "model_propensity": propensity_model,
            "model_regression": outcome_model,
            "featurizer": None
        },
        "fit_params": {}
    },
    effect_modifiers=confounder_cols
)

# Extract CATE across varying levels of Weather Severity to check if interaction logic is discovered
test_scenarios = pd.DataFrame([
    [0.0, 0.4, 35.0, 0],  # Clear day, standard city bridge route
    [4.0, 0.4, 35.0, 0],  # Moderate overcast conditions
    [8.0, 0.4, 22.0, 0],  # Severe winter morning fog sequence
    [10.0, 0.3, 20.0, 1]  # Extreme hazardous visibility on 140km/h highway
], columns=confounder_cols)

print("\n[Estimated Conditional Average Treatment Effects (CATE)]")
#cate_predictions = estimate.causal_estimator.estimator.effect(test_scenarios.values)

underlying_estimator = getattr(estimate, 'estimator', None) or getattr(estimate, 'causal_estimator', None)
econml_model = underlying_estimator.estimator

# Call the effect function directly from the EconML layer
cate_predictions = econml_model.effect(X=test_scenarios.values)

for idx, row in test_scenarios.iterrows():
    print(f"Scenario: Weather Severity = {row['Weather_Severity_Index']:.1f} | Base Rate = {row['Historical_Accident_Base_Rate']:.2f} "
          f"-> Marginal CATE (Risk Reduction): {cate_predictions[idx]:.2f} points")

for idx, row in test_scenarios.iterrows():
    print(f"Scenario: Weather Severity = {row['Weather_Severity_Index']:.1f} | Is Highway = {int(row['Is_Highway'])} "
          f"-> Marginal CATE (Risk Reduction): {cate_predictions[idx]:.2f} points")


# =============================================================================
# 4. EXPLICITLY SAVING THE ENGINE ARTIFACT TO DISK
# =============================================================================
print("\n--- STAGE 3: Saving Causal Model Artifact ---")
output_filename = f"abu_dhabi_causal_model_Speed_Limit_Drop_Applied.pkl"

model_payload = {
    'policy_target': 'Speed_Limit_Drop_Applied',
    'confounder_features': confounder_cols,
    'target_outcome': 'Accident_Risk_Score',
    'trained_learner': estimate
}

print(f"-> Serialization phase: Compressing structural artifacts to local environment...")
joblib.dump(model_payload, output_filename)
print(f"SUCCESS: Exported individual model file to disk: '{output_filename}'")



# =============================================================================
# 4. VERIFYING & REFUTING THE STRUCTURAL MODEL
# =============================================================================
print("\n--- STAGE 3: Structural Refutation & Stress-Testing Pipeline ---")

# Refutation 1: Placebo Treatment Refuter
print("\nRunning Test 1/3: Placebo Treatment Refuter...")
refute_placebo = model.refute_estimate(
    identified_estimand, estimate,
    method_name="placebo_treatment_refuter", 
    placebo_type="permutation"
)
print(f"New Placebo Treatment Causal Effect: {refute_placebo.new_effect:.4f} (Expected: ~0.00)")


# Refutation 2: Dummy Outcome Refuter
print("\nRunning Test 2/3: Dummy Outcome Refuter...")
refute_dummy_outcome = model.refute_estimate(
    identified_estimand, estimate,
    method_name="dummy_outcome_refuter", 
    outcome_type="normal"
)
print(f"New Dummy Outcome Causal Effect: {refute_dummy_outcome.new_effect:.4f} (Expected: ~0.00)")

# Refutation 3: Add Common Cause Confounder Refuter
print("\nRunning Test 3/3: Common Cause Confounder Refuter...")
refute_common_cause = model.refute_estimate(
    identified_estimand, estimate,
    method_name="add_unobserved_common_cause",
    confounders_effect_on_treatment="linear",
    confounders_effect_on_outcome="linear"
)
print(f"Original Estimated Effect Range Mean: {refute_common_cause.value:.4f}")
print(f"New Estimated Effect under Hidden Shock: {refute_common_cause.new_effect:.4f} (Expected: Stable/No wild swings)")
