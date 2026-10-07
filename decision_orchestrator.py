import os
import joblib
import numpy as np
import pandas as pd

class AbuDhabiDecisionOrchestrator:
    def __init__(self, model_directory="."):
        """
        Initializes the Decision Engine by mapping and loading separate causal models.
        """
        self.model_directory = model_directory
        self.policy_models = {}
        
        # Define the exact file structure you exported
        self.model_files = {
            'Speed_Limit_Drop_Applied': 'abu_dhabi_causal_model_Speed_Limit_Drop_Applied.pkl',
            'Nav_Detour_Triggered': 'abu_dhabi_causal_model_Nav_Detour_Triggered.pkl',
            'Transit_Flow_Sync_Active': 'abu_dhabi_causal_model_Transit_Flow_Sync_Active.pkl'
        }
        
        # Real-world operational constraints configuration matrix
        self.policy_constraints = {
            'Speed_Limit_Drop_Applied': {'base_cost_aed': 1500, 'base_time_mins': 2, 'risk_tolerance': 0.95},
            'Nav_Detour_Triggered':     {'base_cost_aed': 4500, 'base_time_mins': 8, 'risk_tolerance': 0.80},
            'Transit_Flow_Sync_Active':  {'base_cost_aed': 9000, 'base_time_mins': 15, 'risk_tolerance': 0.65}
        }
        
        self.load_models()

    def load_models(self):
        """Safely unpickles payload artifacts and isolates the underlying EconML models."""
        for policy, filename in self.model_files.items():
            full_path = os.path.join(self.model_directory, filename)
            if os.path.exists(full_path):
                try:
                    payload = joblib.load(full_path)
                    
                    # Dig through the DoWhy wrapping layers straight to the core EconML estimator
                    dowhy_estimate = payload['trained_learner']
                    underlying_estimator = getattr(dowhy_estimate, 'estimator', None) or getattr(dowhy_estimate, 'causal_estimator', None)
                    econml_model = underlying_estimator.estimator
                    
                    self.policy_models[policy] = {
                        'model': econml_model,
                        'features': payload['confounder_features']
                    }
                    print(f"SUCCESS: Loaded Causal Engine for '{policy}'")
                except Exception as e:
                    print(f"ERROR: Failed parsing structural layers for {filename}: {str(e)}")
            else:
                print(f"WARNING: Target artifact '{filename}' not detected in path context.")

    def _calculate_operational_friction(self, policy, weather_severity, is_highway):
        """
        Dynamically adjusts static cost & time parameters based on environment context.
        """
        base = self.policy_constraints[policy]
        
        # Scale factors based on the reality of infrastructure deployment
        highway_multiplier = 1.5 if is_highway == 1 else 1.0
        weather_multiplier = 1.0 + (weather_severity / 10.0) # Storms slow down deployment
        
        calculated_time = base['base_time_mins'] * weather_multiplier * (1.2 if is_highway == 1 else 1.0)
        calculated_cost = base['base_cost_aed'] * highway_multiplier
        
        return round(calculated_cost, 2), round(calculated_time, 1)

    def compute_policy_recommendations(self, weather_severity, base_rate, temperature, is_highway, weights=None):
        """
        Runs multi-model inference and calculates the Multi-Objective composite priority vector.
        """
        if not self.policy_models:
            raise ValueError("Execution halted: No valid causal model parameters have been initialized.")
            
        # Default operational optimization weights (Balanced Profile)
        if weights is None:
            weights = {'risk': 0.5, 'cost': 0.3, 'time': 0.2}
            
        # Format user dashboard inputs to match model training features configuration
        # Expected structure: [Weather_Severity_Index, Historical_Accident_Base_Rate, Pavement_Temperature_C, Is_Highway]
        scenario_features = np.array([[weather_severity, base_rate, temperature, is_highway]])
        
        raw_results = []
        
        # Step 1: Run inference across all active models and extract operational data
        for policy, components in self.policy_models.items():
            econml_model = components['model']
            
            # Predict CATE (Conditional Average Treatment Effect)
            # Typically returns a negative delta (e.g., -14.5 risk score drop)
            cate_prediction = float(econml_model.effect(X=scenario_features)[0])
            risk_reduction_magnitude = abs(cate_prediction) # Focus on magnitude for optimization
            
            # Extract real-world constraints
            cost_aed, time_mins = self._calculate_operational_friction(policy, weather_severity, is_highway)
            
            raw_results.append({
                'policy': policy,
                'risk_drop': risk_reduction_magnitude,
                'cost': cost_aed,
                'time': time_mins
            })
            
        # Convert to DataFrame to quickly execute vectorized Min-Max normalizations
        df_metrics = pd.DataFrame(raw_results)
        
        # Step 2: Normalize metrics between 0 and 1 to prevent unit mismatching
        # Standard safety handles protect against division by zero if values are identical
        def min_max_scale(series, invert=False):
            if series.max() == series.min():
                return pd.Series(1.0, index=series.index) if not invert else pd.Series(0.0, index=series.index)
            scaled = (series - series.min()) / (series.max() - series.min())
            return 1.0 - scaled if invert else scaled

        # We want high risk drop (higher is better), but low cost and low time (lower is better)
        df_metrics['norm_risk'] = min_max_scale(df_metrics['risk_drop'], invert=False)
        df_metrics['norm_cost'] = min_max_scale(df_metrics['cost'], invert=True)
        df_metrics['norm_time'] = min_max_scale(df_metrics['time'], invert=True)
        
        # Step 3: Compute the Composite Score
        df_metrics['composite_score'] = (
            (df_metrics['norm_risk'] * weights['risk']) +
            (df_metrics['norm_cost'] * weights['cost']) +
            (df_metrics['norm_time'] * weights['time'])
        ) * 100 # Standardize to a clean 0-100 scale
        
        # Step 4: Format output payload array
        sorted_metrics = df_metrics.sort_values(by='composite_score', ascending=False)
        
        recommendations = []
        for rank, (_, row) in enumerate(sorted_metrics.iterrows(), 1):
            recommendations.append({
                "rank": rank,
                "policy_name": row['policy'].replace('_', ' '),
                "raw_policy_key": row['policy'],
                "risk_mitigation": f"-{row['risk_drop']:.2f} Risk Pts",
                "cost_profile": f"{row['cost']:,} AED",
                "time_profile": f"{row['time']} Mins",
                "score": round(row['composite_score'], 1),
                "evidence_log": f"EconML model outputs dynamic treatment effect mapping for context values."
            })
            
        return recommendations
if __name__ == "__main__":
    # Initialize engine
    orchestrator = AbuDhabiDecisionOrchestrator()
    
    print("\n--- TEST RUN 1: Hazardous Storm Scenario (Priority on Risk Mitigation) ---")
    storm_weights = {'risk': 0.8, 'cost': 0.1, 'time': 0.1}
    # Parameters: Weather=9.0 (severe), Base_Rate=0.4, Temp=21.0, Is_Highway=1
    res_storm = orchestrator.compute_policy_recommendations(9.0, 0.4, 21.0, 1, weights=storm_weights)
    for rec in res_storm:
        print(f"Rank {rec['rank']}: {rec['policy_name']} | Score: {rec['score']} | Risk Drop: {rec['risk_mitigation']} | Cost: {rec['cost_profile']}")

    print("\n--- TEST RUN 2: Standard Day Scenario (Priority on Fiscal Cost Efficiency) ---")
    standard_weights = {'risk': 0.3, 'cost': 0.5, 'time': 0.2}
    # Parameters: Weather=1.0 (clear), Base_Rate=0.3, Temp=34.0, Is_Highway=0
    res_clear = orchestrator.compute_policy_recommendations(1.0, 0.3, 34.0, 0, weights=standard_weights)
    for rec in res_clear:
        print(f"Rank {rec['rank']}: {rec['policy_name']} | Score: {rec['score']} | Risk Drop: {rec['risk_mitigation']} | Cost: {rec['cost_profile']}")
