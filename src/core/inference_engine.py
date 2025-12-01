import pandas as pd
import numpy as np
import json
import onnxruntime as ort
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class FixedBenchmarkInference:
    def __init__(self, model_path='fixed_benchmark_model.onnx', 
                 scaler_path='fixed_benchmark_scaler.json',
                 metadata_path='model_metadata.json'):
        """
        Initialize the Fixed Benchmark Inference system
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.metadata_path = metadata_path
        
        # Load components
        self.load_benchmark_scaler()
        self.load_model()
        self.load_metadata()
        
    def load_benchmark_scaler(self):
        """Load the fixed benchmark scaler parameters"""
        try:
            with open(self.scaler_path, 'r') as f:
                scaler_data = json.load(f)
            
            self.benchmark_mean = np.array(scaler_data['mean'], dtype=np.float32)
            self.benchmark_scale = np.array(scaler_data['scale'], dtype=np.float32)
            self.benchmark_source = scaler_data['benchmark_source']
            
            print(f"✅ Loaded Fixed Benchmark Scaler from {self.benchmark_source}")
            print(f"   - {scaler_data['num_samples_used']} Bitcoin samples used as benchmark")
            
        except Exception as e:
            raise Exception(f"Failed to load benchmark scaler: {e}")
    
    def load_model(self):
        """Load the ONNX model"""
        try:
            self.onnx_session = ort.InferenceSession(self.model_path)
            self.input_name = self.onnx_session.get_inputs()[0].name
            print(f"✅ Loaded ONNX model from {self.model_path}")
            
        except Exception as e:
            raise Exception(f"Failed to load ONNX model: {e}")
    
    def load_metadata(self):
        """Load model metadata"""
        try:
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
            
            self.feature_names = self.metadata['feature_names']
            self.threshold = self.metadata['deployment_threshold']
            self.expected_features = self.metadata['num_features']
            
            print(f"✅ Loaded metadata - Expected {self.expected_features} features")
            print(f"   - Deployment threshold: {self.threshold:.4f}")
            
        except Exception as e:
            raise Exception(f"Failed to load metadata: {e}")
    
    def create_enhanced_pattern_features(self, df):
        """Create enhanced pattern features for any blockchain data"""
        print("🔧 Creating enhanced pattern features...")
        
        # Using .get() with default values for missing columns
        df['partner_transaction_ratio'] = df.get('transacted_w_address_total', 0) / (df.get('total_txs', 1) + 1e-8)
        df['activity_density'] = df.get('total_txs', 0) / (df.get('lifetime_in_blocks', 1) + 1e-8)
        df['transaction_size_variance'] = (df.get('btc_transacted_max', 0) - df.get('btc_transacted_min', 0)) / (df.get('btc_transacted_mean', 1) + 1e-8)
        df['flow_imbalance'] = (df.get('btc_sent_total', 0) - df.get('btc_received_total', 0)) / (df.get('btc_transacted_total', 1) + 1e-8)
        df['temporal_spread'] = (df.get('last_block_appeared_in', 0) - df.get('first_block_appeared_in', 0)) / (df.get('num_timesteps_appeared_in', 1) + 1e-8)
        df['fee_percentile'] = df.get('fees_total', 0) / (df.get('btc_transacted_total', 1) + 1e-8)
        df['interaction_intensity'] = df.get('num_addr_transacted_multiple', 0) / (df.get('transacted_w_address_total', 1) + 1e-8)
        df['value_per_transaction'] = df.get('btc_transacted_total', 0) / (df.get('total_txs', 1) + 1e-8)
        df['burst_activity'] = df.get('total_txs', 0) * df['activity_density']
        df['mixing_intensity'] = df['partner_transaction_ratio'] * df['interaction_intensity']
        
        print(f"   - Created 10 enhanced pattern features")
        return df
    
    def prepare_ethereum_features(self, df):
        """Prepare Ethereum features to match Bitcoin feature structure"""
        print("🔄 Preparing Ethereum features...")
        
        # Create a copy to avoid modifying original
        df_prepared = df.copy()
        
        # Fill missing values with 0 for numeric columns
        numeric_columns = df_prepared.select_dtypes(include=[np.number]).columns
        df_prepared[numeric_columns] = df_prepared[numeric_columns].fillna(0)
        
        # Create enhanced features
        df_prepared = self.create_enhanced_pattern_features(df_prepared)
        
        # Extract only the features expected by the model
        available_features = []
        missing_features = []
        
        for feature in self.feature_names:
            if feature in df_prepared.columns:
                available_features.append(feature)
            else:
                missing_features.append(feature)
        
        print(f"   - Available features: {len(available_features)}/{len(self.feature_names)}")
        
        if missing_features:
            print(f"   - Missing features: {missing_features}")
            print("   - Creating missing features with zeros...")
            
            # Create missing features with zeros
            for feature in missing_features:
                df_prepared[feature] = 0.0
        
        # Select features in the exact order expected by the model
        feature_matrix = df_prepared[self.feature_names].astype(np.float32)
        
        return feature_matrix
    
    def apply_fixed_benchmark_scaling(self, features):
        """Apply the fixed benchmark scaling (Z-score transformation)"""
        print("📏 Applying Fixed Benchmark scaling...")
        
        # Convert to numpy array if needed
        if isinstance(features, pd.DataFrame):
            features_array = features.values
        else:
            features_array = features
        
        # Apply Z-score transformation using Bitcoin benchmark
        z_scores = (features_array - self.benchmark_mean) / (self.benchmark_scale + 1e-8)
        
        print(f"   - Transformed {z_scores.shape[1]} features to Z-scores")
        print(f"   - Z-score interpretation: How many std deviations from Bitcoin normal")
        
        return z_scores.astype(np.float32)
    
    def predict_batch(self, features_scaled):
        """
        Make predictions using the ONNX model, processing one address at a time.
        This is the corrected function.
        """
        print("🔮 Making predictions (processing one-by-one)...")
        
        all_probabilities = []
        all_binary_predictions = []

        # Loop through each row (each address's features) in the input
        for single_feature_row in features_scaled:
            # Reshape the row to (1, num_features) to match the model's expected input shape.
            # The model expects a batch size of 1.
            input_data = single_feature_row.reshape(1, -1)

            # Run inference for this single row.
            # The model returns two outputs:
            # 1. outputs[0]: The predicted label (a scalar value, e.g., [1])
            # 2. outputs[1]: The probabilities for each class (e.g., [[prob_licit, prob_illicit]])
            outputs = self.onnx_session.run(None, {self.input_name: input_data})
            
            # CORRECTED: The probabilities are in the *second* output (index 1).
            # For a single input, its shape is (1, 2). We need the second element of the inner list.
            illicit_probability = outputs[1][0][1]
            
            # Apply our custom deployment threshold to the probability to get the final prediction.
            binary_prediction = 1 if illicit_probability >= self.threshold else 0

            all_probabilities.append(illicit_probability)
            all_binary_predictions.append(binary_prediction)

        # Convert lists to numpy arrays to match the expected output format of the rest of the script
        return np.array(all_probabilities), np.array(all_binary_predictions)
    
    def analyze_ethereum_addresses(self, ethereum_csv_path):
        """Complete analysis pipeline for Ethereum addresses"""
        print("🚀 FIXED BENCHMARK ETHEREUM ANALYSIS")
        print("="*60)
        
        # Load Ethereum data
        print(f"[1/5] Loading Ethereum data from {ethereum_csv_path}")
        df_eth = pd.read_csv(ethereum_csv_path)
        print(f"   - Loaded {len(df_eth)} Ethereum addresses")
        
        # Prepare features
        print(f"\n[2/5] Preparing features...")
        features_prepared = self.prepare_ethereum_features(df_eth)
        print(f"   - Feature matrix shape: {features_prepared.shape}")
        
        # Apply fixed benchmark scaling
        print(f"\n[3/5] Applying Fixed Benchmark scaling...")
        features_scaled = self.apply_fixed_benchmark_scaling(features_prepared.values) # Pass numpy array
        
        # Make predictions
        print(f"\n[4/5] Making predictions...")
        probabilities, predictions = self.predict_batch(features_scaled)
        
        # Analyze results
        print(f"\n[5/5] Analyzing results...")
        
        # Create results dataframe
        results_df = df_eth.copy()
        results_df['illicit_probability'] = probabilities
        results_df['predicted_class'] = predictions
        results_df['risk_level'] = pd.cut(probabilities, 
                                          bins=[-0.1, 0.3, 0.7, 1.1], 
                                          labels=['Low', 'Medium', 'High'],
                                          right=False)
        
        # Summary statistics
        total_addresses = len(results_df)
        illicit_count = sum(predictions)
        illicit_percentage = (illicit_count / total_addresses) * 100 if total_addresses > 0 else 0
        
        print(f"\n📊 ANALYSIS SUMMARY:")
        print(f"   - Total addresses analyzed: {total_addresses}")
        print(f"   - Predicted illicit: {illicit_count} ({illicit_percentage:.1f}%)")
        print(f"   - Predicted licit: {total_addresses - illicit_count} ({100-illicit_percentage:.1f}%)")
        print(f"   - Average risk score: {np.mean(probabilities):.3f}")
        print(f"   - Highest risk score: {np.max(probabilities):.3f}")
        
        # Risk level distribution
        risk_dist = results_df['risk_level'].value_counts()
        print(f"\n🎯 RISK DISTRIBUTION:")
        for level in ['Low', 'Medium', 'High']:
            count = risk_dist.get(level, 0)
            pct = (count / total_addresses) * 100 if total_addresses > 0 else 0
            print(f"   - {level} Risk: {count} addresses ({pct:.1f}%)")
        
        # Show top risky addresses
        print(f"\n⚠️  TOP 10 HIGHEST RISK ADDRESSES:")
        top_risky = results_df.nlargest(min(10, total_addresses), 'illicit_probability')
        
        for i, (idx, row) in enumerate(top_risky.iterrows(), 1):
            address = row.get('address', f'Address_{idx}')
            prob = row['illicit_probability']
            risk = row['risk_level']
            print(f"   {i:2d}. {address} - Risk: {prob:.3f} ({risk})")
        
        # Save results
        output_path = 'ethereum_analysis_results.csv'
        results_df.to_csv(output_path, index=False)
        print(f"\n💾 Full results saved to: {output_path}")
        
        return results_df
    
    def analyze_single_address(self, address_features):
        """Analyze a single address"""
        if isinstance(address_features, dict):
            # Convert dict to DataFrame
            df_single = pd.DataFrame([address_features])
        else:
            df_single = address_features
        
        # Prepare and scale features
        features_prepared = self.prepare_ethereum_features(df_single)
        features_scaled = self.apply_fixed_benchmark_scaling(features_prepared.values) # Pass numpy array
        
        # Make prediction
        probability, prediction = self.predict_batch(features_scaled)
        
        result = {
            'illicit_probability': float(probability[0]),
            'predicted_class': int(prediction[0]),
            'risk_level': 'High' if probability[0] > 0.7 else 'Medium' if probability[0] > 0.3 else 'Low',
            'interpretation': f"This address deviates from Bitcoin normal behavior with {probability[0]:.1%} confidence of illicit activity"
        }
        
        return result


def main():
    """Main execution function"""
    print("🔧 FIXED BENCHMARK ETHEREUM INFERENCE")
    print("="*50)
    
    # Initialize the inference system
    try:
        inference_system = FixedBenchmarkInference()
        print("✅ Inference system initialized successfully!\n")
    except Exception as e:
        print(f"❌ Failed to initialize inference system: {e}")
        return
    
    # Analyze Ethereum addresses
    ethereum_csv_path = "D:/AAL/Coding/piton/BlockchainAnalyzer/xgboost/ethereum_features.csv"
    
    try:
        results = inference_system.analyze_ethereum_addresses(ethereum_csv_path)
        print("\n🎉 Analysis completed successfully!")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
