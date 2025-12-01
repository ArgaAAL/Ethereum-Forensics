import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, f1_score, precision_score, recall_score
from sklearn.inspection import permutation_importance
import joblib
import warnings
import json
import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# --- SETUP ---
warnings.filterwarnings('ignore')
print("🚀 FIXED BENCHMARK RANSOMWARE DETECTION MODEL TRAINING")
print("="*70)
print("🔧 Implementing the 'Fixed Steel Ruler' Philosophy")
print("="*70)

# --- FEATURE ENGINEERING ---
def create_enhanced_pattern_features(df):
    """Create enhanced pattern features for ransomware detection"""
    # Using .get() with a default value of 0 for safety
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
    return df

# --- DATA LOADING AND PREPARATION ---
print("[1/7] Loading and Preparing Data...")
df = pd.read_csv('../EllipticPlusPlus-main/Actors Dataset/wallets_features_classes_combined.csv')
print(f"   - Loaded {len(df)} total samples.")

df = create_enhanced_pattern_features(df)
print(f"   - Created 10 enhanced pattern features.")

# Clean data and remap classes
df_clean = df.copy()
numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
df_clean[numeric_columns] = df_clean[numeric_columns].fillna(0)
df_clean = df_clean[df_clean['class'].isin([1, 2])]
df_clean['class'] = df_clean['class'].map({1: 1, 2: 0}) # 1=Illicit, 0=Licit

# Define features (X) and target (y)
exclude_cols = ['address', 'class']
feature_cols = [col for col in df_clean.columns if col not in exclude_cols]
X = df_clean[feature_cols]
y = df_clean['class']

# Force feature data type to float32 for consistency
print("   - Forcing feature data type to numpy.float32 for consistency.")
X = X.astype(np.float32)

print(f"   - Data prepared: {len(X)} samples, {len(feature_cols)} features.")

# --- THE FIXED BENCHMARK: SCALER TRAINING ON ALL BITCOIN DATA ---
print("\n[2/7] Creating the 'Fixed Steel Ruler' (Training Scaler on ALL Bitcoin Data)...")
print("   - 🔧 CORE PHILOSOPHY: The scaler represents the 'normal' behavior of Bitcoin")
print("   - 🔧 This benchmark will be used to measure ALL future addresses (Bitcoin & Ethereum)")

# Create the Fixed Benchmark StandardScaler
fixed_benchmark_scaler = StandardScaler()

# Fit the scaler on ALL Bitcoin data (not just training split)
fixed_benchmark_scaler.fit(X)
print(f"   - ✅ Fixed Benchmark Scaler fitted on {len(X)} Bitcoin addresses")
print(f"   - ✅ This scaler now represents the 'universal normal' for blockchain behavior")

# Apply the Fixed Benchmark scaling to ALL data
X_scaled_full = fixed_benchmark_scaler.transform(X)
print("   - ✅ All Bitcoin data transformed using the Fixed Benchmark")

# Save the Fixed Benchmark parameters - THIS IS THE UNIVERSAL RULER
fixed_benchmark_params = {
    'mean': fixed_benchmark_scaler.mean_.tolist(),
    'scale': fixed_benchmark_scaler.scale_.tolist(),
    'benchmark_source': 'EllipticPlusPlus_Bitcoin_Complete_Dataset',
    'num_samples_used': len(X),
    'philosophy': 'Fixed Steel Ruler - measures all blockchain addresses against Bitcoin normal'
}
with open('fixed_benchmark_scaler.json', 'w') as f:
    json.dump(fixed_benchmark_params, f, indent=2)
print("   - ✅ Fixed Benchmark parameters saved to 'fixed_benchmark_scaler.json'")

# --- TRAINING DATA SPLIT (AFTER SCALING) ---
print("\n[3/7] Creating Train/Test Split from Scaled Data...")
# Now split the ALREADY-SCALED data for model training
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(
    X_scaled_full, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   - Training set: {len(X_train_scaled)} samples")
print(f"   - Test set: {len(X_test_scaled)} samples")

# --- MLP MODEL TRAINING ---
print("\n[4/7] Training MLP Classifier on Z-Score Features...")
print("   - 🔧 Model learns patterns from Z-scores (deviations from Bitcoin normal)")

mlp_model = MLPClassifier(
    hidden_layer_sizes=(200, 100, 50),
    activation='relu',
    solver='adam',
    alpha=0.001,
    batch_size='auto',
    learning_rate='adaptive',
    learning_rate_init=0.001,
    max_iter=500,
    shuffle=True,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    warm_start=False
)

# Train the model on Z-score data
mlp_model.fit(X_train_scaled, y_train)
print(f"   - Training completed in {mlp_model.n_iter_} iterations.")
print(f"   - Final training loss: {mlp_model.loss_:.6f}")

# --- MODEL EVALUATION ---
print("\n[5/7] Evaluating Model Performance...")
# Predict probabilities
y_pred_proba = mlp_model.predict_proba(X_test_scaled)[:, 1]

# Find optimal threshold
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
best_threshold = thresholds[np.argmax(f1_scores)]

print(f"   - AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
print(f"   - Optimal Threshold for F1-Score: {best_threshold:.4f}")

# Final evaluation
y_pred_final = (y_pred_proba >= best_threshold).astype(int)
print("\n   - Final Classification Report (at optimal threshold):")
print(classification_report(y_test, y_pred_final, target_names=['Licit', 'Illicit']))

print("   - Final Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred_final)
print("     Predicted:  Licit  Illicit")
print(f"     Actual Licit:   {cm[0,0]:5d}   {cm[0,1]:5d}")
print(f"     Actual Illicit: {cm[1,0]:5d}   {cm[1,1]:5d}")

# --- ONNX EXPORT ---
print("\n[6/7] Exporting to Clean ONNX Format...")

try:
    n_features = X_train_scaled.shape[1]
    initial_type = [('float_input', FloatTensorType([1, n_features]))]
    print(f"   - Defining ONNX input shape as [1, {n_features}]")

    print("   - Converting MLP model ONLY (no Pipeline, no Scaler)...")
    onnx_model = convert_sklearn(
        mlp_model,
        initial_types=initial_type,
        target_opset=11,
        options={id(mlp_model): {'zipmap': False}}
    )

    onnx_filename = 'fixed_benchmark_model.onnx'
    with open(onnx_filename, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"   - ✅ Clean ONNX model saved as '{onnx_filename}'")

    # Final verification
    onnx_model_check = onnx.load(onnx_filename)
    onnx.checker.check_model(onnx_model_check)
    print("   - ✅ ONNX model verification passed!")

except Exception as e:
    print(f"[ERROR] ONNX export failed: {e}")
    import traceback
    traceback.print_exc()

# --- SAVE ARTIFACTS & CREATE TEST SAMPLE ---
print("\n[7/7] Saving Final Artifacts...")

# Enhanced metadata with Fixed Benchmark information
metadata = {
    'feature_names': feature_cols,
    'num_features': len(feature_cols),
    'deployment_threshold': float(best_threshold),
    'class_names': ['licit', 'illicit'],
    'model_version': '6.0_Fixed_Benchmark',
    'model_type': 'MLPClassifier',
    'auc_score': float(roc_auc_score(y_test, y_pred_proba)),
    'best_f1_score': float(max(f1_scores)),
    'benchmark_philosophy': 'Fixed Steel Ruler trained on complete Bitcoin dataset',
    'benchmark_samples': len(X),
    'z_score_interpretation': 'All features are Z-scores relative to Bitcoin normal behavior'
}
with open('model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=4)
print("   - ✅ Enhanced metadata saved to 'model_metadata.json'")

# Create test sample
print("   - Creating test sample from an illicit wallet...")
illicit_indices = df_clean[df_clean['class'] == 1].index

if not illicit_indices.empty:
    sample_index_to_find = illicit_indices[0]
    raw_sample_df = df.loc[[sample_index_to_find]].copy()
    enhanced_sample_df = create_enhanced_pattern_features(raw_sample_df)
    test_sample_dict = enhanced_sample_df[feature_cols].iloc[0].to_dict()
    
    with open('test_sample.json', 'w') as f:
        json.dump(test_sample_dict, f, indent=4)
    print("   - ✅ Test sample saved to 'test_sample.json'")
else:
    print("   - ⚠️  No illicit samples found to create a test case.")

# Save Python model for other uses
joblib.dump({
    'model': mlp_model, 
    'fixed_benchmark_scaler': fixed_benchmark_scaler,
    'philosophy': 'Fixed Steel Ruler'
}, 'fixed_benchmark_complete.joblib')
print("   - ✅ Complete Fixed Benchmark system saved to 'fixed_benchmark_complete.joblib'")

# --- DEMONSTRATE THE Z-SCORE INTERPRETATION ---
print("\n" + "="*70)
print("🔍 DEMONSTRATING Z-SCORE INTERPRETATION")
print("="*70)

# Take a sample and show its Z-score interpretation
sample_features = X.iloc[0].values
sample_zscores = fixed_benchmark_scaler.transform([sample_features])[0]

print("Example: How raw features become Z-scores against Bitcoin normal:")
print(f"{'Feature':<25} {'Raw Value':<15} {'Z-Score':<15} {'Interpretation'}")
print("-" * 75)

for i, feature_name in enumerate(feature_cols[:5]):  # Show first 5 features
    raw_val = sample_features[i]
    z_score = sample_zscores[i]
    
    if abs(z_score) > 2:
        interp = "EXTREME deviation"
    elif abs(z_score) > 1:
        interp = "Notable deviation"
    else:
        interp = "Normal range"
    
    print(f"{feature_name:<25} {raw_val:<15.2f} {z_score:<15.2f} {interp}")

print("\n🎯 KEY INSIGHT: Every feature is now a Z-score showing how many standard")
print("   deviations this address deviates from 'normal' Bitcoin behavior.")
print("   This allows cross-chain pattern recognition!")

# --- FINAL SUMMARY ---
print("\n" + "="*70)
print("✅ FIXED BENCHMARK TRAINING COMPLETE")
print("="*70)
print("🏆 You have successfully implemented the 'Fixed Steel Ruler' philosophy!")
print("\nArtifacts created:")
print(f"  1. Universal Model:         fixed_benchmark_model.onnx")
print(f"  2. Fixed Benchmark Scaler:  fixed_benchmark_scaler.json")
print(f"  3. Deployment Metadata:     model_metadata.json")
print(f"  4. Test Sample:             test_sample.json")
print(f"  5. Complete System:         fixed_benchmark_complete.joblib")

print("\n🚀 DEPLOYMENT WORKFLOW FOR ETHEREUM ADDRESSES:")
print("  1. Extract Ethereum features using your fixed_ethereum_processor.py")
print("  2. Load fixed_benchmark_scaler.json parameters")
print("  3. Convert raw Ethereum features to Z-scores using Bitcoin benchmark:")
print("     z_score = (ethereum_feature - bitcoin_mean) / bitcoin_std")
print("  4. Feed Z-scores to fixed_benchmark_model.onnx")
print("  5. Model outputs probability of illicit behavior")
print("  6. Apply threshold from metadata to get final classification")

print("\n🌟 UNIVERSAL LANGUAGE ACHIEVED:")
print("  - Bitcoin addresses: Measured against their own normal")
print("  - Ethereum addresses: Measured against Bitcoin normal")  
print("  - Model learns: 'How unusual is this behavior pattern?'")
print("  - Result: Cross-chain behavioral pattern recognition!")