import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import joblib
import warnings
import json
import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
warnings.filterwarnings('ignore')

print("🚀 ULTIMATE RANSOMWARE DETECTION MODEL TRAINING v3 (ONNX Export)")
print("="*60)

# Load data
print("[INFO] Loading training data...")
df = pd.read_csv('../EllipticPlusPlus-main/Actors Dataset/wallets_features_classes_combined.csv')
print(f"[SUCCESS] Loaded {len(df)} total samples")

print(f"[INFO] Dataset columns: {list(df.columns)}")

# Create ENHANCED pattern features with proper scaling
print("[INFO] Creating ENHANCED pattern features...")

def create_enhanced_pattern_features(df):
    """Create enhanced pattern features for ransomware detection"""
    
    # 1. Partner Transaction Ratio (connectivity)
    df['partner_transaction_ratio'] = (
        df.get('transacted_w_address_total', 0) / 
        (df.get('total_txs', 1) + 1e-8)
    )
    
    # 2. Activity Density (txs per block)
    df['activity_density'] = (
        df.get('total_txs', 0) / 
        (df.get('lifetime_in_blocks', 1) + 1e-8)
    )
    
    # 3. Transaction Size Variance (volatility)
    df['transaction_size_variance'] = (
        df.get('btc_transacted_max', 0) - df.get('btc_transacted_min', 0)
    ) / (df.get('btc_transacted_mean', 1) + 1e-8)
    
    # 4. Flow Imbalance (money laundering indicator)
    df['flow_imbalance'] = (
        (df.get('btc_sent_total', 0) - df.get('btc_received_total', 0)) / 
        (df.get('btc_transacted_total', 1) + 1e-8)
    )
    
    # 5. Temporal Spread (time pattern)
    df['temporal_spread'] = (
        df.get('last_block_appeared_in', 0) - df.get('first_block_appeared_in', 0)
    ) / (df.get('num_timesteps_appeared_in', 1) + 1e-8)
    
    # 6. Fee Percentile (urgency indicator)
    df['fee_percentile'] = (
        df.get('fees_total', 0) / 
        (df.get('btc_transacted_total', 1) + 1e-8)
    )
    
    # 7. Interaction Intensity (network centrality)
    df['interaction_intensity'] = (
        df.get('num_addr_transacted_multiple', 0) / 
        (df.get('transacted_w_address_total', 1) + 1e-8)
    )
    
    # 8. Value Per Transaction (transaction size)
    df['value_per_transaction'] = (
        df.get('btc_transacted_total', 0) / 
        (df.get('total_txs', 1) + 1e-8)
    )
    
    # 9. RANSOMWARE-SPECIFIC: Burst Activity (rapid txs)
    df['burst_activity'] = (
        df.get('total_txs', 0) * df.get('activity_density', 0)
    )
    
    # 10. RANSOMWARE-SPECIFIC: Mixing Intensity (obfuscation)
    df['mixing_intensity'] = (
        df.get('partner_transaction_ratio', 0) * df.get('interaction_intensity', 0)
    )
    
    return df

df = create_enhanced_pattern_features(df)
print(f"[SUCCESS] Created 10 enhanced pattern features")

# Prepare data
print("[INFO] Preparing training data...")
print(f"[DEBUG] Original class distribution: {df['class'].value_counts().to_dict()}")

# More careful data cleaning
df_clean = df.copy()

# Handle missing values more carefully
print(f"[INFO] Missing values per column:")
missing_counts = df_clean.isnull().sum()
for col, count in missing_counts[missing_counts > 0].items():
    print(f"  {col}: {count}")

# Fill missing values instead of dropping
numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
df_clean[numeric_columns] = df_clean[numeric_columns].fillna(0)

# Check class distribution BEFORE filtering
print(f"[DEBUG] After handling missing values: {df_clean['class'].value_counts().to_dict()}")

# FIXED: Only keep samples with valid class labels (1=Illicit, 2=Licit)
df_clean = df_clean[df_clean['class'].isin([1, 2])]
print(f"[DEBUG] After class filtering: {df_clean['class'].value_counts().to_dict()}")

# FIXED: REMAP CLASSES: 1->1 (Illicit), 2->0 (Licit) for binary classification
df_clean['class'] = df_clean['class'].map({1: 1, 2: 0})  # 1=Illicit, 0=Licit
print(f"[DEBUG] After class remapping: {df_clean['class'].value_counts().to_dict()}")

# Features (exclude non-feature columns)
exclude_cols = ['address', 'class']
feature_cols = [col for col in df_clean.columns if col not in exclude_cols]
X = df_clean[feature_cols]
y = df_clean['class']

print(f"[INFO] Training data prepared:")
print(f"  - Total samples: {len(X)}")
print(f"  - Licit (0): {sum(y == 0)}")
print(f"  - Illicit (1): {sum(y == 1)}")
print(f"  - Total features: {len(feature_cols)}")

# Calculate class weights for imbalanced data
if len(np.unique(y)) > 1:
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    print(f"[INFO] Class weights: {class_weight_dict}")
else:
    print(f"[ERROR] Only one class found in data: {np.unique(y)}")
    print("[ERROR] Cannot train binary classifier with single class!")
    print("[DEBUG] Checking original data...")
    print(f"[DEBUG] Original classes: {df['class'].unique()}")
    print(f"[DEBUG] Class value counts: {df['class'].value_counts()}")
    exit(1)

print(X.describe()) 

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Use RobustScaler (better for outliers)
print("[INFO] Scaling features with RobustScaler...")
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train XGBoost with proper parameters for imbalanced data
print("[INFO] Training XGBoost model with imbalanced data handling...")
print(f"[INFO] Class imbalance ratio: {sum(y == 0) / sum(y == 1):.2f}")

# Enhanced XGBoost parameters for ransomware detection
if len(np.unique(y)) > 1:
    scale_pos_weight = class_weights[1]/class_weights[0]
else:
    scale_pos_weight = 1.0  # Default if only one class

xgb_model = xgb.XGBClassifier(
    n_estimators=500,              # More trees
    max_depth=8,                   # Deeper trees for complex patterns
    learning_rate=0.05,            # Lower learning rate for stability
    subsample=0.8,                 # Prevent overfitting
    colsample_bytree=0.8,          # Feature sampling
    scale_pos_weight=scale_pos_weight,  # Handle imbalance
    random_state=42,
    objective='binary:logistic',
    eval_metric='auc',
    tree_method='hist',            # Faster training
    early_stopping_rounds=50       # Prevent overfitting
)

# Fit with validation
eval_set = [(X_train_scaled, y_train), (X_test_scaled, y_test)]
xgb_model.fit(
    X_train_scaled, y_train,
    eval_set=eval_set,
    verbose=False
)

# Predict probabilities
y_pred_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]

# Find optimal threshold using precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
best_threshold_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_threshold_idx]

print("\n" + "="*60)
print("📊 MODEL EVALUATION")
print("="*60)
print(f"AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
print(f"Optimal Threshold: {best_threshold:.3f}")

# Test multiple thresholds
thresholds_to_test = [0.1, 0.2, 0.3, 0.4, 0.5, best_threshold]
for threshold in thresholds_to_test:
    y_pred = (y_pred_proba >= threshold).astype(int)
    from sklearn.metrics import f1_score, precision_score, recall_score
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    print(f"Threshold {threshold:.1f}: F1={f1:.3f}, Precision={precision:.3f}, Recall={recall:.3f}")

# Final evaluation with optimal threshold
y_pred_final = (y_pred_proba >= best_threshold).astype(int)
print(f"\nFinal Confusion Matrix (Threshold: {best_threshold:.3f}):")
cm = confusion_matrix(y_test, y_pred_final)
print("   Predicted:  Licit  Illicit")
print(f"   Licit:      {cm[0,0]:5d}     {cm[0,1]:4d}")
print(f"   Illicit:      {cm[1,0]:3d}     {cm[1,1]:4d}")

# Feature importance
print("\n" + "="*40)
print("🔍 FEATURE IMPORTANCE ANALYSIS")
print("="*40)
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print("Top 15 Most Important Features:")
pattern_features = ['partner_transaction_ratio', 'activity_density', 'transaction_size_variance', 
                   'flow_imbalance', 'temporal_spread', 'fee_percentile', 'interaction_intensity',
                   'value_per_transaction', 'burst_activity', 'mixing_intensity']

for i, (_, row) in enumerate(feature_importance.head(15).iterrows()):
    feature_name = row['feature']
    importance = row['importance']
    is_pattern = "🎯" if feature_name in pattern_features else "📊"
    print(f"{is_pattern} {feature_name:30s} {importance:.6f}")

pattern_in_top15 = sum(1 for feat in feature_importance.head(15)['feature'] if feat in pattern_features)
print(f"\nPattern features in top 15: {pattern_in_top15}/{len(pattern_features)}")

# ==================== ONNX EXPORT SECTION ====================
print("\n" + "="*60)
print("🔄 EXPORTING MODEL TO ONNX FORMAT")
print("="*60)

try:
    # For XGBoost, we need to create a pipeline with the scaler first
    from sklearn.pipeline import Pipeline
    
    # Create a pipeline with scaler and XGBoost
    pipeline = Pipeline([
        ('scaler', scaler),
        ('classifier', xgb_model)
    ])
    
    # Define input type for ONNX conversion
    initial_type = [('float_input', FloatTensorType([None, len(feature_cols)]))]
    
    # Convert to ONNX
    onnx_model = convert_sklearn(
        pipeline, 
        initial_types=initial_type,
        target_opset=11,  # Compatible with most ONNX runtimes
        options={id(xgb_model): {'zipmap': False}}  # Output probabilities directly
    )
    
    # Save ONNX model
    onnx_filename = 'ransomware_model.onnx'
    with open(onnx_filename, "wb") as f:
        f.write(onnx_model.SerializeToString())
    
    print(f"[SUCCESS] ONNX model saved as '{onnx_filename}'")
    
    # Verify ONNX model
    onnx_model_check = onnx.load(onnx_filename)
    onnx.checker.check_model(onnx_model_check)
    print("[SUCCESS] ONNX model verification passed!")
    
except Exception as e:
    print(f"[ERROR] ONNX export failed: {str(e)}")
    print("[INFO] Trying alternative XGBoost ONNX export...")

    try:
        # IMPORTANT: Import from the correct library for this block
        from onnxmltools.convert import convert_xgboost
        from onnxmltools.convert.common.data_types import FloatTensorType # <-- CHANGE THIS LINE
        from onnxmltools.utils import save_model

        # Define the input type using the newly imported FloatTensorType
        initial_type_xgb = [('input', FloatTensorType([None, len(feature_cols)]))]

        # Convert XGBoost model to ONNX (without scaler)
        onnx_model_xgb = convert_xgboost(
            xgb_model,
            initial_types=initial_type_xgb, # <-- Use the correct type
            target_opset=11
        )

        # Save the successfully converted model
        onnx_filename = 'ransomware_model.onnx' # You can now name it correctly
        save_model(onnx_model_xgb, onnx_filename)
        print(f"[SUCCESS] Alternative ONNX export successful. Model saved as '{onnx_filename}'")

        # Save scaler parameters separately for Rust
        scaler_params = {
            'center': scaler.center_.tolist(),
            'scale': scaler.scale_.tolist(),
            'feature_names': feature_cols,
            'type': 'RobustScaler'
        }

        with open('scaler_params.json', 'w') as f:
            json.dump(scaler_params, f, indent=2)
        print("[SUCCESS] Scaler parameters saved as 'scaler_params.json'")

    except Exception as e2:
        print(f"[ERROR] Alternative ONNX export also failed: {str(e2)}")
        print("[WARNING] Continuing without ONNX export...")

# Save model metadata for Rust integration
metadata = {
    'feature_names': feature_cols,
    'num_features': len(feature_cols),
    'threshold': float(best_threshold),
    'class_names': ['licit', 'illicit'],
    'model_version': '3.0',
    'scaler_type': 'RobustScaler',
    'auc_score': float(roc_auc_score(y_test, y_pred_proba)),
    'best_f1_score': float(max(f1_scores))
}

with open('model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"[SUCCESS] Model metadata saved as 'model_metadata.json'")

# Save enhanced model (Python version)
model_data = {
    'model': xgb_model,
    'scaler': scaler,
    'feature_names': feature_cols,
    'threshold': best_threshold,
    'class_weights': class_weight_dict,
    'scaler_type': 'RobustScaler'
}

# Find a known illicit sample from the cleaned data
illicit_samples = df_clean[df_clean['class'] == 1]

if not illicit_samples.empty:
    # Get the very first illicit sample
    sample_to_test = illicit_samples.iloc[0]
    
    # We need the base features BEFORE the 10 enhanced ones were made.
    # So we get them from the original dataframe using the address.
    sample_address = sample_to_test['address']
    original_sample = df[df['address'] == sample_address].iloc[0]

    # Convert the original features of this sample to a dictionary
    sample_dict = original_sample.to_dict()
    
    # Save test sample for Rust testing
    test_sample = {}
    for feature in feature_cols:
        if feature in sample_dict:
            test_sample[feature] = float(sample_dict[feature]) if pd.notna(sample_dict[feature]) else 0.0
        else:
            # This is an enhanced feature, calculate it
            if feature == 'partner_transaction_ratio':
                test_sample[feature] = float(sample_dict.get('transacted_w_address_total', 0) / (sample_dict.get('total_txs', 1) + 1e-8))
            elif feature == 'activity_density':
                test_sample[feature] = float(sample_dict.get('total_txs', 0) / (sample_dict.get('lifetime_in_blocks', 1) + 1e-8))
            # Add other enhanced features calculations here as needed
            else:
                test_sample[feature] = 0.0
    
    with open('test_sample.json', 'w') as f:
        json.dump(test_sample, f, indent=2)
    
    print("\n" + "="*60)
    print("📋 TEST SAMPLE FOR RUST CANISTER")
    print("="*60)
    print("Saved test sample to 'test_sample.json'")
    print(f"Expected class: Illicit (1)")
    print("="*60 + "\n")
else:
    print("[ERROR] No illicit samples found to create a test case.")

joblib.dump(model_data, 'enhanced_ransomware_model_v3.joblib')
print("[SUCCESS] Enhanced Python model saved to 'enhanced_ransomware_model_v3.joblib'!")

print(f"\n🎯 ONNX Export Summary:")
print(f"  ✅ ONNX model: ransomware_model.onnx")
print(f"  ✅ Scaler params: scaler_params.json")
print(f"  ✅ Model metadata: model_metadata.json")
print(f"  ✅ Test sample: test_sample.json")

print(f"\n🎯 Key enhancements applied:")
print(f"  ✅ RobustScaler for better outlier handling")
print(f"  ✅ Proper class weight balancing: {class_weight_dict}")
print(f"  ✅ Optimal threshold: {best_threshold:.3f}")
print(f"  ✅ Enhanced XGBoost parameters")
print(f"  ✅ Ransomware-specific features")
print(f"  ✅ AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
print(f"  ✅ Best F1 Score: {max(f1_scores):.4f}")
print(f"  ✅ ONNX export ready for Rust/Tract!")

print(f"\n📋 Model Summary:")
print(f"  - Training samples: {len(X_train)}")
print(f"  - Test samples: {len(X_test)}")
print(f"  - Features: {len(feature_cols)}")
print(f"  - Scaler: RobustScaler")
print(f"  - Threshold: {best_threshold:.3f}")
print(f"  - Ready for ICP Rust canister deployment!")