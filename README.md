# Fraud Detection LSTM Model

## Overview
This project implements a **Long Short-Term Memory (LSTM) neural network with attention mechanism** for detecting fraudulent financial transactions. The model analyzes sequences of historical transactions to predict fraud probability in real-time.

## Key Features
- **LSTM Architecture**: 2-layer LSTM with 64 hidden units to capture temporal patterns in transaction sequences
- **Attention Mechanism**: Dynamically weights transaction history steps to focus on suspicious patterns
- **Class Imbalance Handling**: Uses weighted loss function (pos_weight=50) to handle ~0.2% fraud rate
- **Gradient Clipping**: Prevents exploding gradients during training
- **Learning Rate Scheduling**: Adaptive learning rate reduction when loss plateaus
- **Risk Tiers**: Outputs fraud probability with LOW/MEDIUM/HIGH risk classification

## Model Architecture

```
Input (batch, 30 timesteps, 12 features)
    ↓
LSTM Layer 1 (12 → 64 hidden units)
    ↓
LSTM Layer 2 (64 → 64 hidden units)
    ↓
Attention Layer (computes softmax weights across time)
    ↓
Context Vector (weighted sum of LSTM outputs)
    ↓
Classifier (64 → 32 → 1)
    - ReLU activation
    - Dropout (30%)
    - Sigmoid output
    ↓
Fraud Probability (0-1)
```

## Input Features (12 per Transaction)
1. `amount_log` - Log-scaled transaction amount
2. `hour_sin`, `hour_cos` - Cyclical encoding of transaction hour
3. `day_sin`, `day_cos` - Cyclical encoding of day of week
4. `merchant_cat_enc` - Encoded merchant category
5. `is_online` - Binary flag for online/in-person
6. `is_international` - Binary flag for international transaction
7. `velocity_1h` - Transaction count in last 1 hour
8. `velocity_24h` - Transaction count in last 24 hours
9. `avg_amount_7d` - Average transaction amount over 7 days
10. `days_since_last_txn` - Days elapsed since last transaction

## Dataset Structure
- **Total Samples**: 1000 (synthetic in demo)
- **Sequence Length**: 30 timesteps (transaction history)
- **Features**: 12 per transaction
- **Labels**: Binary (0=legitimate, 1=fraud)
- **Class Distribution**: ~0.2% fraud rate (class imbalance)

## Training Configuration
| Parameter | Value | Purpose |
|-----------|-------|---------|
| Epochs | 5-10 | Number of complete passes through data |
| Batch Size | 32 | Samples per gradient update |
| Learning Rate | 0.001 | Step size for weight updates |
| Optimizer | Adam | Adaptive moment estimation |
| Loss Function | BCEWithLogitsLoss | Handle class imbalance with pos_weight |
| pos_weight | 50.0 | Penalize false negatives 50x more |
| Gradient Clip | 1.0 | Prevent exploding gradients |
| Scheduler | ReduceLROnPlateau | Reduce LR if loss doesn't improve for 2 epochs |

## Usage

### Training the Model
```python
# Create dataset and loader
dataset = DummyFraudDataset(num_samples=1000, seq_len=30, input_dim=12)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model
model = FraudLSTM(input_dim=12, hidden_dim=64, num_layers=2, dropout=0.3)

# Train
train(model, loader, epochs=5, lr=1e-3, pos_weight=50.0)
```

### Inference
```python
# Create a transaction sequence (30 timesteps, 12 features)
sample_txn = torch.randn(30, 12)

# Score the transaction
result = score_transaction(model, sample_txn, threshold=0.15)

print(result)
# Output:
# {
#     'fraud_score': 0.2547,
#     'flag': True,  # Fraud alert
#     'risk_tier': 'MEDIUM'
# }
```

## Output Interpretation

**fraud_score**: Probability of fraud (0-1)
- **0.0 - 0.15**: Low risk (normal transaction)
- **0.15 - 0.70**: Medium risk (review recommended)
- **0.70 - 1.0**: High risk (likely fraud)

**flag**: Boolean alert (True = fraud detected)

**risk_tier**: Risk classification
- **LOW**: score < 0.15
- **MEDIUM**: 0.15 ≤ score ≤ 0.70
- **HIGH**: score > 0.70

## Threshold Calibration
The default threshold is **0.15**, which means transactions with fraud_score > 0.15 are flagged.

To adjust for different business needs:
```python
# Higher threshold = fewer false positives, higher false negatives
result = score_transaction(model, sample_txn, threshold=0.30)

# Lower threshold = catch more fraud, more false positives
result = score_transaction(model, sample_txn, threshold=0.10)
```

## Performance Metrics to Monitor
- **Loss**: Binary cross-entropy with class weighting
- **Precision**: Among flagged transactions, how many are actually fraud
- **Recall**: Of all frauds, how many are caught
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under receiver operating characteristic curve

## Dependencies
```
torch>=1.9.0
torch.nn (PyTorch neural network module)
torch.utils.data (Dataset, DataLoader)
numpy
```

## Model Advantages
✅ Handles sequential transaction data naturally
✅ Captures long-range dependencies in transaction history
✅ Attention mechanism identifies key suspicious transactions
✅ Weighted loss handles severe class imbalance
✅ Efficient inference (single forward pass per transaction)

## Limitations & Future Improvements
- Demo uses synthetic data; real-world data has different distributions
- Cold-start problem for new customers (limited history)
- Requires significant labeled fraud examples for production
- Could benefit from ensemble methods
- Explainability could be improved with attention visualization
