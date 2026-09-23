# Loan Review Policy Analytics: A/B Testing, Clustering & Deep Learning for Default Prediction

Group project for IB98D0 – Advanced Data Analytics, Warwick Business School. Built on the LendingClub dataset to help a lender reduce default losses while maintaining profitability, using three complementary analytical layers: A/B testing of review policies, K-means borrower segmentation, and a deep learning default classifier.

## Business objective

Reduce losses from loan defaults while maintaining profitability, by getting more customers approved who will actually repay — not simply approving fewer people.

## Part 1 — A/B testing of loan review policies

Two candidate policies compared head-to-head:

- **Policy A (Conservative Screening):** restricted to grades A–C, tight 25% DTI cap, no loan-amount limit.
- **Policy B (Aggressive Growth):** grade-specific DTI and loan-to-income thresholds (looser for stronger grades), opening approvals to a much wider borrower base.

**Metric:** profit per application (recovered principal + interest + fees + recoveries − funded amount − collection recovery fee, divided by total applicants) — chosen because it penalises both bad approvals *and* wrongly rejecting creditworthy borrowers, unlike a pure default-rate metric.

**Result (Welch two-sample t-test, p < 2.2e-16):**

| | Approval rate | Default rate (approved) | Profit/application |
|---|---|---|---|
| Policy A | 65.74% | 12.79% | £687.4 |
| Policy B | 95.36% | 16.05% | **£985.5** |

Policy B generates ~£298 more profit per application despite higher default exposure — the extra volume and pricing more than compensate for the extra risk.

**Limitation flagged explicitly:** the LendingClub dataset only contains historically *approved* loans — rejected applicants are invisible. The A/B test therefore runs on a pre-screened, lower-risk population, so results are strong correlational estimates conditional on the historical portfolio, not a causal guarantee that Policy B increases profit for a representative applicant pool.

## Part 2 — Cluster analysis

**Features:** 11 variables across four risk dimensions — financial capacity (income, DTI, funded amount), credit history (delinquencies, inquiries, derogatory records, open accounts), current credit usage (revolving balance/utilisation, total accounts), and loan pricing (interest rate).

**Preprocessing:** missing values dropped; multicollinearity checked (none found, |r| > 0.75 threshold); multivariate outliers removed via Mahalanobis distance (p < 0.001 → 964 observations, 2.25%, removed, leaving 41,872); z-score standardisation; silhouette analysis run on an 8,000-observation subsample for tractability.

**Method:** K-means, k selected via Elbow Method (narrowed to 2–4) + Silhouette Analysis (k=3 had the best average width, 0.173).

**Three segments identified:**

| Cluster | Size | Avg loan | Avg income | DTI | Default rate | Label |
|---|---|---|---|---|---|---|
| 1 | 16,172 | £18,126 | £87,754 | 19.8% | 19.5% | High risk |
| 2 | 24,028 | £9,908 | £54,853 | 15.6% | 15.2% | Low risk |
| 3 | 1,672 | £11,366 | £63,851 | 15.9% | 18.6% | Medium risk |

Notably, Cluster 1 has the *highest* average income but also the highest default rate — income alone doesn't predict safety once debt burden is accounted for.

**Policy B is more profitable in every cluster**, but the size of the gain varies sharply: Cluster 1 gains £521.60/application (default rate jumps 12.5%→24.4%, but pricing compensates), Cluster 2 gains a more modest £151.80 (best served by a more balanced approach), and Cluster 3 — small and risky — stays profitable under Policy B thanks to its high interest rate (17.6%).

## Part 3 — Deep learning for default prediction

**Target:** `default_flag` (binary). Same 41,872-loan dataset and features as Part 2, split 70/10/20 (train/validation/test) via stratified sampling.

**Baseline:** cluster-average default rate assigned to every borrower in that cluster (19.67% / 15.14% / 18.90%).

**Model:** feed-forward MLP — hidden layers 64/32/16 (ReLU), dropout 0.30/0.20, sigmoid output. Binary cross-entropy loss, Adam optimiser, batch size 64, up to 50 epochs with early stopping (patience 5, best weights restored). SMOTE applied to the training set only (balanced to 24,318/24,318) to address the ~10/90 class imbalance; validation/test sets left untouched. Threshold tuned on the validation set by F1-score (optimal: 0.53), vs a manual 0.17 threshold for the cluster baseline.

**Result:**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Cluster baseline | 0.5702 | 0.1939 | 0.4829 | 0.2767 | 0.5353 |
| **MLP** | **0.6976** | **0.2896** | **0.5338** | **0.3755** | **0.6800** |

The MLP wins on every metric. Crucially, within any single cluster the baseline's ROC-AUC collapses to exactly 0.50 (no discriminatory power — everyone in a cluster gets the same score), while the MLP holds 0.61–0.70 *within* clusters too, showing it captures borrower-level variation that pure segmentation misses. Clustering and deep learning are shown to be complementary rather than substitutes: clustering aids portfolio-level interpretation, the MLP enables individual-level scoring.

## Part 4 — Integrated recommendation

No single uniform policy suits all borrowers. Policy B wins on profit through volume and pricing, not superior risk selection — Policy A is overly cautious for safe borrowers, while aggressive lending spikes defaults specifically in the high-risk cluster. The recommendation is a **segmented, hybrid lending strategy**: cluster-informed policy plus MLP-based continuous risk scoring in place of a single rigid cutoff.

**Recommended next steps before deployment:** calibrate risk thresholds per cluster, apply reject inference to correct for the approved-only selection bias, run time-based (out-of-sample-period) validation, and pilot live against the current policy before full rollout.

## Method / tools

- R (Welch t-test, K-means, Elbow/Silhouette analysis, correlation and outlier diagnostics)
- Python (MLP via a Keras/TensorFlow-style feed-forward network, SMOTE, threshold tuning, ROC/confusion matrix evaluation)
- LendingClub historical loan dataset

## Repo contents

- `IB98D0_Group_Number_5-report.pdf` — full report and appendix (confusion matrices, correlation matrix, dendrogram, silhouette plots, cluster/policy interaction charts, ROC curves, threshold selection plot)
