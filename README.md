# Home Credit Default Risk: Approving More of the Right Borrowers

**Ivania Fearnow** · MSBA Practice Capstone (IS 6812) · Python, pandas, scikit-learn, LightGBM, SHAP

A credit-risk model that lets a lender keep approving about 86% of applicants while cutting the default rate on approved loans from 8.1% to 5.4%. Under my profit assumptions, that is worth about 12.5% more portfolio profit than approving everyone.

| | |
| --- | --- |
| **Final model** | LightGBM, 177 features from application data and credit-bureau history |
| **Validation AUC** | 0.7645 (3-fold cross-validation) |
| **Kaggle AUC** | 0.7587 private / 0.7582 public on 48,744 unseen applicants |
| **Recommended policy** | Decline or review applicants above a 15% predicted default risk |
| **Business impact** | Default rate on approved loans 8.1% → 5.4%; estimated profit +12.5% |

---

## Contents

1. [Business Problem and Project Objective](#business-problem-and-project-objective)
2. [My Solution](#my-solution)
3. [Business Value](#business-value)
4. [Difficulties I Encountered](#difficulties-i-encountered)
5. [What I Learned](#what-i-learned)
6. [Notebooks and Repository Guide](#notebooks-and-repository-guide)

---

## Business Problem and Project Objective

Home Credit lends to people with little or no traditional credit history, the customers that banks often turn away. That is its mission, and it is also its risk. In 307,511 past applications, **about 1 in 12 borrowers (8.1%) had trouble repaying.**

Mistakes cost money in both directions:

- **Approving a borrower who defaults** creates credit losses.
- **Declining a borrower who would have repaid** gives up interest income and denies credit to a customer the company exists to serve.

A simple rule can't solve this. Approving everyone is "92% accurate," yet it approves every single defaulter.

**Objective:** build a model that ranks applicants by their risk of repayment difficulty, using information available at the time of application. Then turn that ranking into an approval policy that balances approval volume against losses.

## My Solution

I built an end-to-end pipeline in three stages, with each stage documented in its own notebook.

**1. Exploratory data analysis.** I audited data quality and screened all 122 application fields for predictive value, availability at decision time, and verifiability. The key finding was that **no single variable is strong enough to decide on its own.** The best one, an external credit score, reached an AUC of only 0.654, below the 0.70 bar I set in advance. A model that combines many weak signals was needed.

**2. Data preparation and feature engineering.** I wrote a reproducible script ([`data_preparation.py`](src/features/data_preparation.py)) that fits all of its parameters on training data only and reuses them unchanged on test data. The script:
- Repaired a fake "1,000 years employed" value found in 18% of rows, keeping a flag for it.
- Added missing-value indicators, because missing data turned out to be a risk signal in itself.
- Built affordability ratios: payment-to-income, credit-to-income, and credit-to-goods-price.
- Summarized each applicant's loans at other lenders into 40+ credit-bureau features.

**3. Modeling.** I compared models using ROC AUC instead of accuracy:

| Model | Cross-validated AUC |
| --- | ---: |
| Majority-class baseline | 0.500 |
| Logistic regression (small feature set) | 0.731 |
| Random forest | 0.737 |
| Logistic regression (all features) | 0.742 |
| LightGBM, before tuning (sample) | 0.743 |
| **LightGBM, tuned, full data + bureau history (final)** | **0.765** |

Adding credit-bureau history raised AUC by 0.007. I also tested downsampling, upsampling, SMOTE, and class weighting to handle the class imbalance, and **none of them improved AUC.**

**4. Model card.** I wrote a model card for business decision-makers. It covers the threshold choice, SHAP explanations, a mapping from model features to plain-language denial reasons (adverse action), and a fairness review.

## Business Value

The model ranks applicants well, so the business can choose where to draw the line. At the recommended **15% risk threshold**, measured on out-of-fold predictions that the model did not train on:

- **85.9% of applicants are still approved.** The company keeps serving its core market.
- **The default rate among approved loans drops from 8.1% to 5.4%,** about a third fewer bad loans.
- **43% of future defaulters are caught** before the money goes out the door.
- **Estimated profit rises about 12.5%** compared with approving everyone. That is roughly +1.69 billion in the dataset's currency units across the historical portfolio.

I based the profit estimate on a 14% margin on repaid loans (rounded from Home Credit's reported net interest margin) and a 75% loss on defaulted loans. A sensitivity check moved the best threshold between 0.10 and 0.24 when those assumptions changed, so the threshold should be set together with the finance team.

**My recommendation:** don't use the model for fully automated approvals or denials yet. Instead, run a **human-reviewed pilot with three decision tiers**: auto-approve the safest applicants, send the middle group to an underwriter, and decline or add checks for the riskiest. Before the pilot, three things need to happen:
- Remove or remediate the gender feature.
- Validate the plain-language denial reasons.
- Get legal and compliance sign-off.

## Difficulties I Encountered

- **Accuracy is a misleading metric here.** With only 8% defaults, a model that predicts "no default" for everyone looks 92% accurate. I switched to ROC AUC and framed every result as an approval-versus-default trade-off.
- **Dirty and deceptive data.** 18% of employment records held a placeholder of 365,243 days, about 1,000 years. About 70% of applicants were missing housing details. Simply deleting rows or filling in averages would have thrown away signal, because people with missing housing data defaulted at 9.2% versus 7.0%.
- **Preventing data leakage.** Joining 1.7 million credit-bureau records to applicants without multiplying rows took care. So did making sure that all cleaning and encoding steps were learned only from training folds.
- **Tuning on a sample backfired.** Hyperparameter search on 5,000 rows chose a model that underfit the full 307,000-row dataset. I had to re-check the settings on the full data, which was slow on a laptop.
- **Techniques that "should" help didn't.** SMOTE and resampling, standard advice for imbalanced data, gave no AUC gain. I reported that honestly rather than keeping them for appearance.
- **Validation was slightly optimistic.** Cross-validated AUC (0.765) came in about 0.006 above Kaggle's private score (0.759), a reminder that internal validation overstates real-world performance.
- **Responsible-lending tensions.** The strongest predictors are opaque external scores, and the model initially used gender. Both raise compliance issues that a model can't resolve alone.

## What I Learned

- **Start from the business decision, not the algorithm.** A 0.765 AUC means little to a lender. "Approve 86% and cut defaults by a third" does.
- **Strong models come from many weak signals.** No single variable passed my bar, but together they did. Feature engineering, especially credit-bureau history and affordability ratios, added more than model choice did.
- **Missing data is information.** Sometimes *why* something is missing predicts more than the value would have.
- **Build reproducible pipelines.** Moving logic from notebooks into tested Python modules made it easier to iterate and to trust results.
- **Report negative results.** Saying "class weighting didn't help" is as valuable as a win, and it keeps the analysis honest.
- **Fairness and explainability are part of the job.** A model that can't produce denial reasons or that uses a protected attribute isn't deployable, however accurate it is.
- **Use AI critically.** I used Claude in Positron under a stage-gated workflow. It was strong at mechanical tasks but made confident interpretation errors that I had to catch and correct (see the "How I used AI" section of the EDA notebook).

## Notebooks and Repository Guide

| Notebook | What it covers |
| --- | --- |
| [01 – Exploratory Data Analysis](notebooks/01_eda.ipynb) | Data quality audit, predictor screening, segment stability |
| [02 – Modeling](notebooks/02_modeling.ipynb) | Baseline, model comparison, class imbalance, tuning, bureau features, Kaggle submission |
| [03 – Model Card](notebooks/03_model_card.ipynb) | Threshold and profit analysis, SHAP explainability, adverse action, fairness |

- [`src/`](src/): reusable data-preparation and modeling code
- [`docs/technical_details.md`](docs/technical_details.md): full technical documentation, including every data-preparation decision
- [`docs/`](docs/): EDA plan, decisions log, and data map

**Data:** [Home Credit Default Risk on Kaggle](https://www.kaggle.com/c/home-credit-default-risk). Data files are not included in this repository. To reproduce, download them into `data/raw/` and run `python src/features/data_preparation.py`, then the notebooks in order.
