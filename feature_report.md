# Repayment-risk features for thin-file consumer applicants

## Scope and interpretation

Thin-file applicants have too little traditional credit-report history to support a conventional score reliably. Lenders therefore may combine whatever conventional credit information exists with application data, prior experience with the same lender, and alternative data. The sources below document variables used or considered in real underwriting and peer-reviewed evidence on repayment prediction; they are not Kaggle notebooks or blogs.

This report maps those feature families to the supplied Home Credit data. **Buildable** means the necessary raw fields exist and can be joined at the current-applicant level; it does not mean that the feature has been computed, validated, is complete for every applicant, or is legally appropriate to use. **Partial** means the data provide only a point-in-time value or a narrower proxy than the documented feature. **Not buildable** means the necessary underlying observations are absent.

No applicant-level values were read or analyzed for this report. The mapping uses `data_map.md`, the project column dictionary, and CSV headers only. No features were computed.

## What lenders and the research literature use

The evidence supports the following feature families:

1. **Traditional credit-file performance:** payment history, delinquencies/collections, current and past-due balances, credit utilization, history length, new inquiries, and credit mix. Upstart's lender disclosure identifies payment history, new credit, credit mix, utilization, history length, collections/delinquencies, past-due amounts, public records, and recent inquiries in its underwriting/eligibility framework. The CFPB likewise describes payment record and amount of debt as core credit-score inputs. [Upstart no-action-letter request, pp. 2–5](https://files.consumerfinance.gov/f/documents/201709_cfpb_upstart-no-action-letter-request.pdf); [CFPB alternative-data inquiry](https://www.consumerfinance.gov/archive/newsroom/cfpb-explores-impact-alternative-data-credit-access-consumers-who-are-credit-invisible/)
2. **Income, employment, occupation, education, and affordability:** Upstart disclosed the use of financial and credit variables together with education and current employment, and its eligibility rules included debt-to-income requirements. A recent peer-reviewed study of the same type of U.S. fintech platform reports that education, employment, and other nontraditional variables particularly helped “invisible primes.” [Upstart request, pp. 2–3](https://files.consumerfinance.gov/f/documents/201709_cfpb_upstart-no-action-letter-request.pdf); [Di Maggio and Ratnadiwakara (2026), *Management Science*](https://doi.org/10.1287/mnsc.2024.07854)
3. **Cash-flow and bank-account behavior:** income and expense activity over time, fixed and variable expenses, recurring obligations, and residual balances. The five U.S. federal financial regulators specifically describe these as underwriting metrics that can improve assessment of repayment capacity, including for people with income from multiple sources. [Interagency Statement on Alternative Data in Credit Underwriting, pp. 1–2](https://www.occ.treas.gov/news-issuances/news-releases/2019/nr-ia-2019-142a.pdf)
4. **Non-credit bill-payment history:** rent, utilities, mobile-phone, and cable payments. The CFPB identifies these as alternative data that may reveal whether a thin-file consumer meets recurring obligations. [CFPB Request for Information on Alternative Data](https://files.consumerfinance.gov/f/documents/20170214_cfpb_Alt-Data-RFI.pdf)
5. **Prior relationship with the same lender:** performance on outstanding and repaid loans, including recent on-time payments and outstanding principal. Upstart disclosed explicit returning-borrower rules based on these variables. [Upstart request, p. 2](https://files.consumerfinance.gov/f/documents/201709_cfpb_upstart-no-action-letter-request.pdf)
6. **Requested loan structure:** amount, term/payment burden, price/down payment, product and purpose. These determine the new obligation and are standard application inputs; peer-reviewed work describes loan amount and number of periods among traditional credit variables. [Wang et al. (2024), *Mathematics*](https://doi.org/10.3390/math12182907)
7. **Application, stability, and asset/housing characteristics:** income, age, time at address, time in employment, property values, and related application data appear in the peer-reviewed credit-scoring literature. These variables require fair-lending controls; a predictive association is not itself permission to use a variable. [Djeundje et al. (2021), *Expert Systems with Applications*](https://doi.org/10.1016/j.eswa.2020.113766)
8. **Digital footprints and online behavior:** device/operating system, email characteristics, time and channel of access, and website interaction. Berg et al. find that simple digital-footprint variables predict default even for unscorable consumers. The CFPB's formal inquiry identifies web-browsing, social-media, and related behavioral data as possible alternative sources. [Berg et al. (2020), *Review of Financial Studies*](https://doi.org/10.1093/rfs/hhz099); [CFPB Request for Information on Alternative Data](https://files.consumerfinance.gov/f/documents/20170214_cfpb_Alt-Data-RFI.pdf)
9. **Mobile-phone behavioral metadata:** calling and usage patterns. Peer-reviewed evidence shows that features derived from phone-use records predict repayment among borrowers with thin or nonexistent formal financial histories. [Björkegren and Grissen (2020), *World Bank Economic Review*](https://doi.org/10.1093/wber/lhz006)
10. **Psychometric and related behavioral measures:** personality/behavioral questionnaires and email-usage characteristics have been tested as alternative predictors of consumer default in peer-reviewed work. [Djeundje et al. (2021), *Expert Systems with Applications*](https://doi.org/10.1016/j.eswa.2020.113766)
11. **Social-network information:** peer-to-peer lending research finds that borrowers' online friendship ties can signal credit quality and are associated with subsequent default. [Lin, Prabhala, and Viswanathan (2013), *Management Science*](https://doi.org/10.1287/mnsc.1120.1560)

The list is an evidence-based inventory, not an endorsement. Regulators warn that alternative data can be inaccurate, incomplete, difficult to explain, or correlated with protected characteristics; they call for testing, monitoring, data-quality controls, and compliance with fair-lending, adverse-action, and consumer-reporting requirements. [Interagency Statement](https://www.occ.treas.gov/news-issuances/news-releases/2019/nr-ia-2019-142a.pdf); [CFPB Request for Information on Alternative Data](https://files.consumerfinance.gov/f/documents/20170214_cfpb_Alt-Data-RFI.pdf)

## Feature-by-feature data mapping

All application-table references below apply to both `application_train` and `application_test` unless stated otherwise. `SK_ID_CURR` is the applicant/current-application join key. For historical Home Credit tables, `SK_ID_PREV` identifies the prior application/credit. For bureau monthly history, the actual join key is `SK_ID_BUREAU` (the dictionary incorrectly says `SK_BUREAU_ID`).

### 1. Prior payment performance and delinquency — **Buildable**

The data can form features for whether and how severely prior obligations were late, how much was overdue, and whether scheduled installments were paid on time and in full.

- `bureau`: `SK_ID_CURR`, `SK_ID_BUREAU`, `CREDIT_ACTIVE`, `CREDIT_DAY_OVERDUE`, `AMT_CREDIT_MAX_OVERDUE`, `AMT_CREDIT_SUM_OVERDUE`, `CNT_CREDIT_PROLONG`, `DAYS_ENDDATE_FACT`.
- `bureau_balance`: `SK_ID_BUREAU`, `MONTHS_BALANCE`, `STATUS`. Join to `bureau` on `SK_ID_BUREAU`; `STATUS` supplies monthly delinquency bands/closed/unknown status.
- `POS_CASH_balance`: `SK_ID_CURR`, `SK_ID_PREV`, `MONTHS_BALANCE`, `NAME_CONTRACT_STATUS`, `SK_DPD`, `SK_DPD_DEF`.
- `credit_card_balance`: `SK_ID_CURR`, `SK_ID_PREV`, `MONTHS_BALANCE`, `NAME_CONTRACT_STATUS`, `SK_DPD`, `SK_DPD_DEF`, `AMT_INST_MIN_REGULARITY`, `AMT_PAYMENT_CURRENT`, `AMT_PAYMENT_TOTAL_CURRENT`.
- `installments_payments`: `SK_ID_CURR`, `SK_ID_PREV`, `NUM_INSTALMENT_VERSION`, `NUM_INSTALMENT_NUMBER`, `DAYS_INSTALMENT`, `DAYS_ENTRY_PAYMENT`, `AMT_INSTALMENT`, `AMT_PAYMENT`.

Limitations: `data_map.md` reports only 29.99% measurable train-applicant coverage for `bureau_balance`; some monthly tables have unmatched `SK_ID_PREV` parents; and `installments_payments` has no unique row key because split/repeated payments can occur. A build must preserve multiple payment records rather than assuming one row per installment.

### 2. Outstanding debt, exposure, limits, and utilization — **Buildable**

The data can form current debt, available limit, balance-to-limit, amount overdue, installment burden, and lender-card utilization features.

- `bureau`: `SK_ID_CURR`, `SK_ID_BUREAU`, `CREDIT_ACTIVE`, `AMT_CREDIT_SUM`, `AMT_CREDIT_SUM_DEBT`, `AMT_CREDIT_SUM_LIMIT`, `AMT_CREDIT_SUM_OVERDUE`, `AMT_ANNUITY`, `CREDIT_TYPE`.
- `credit_card_balance`: `SK_ID_CURR`, `SK_ID_PREV`, `MONTHS_BALANCE`, `AMT_BALANCE`, `AMT_CREDIT_LIMIT_ACTUAL`, `AMT_RECEIVABLE_PRINCIPAL`, `AMT_RECIVABLE`, `AMT_TOTAL_RECEIVABLE`, `AMT_INST_MIN_REGULARITY`, `CNT_INSTALMENT_MATURE_CUM`.
- `POS_CASH_balance`: `SK_ID_CURR`, `SK_ID_PREV`, `MONTHS_BALANCE`, `CNT_INSTALMENT`, `CNT_INSTALMENT_FUTURE`.

Limitations: these are reported credit obligations, not a complete household balance sheet. `credit_card_balance` represents only 28.26% of train applicants according to `data_map.md`.

### 3. Credit-history length, recency, mix, and new-credit seeking — **Buildable**

The data can form age of oldest/newest tradeline, time since updates or closure, active/closed mix, credit-type mix, and recent bureau-inquiry features.

- `bureau`: `SK_ID_CURR`, `SK_ID_BUREAU`, `DAYS_CREDIT`, `DAYS_CREDIT_ENDDATE`, `DAYS_ENDDATE_FACT`, `DAYS_CREDIT_UPDATE`, `CREDIT_ACTIVE`, `CREDIT_TYPE`.
- `bureau_balance`: `SK_ID_BUREAU`, `MONTHS_BALANCE`, `STATUS` for the depth of monthly bureau history.
- `application_train` / `application_test`: `AMT_REQ_CREDIT_BUREAU_HOUR`, `AMT_REQ_CREDIT_BUREAU_DAY`, `AMT_REQ_CREDIT_BUREAU_WEEK`, `AMT_REQ_CREDIT_BUREAU_MON`, `AMT_REQ_CREDIT_BUREAU_QRT`, `AMT_REQ_CREDIT_BUREAU_YEAR`.

Limitations: `bureau` covers 85.69% of train applicants, so absence of rows can mean no reported credit or missing coverage and must not automatically be interpreted as good performance. `DAYS_CREDIT_UPDATE` has 17 positive values that contradict its documented “days before application” meaning.

### 4. Income level, source, employment, and occupation — **Buildable for application snapshots; not for verified history**

- `application_train` / `application_test`: `AMT_INCOME_TOTAL`, `NAME_INCOME_TYPE`, `DAYS_EMPLOYED`, `OCCUPATION_TYPE`, `ORGANIZATION_TYPE`, `FLAG_EMP_PHONE`, `FLAG_WORK_PHONE`.

These columns support stated income, income type, employment tenure, occupation, and employer-type features at application. They do **not** provide payroll deposits, employer verification results, multiple-income-source history, income volatility, or a time series of employment. `DAYS_EMPLOYED` also contains the undocumented `365243` sentinel identified in `data_map.md`; it cannot be treated as literal tenure.

### 5. Education — **Buildable**

- `application_train` / `application_test`: `NAME_EDUCATION_TYPE`.

The dataset supports attained education level. It does not contain school identity, field of study, grades, credentials, or education dates. Upstart's disclosure mentions school and degree; therefore only the degree/level portion is represented here.

### 6. Debt-to-income and proposed-payment affordability — **Partially buildable**

- Income and proposed obligation in `application_train` / `application_test`: `AMT_INCOME_TOTAL`, `AMT_CREDIT`, `AMT_ANNUITY`, `AMT_GOODS_PRICE`, `NAME_CONTRACT_TYPE`.
- Existing reported obligations in `bureau`: `AMT_ANNUITY`, `AMT_CREDIT_SUM_DEBT`, `AMT_CREDIT_SUM_OVERDUE`.
- Prior Home Credit scheduled obligations in `POS_CASH_balance`: `CNT_INSTALMENT`, `CNT_INSTALMENT_FUTURE`; in `installments_payments`: `AMT_INSTALMENT`, `DAYS_INSTALMENT`.

These fields can support income-to-payment and debt-burden measures. A complete DTI cannot be guaranteed because the data do not contain verified monthly housing expense, utilities, child support/alimony, taxes, insurance, or all other recurring non-credit obligations. The current application also has no explicit loan-term column, so the proposed payment schedule cannot be reconstructed independently from `AMT_ANNUITY`.

### 7. Bank-account cash flow: deposits, withdrawals, expenses, volatility, and residual balance — **Not buildable**

No table contains checking/deposit-account transactions, merchant/category descriptions, bank-account balances, overdrafts, NSF events, deposit source/frequency, or withdrawal/transfer records. `AMT_INCOME_TOTAL` is a single application value, and `credit_card_balance` is a Home Credit credit-card account history; neither is bank-account cash flow of the kind described by the interagency statement.

### 8. Rent, utility, mobile-phone, and cable payment history — **Not buildable**

There are no bill amounts, due dates, payment dates, arrears, account ages, or service-provider records for rent, utilities, telecom, or cable.

- `NAME_HOUSING_TYPE` and `FLAG_OWN_REALTY` in `application_train` / `application_test` describe housing tenure/ownership, not rent-payment performance.
- `FLAG_MOBIL`, `FLAG_CONT_MOBILE`, `FLAG_PHONE`, `FLAG_WORK_PHONE`, `FLAG_EMP_PHONE`, and `DAYS_LAST_PHONE_CHANGE` describe phone availability/reachability or recency, not mobile-bill payment.

Those columns must not be presented as bill-payment alternative data.

### 9. Prior relationship with Home Credit — **Buildable**

The data can form prior-application outcomes, repeat-customer status, prior terms, outstanding status, repayment timeliness, and use of Home Credit products.

- `previous_application`: `SK_ID_CURR`, `SK_ID_PREV`, `NAME_CLIENT_TYPE`, `NAME_CONTRACT_STATUS`, `DAYS_DECISION`, `CODE_REJECT_REASON`, `NAME_CONTRACT_TYPE`, `AMT_APPLICATION`, `AMT_CREDIT`, `AMT_ANNUITY`, `AMT_DOWN_PAYMENT`, `RATE_DOWN_PAYMENT`, `RATE_INTEREST_PRIMARY`, `RATE_INTEREST_PRIVILEGED`, `CNT_PAYMENT`, `DAYS_FIRST_DRAWING`, `DAYS_FIRST_DUE`, `DAYS_LAST_DUE_1ST_VERSION`, `DAYS_LAST_DUE`, `DAYS_TERMINATION`, `NAME_PAYMENT_TYPE`, `NAME_PRODUCT_TYPE`, `NAME_PORTFOLIO`, `PRODUCT_COMBINATION`.
- `POS_CASH_balance`: `SK_ID_CURR`, `SK_ID_PREV`, `MONTHS_BALANCE`, `CNT_INSTALMENT`, `CNT_INSTALMENT_FUTURE`, `NAME_CONTRACT_STATUS`, `SK_DPD`, `SK_DPD_DEF`.
- `credit_card_balance`: all repayment/exposure fields listed in sections 1 and 2, plus `AMT_DRAWINGS_ATM_CURRENT`, `AMT_DRAWINGS_CURRENT`, `AMT_DRAWINGS_OTHER_CURRENT`, `AMT_DRAWINGS_POS_CURRENT`, `CNT_DRAWINGS_ATM_CURRENT`, `CNT_DRAWINGS_CURRENT`, `CNT_DRAWINGS_OTHER_CURRENT`, `CNT_DRAWINGS_POS_CURRENT`.
- `installments_payments`: all fields listed in section 1.

Limitations: prior applications are not all originated loans. `data_map.md` reports partial parent resolution for monthly/payment histories, although direct `SK_ID_CURR` links resolve. The `365243` date sentinel appears in multiple `previous_application` date fields and needs explicit missing/sentinel treatment in any later build.

### 10. Requested loan amount, payment, product, price/down payment, term, and purpose — **Partially buildable**

- Current loan in `application_train` / `application_test`: `NAME_CONTRACT_TYPE`, `AMT_CREDIT`, `AMT_ANNUITY`, `AMT_GOODS_PRICE`.
- Prior applications in `previous_application`: `NAME_CONTRACT_TYPE`, `AMT_APPLICATION`, `AMT_CREDIT`, `AMT_ANNUITY`, `AMT_DOWN_PAYMENT`, `AMT_GOODS_PRICE`, `RATE_DOWN_PAYMENT`, `RATE_INTEREST_PRIMARY`, `RATE_INTEREST_PRIVILEGED`, `NAME_CASH_LOAN_PURPOSE`, `CNT_PAYMENT`, `NAME_YIELD_GROUP`, `NAME_PRODUCT_TYPE`, `NAME_PORTFOLIO`, `PRODUCT_COMBINATION`, `NFLAG_INSURED_ON_APPROVAL`.

The current loan's amount, annuity, goods price, and broad contract type are available. The current loan's explicit term, interest rate/APR, down payment, and purpose are absent; those fields exist only for prior applications and cannot be substituted for the current request.

### 11. Residence/contact stability and basic application characteristics — **Partially buildable**

- Tenure/stability in `application_train` / `application_test`: `DAYS_REGISTRATION`, `DAYS_ID_PUBLISH`, `DAYS_LAST_PHONE_CHANGE`, `REG_REGION_NOT_LIVE_REGION`, `REG_REGION_NOT_WORK_REGION`, `LIVE_REGION_NOT_WORK_REGION`, `REG_CITY_NOT_LIVE_CITY`, `REG_CITY_NOT_WORK_CITY`, `LIVE_CITY_NOT_WORK_CITY`.
- Housing and location: `NAME_HOUSING_TYPE`, `FLAG_OWN_REALTY`, `REGION_POPULATION_RELATIVE`, `REGION_RATING_CLIENT`, `REGION_RATING_CLIENT_W_CITY`.
- Basic characteristics: `DAYS_BIRTH`, `CNT_CHILDREN`, `CNT_FAM_MEMBERS`.
- Contact availability: `FLAG_MOBIL`, `FLAG_CONT_MOBILE`, `FLAG_PHONE`, `FLAG_WORK_PHONE`, `FLAG_EMP_PHONE`, `FLAG_EMAIL`.

The columns support relative tenure and consistency proxies but not exact address history, dates at multiple addresses, verified contact ownership, or geographic coordinates. Frequent moves can be a misleading instability signal (the CFPB specifically gives military consumers as an example), so use would require careful validation. Age and family-related data also require jurisdiction-specific fair-lending review.

### 12. Assets, ownership, and housing/property characteristics — **Partially buildable**

- `application_train` / `application_test`: `FLAG_OWN_CAR`, `OWN_CAR_AGE`, `FLAG_OWN_REALTY`, `NAME_HOUSING_TYPE`, `APARTMENTS_AVG`, `YEARS_BEGINEXPLUATATION_AVG`, `YEARS_BUILD_AVG`, `LIVINGAREA_AVG`, `TOTALAREA_MODE`, `HOUSETYPE_MODE`, `WALLSMATERIAL_MODE`, `EMERGENCYSTATE_MODE`.

These fields support ownership and selected dwelling-characteristic proxies. They do not supply verified market value, equity, liens, liquid savings, investment balances, retirement assets, or collateral valuation. Thus they cannot form a complete asset/liquidity profile.

### 13. External or proprietary risk scores — **Buildable only as opaque score inputs**

- `application_train` / `application_test`: `EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3`.

The three normalized scores can be used as supplied model inputs. Their providers, source records, construction, timing, and component variables are not documented, so this dataset cannot reproduce them or establish whether they are conventional bureau scores, alternative-data scores, or mixtures. They should not be assigned a substantive interpretation beyond “normalized score from external data source.”

### 14. Digital footprint, device, email-usage, clickstream, and web behavior — **Not buildable**

The dataset has no device type, operating system, browser, IP address, screen characteristics, email-domain/typing behavior, clickstream, session duration, referral channel for the current application, or social-media content/network data.

- `WEEKDAY_APPR_PROCESS_START` and `HOUR_APPR_PROCESS_START` in `application_train` / `application_test` record application timing only.
- `FLAG_EMAIL` records whether an email was provided, not email behavior.
- `CHANNEL_TYPE` in `previous_application` is the acquisition channel for an earlier Home Credit application, not browsing behavior.

Those narrow fields do not support a Berg-style digital-footprint feature set.

### 15. Mobile-phone usage behavior — **Not buildable**

No table contains call-detail records, counts or durations of incoming/outgoing/missed calls, top-ups, handset activity, SMS behavior, contact-network structure, mobility/location traces, or app use. The application flags `FLAG_MOBIL`, `FLAG_CONT_MOBILE`, and `DAYS_LAST_PHONE_CHANGE` are insufficient to construct behavioral phone features of the kind studied by Björkegren and Grissen.

### 16. Psychometric/personality measures — **Not buildable**

No table contains questionnaire items, response timing, personality scales, cognitive/numeracy measures, self-control measures, or other psychometric responses. Application timing and document/contact flags are not valid substitutes.

### 17. Social-network or peer-risk information — **Partially buildable, but only as opaque counts**

- `application_train` / `application_test`: `OBS_30_CNT_SOCIAL_CIRCLE`, `DEF_30_CNT_SOCIAL_CIRCLE`, `OBS_60_CNT_SOCIAL_CIRCLE`, `DEF_60_CNT_SOCIAL_CIRCLE`.

These fields support counts of observable/defaulted members in an undefined “social circle” at 30/60 days past due. The data do not include network edges, relationship types, network size beyond these counts, peer identities, timestamps, or provenance. They therefore cannot support graph/network-behavior features, and the fields carry material privacy, proxy-discrimination, and explainability risk.

## Outcome and leakage boundary

`application_train.TARGET` is the supervised outcome: payment difficulty on the current sample loan. It is **not** an underwriting feature. `application_test` correctly has no `TARGET`. Any later feature build should use only information available at the current application date; relative-time fields and historical tables make that cutoff expressible, but the cutoff must be enforced explicitly. IDs (`SK_ID_CURR`, `SK_ID_PREV`, `SK_ID_BUREAU`) are join keys, not repayment-risk features.

## Overall conclusion

This dataset is strongest for conventional credit history, prior repayment behavior, outstanding credit exposure, stated income/employment, the requested loan's broad economics, and Home Credit's own prior-customer relationship. It is only partial for affordability, residence stability, assets, and social-circle risk. It cannot construct the main alternative-data signals most often proposed for genuinely thin-file applicants: bank-account cash flow, recurring non-credit bill payments, web/device footprints, mobile-phone behavioral metadata, or psychometrics.

That distinction matters: a phone-availability flag is not telecom payment history, a housing-type field is not rent performance, and a monthly Home Credit card balance is not a consumer checking-account cash-flow record.
