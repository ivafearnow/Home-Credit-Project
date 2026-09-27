# EDA Workflow Guide — Home Credit Default Risk

Ivania Zavidich-Fearnow · IS 6812 · Positron + Python + Jupyter

---

## 0. The approach, and why

Your BPS commits to something narrower than "predict default." It commits to **identifying borrower
characteristics that consistently indicate repayment difficulty, using variables that are available and
easy to verify, and expressing the result as a threshold trade-off between approval rate and
repayment-difficulty rate.**

That sentence is the filter for every EDA decision you make. It means:

- **Class imbalance is the headline, not a footnote.** Your BPS's success metric is stated in
  *relative* terms (5% reduction in difficulty rate at equal approval rate). EDA has to establish the
  base rate first, because every later claim is measured against it.
- **Missingness is a finding, not a cleanup chore.** Your BPS promises variables that are "available
  and easy to verify." A predictor that is 70% missing fails the availability test no matter how
  strong its correlation. EDA is where you learn which variables survive that screen.
- **Segment stability matters now, not at modeling time.** Your BPS says results "must be checked for
  stability across major customer segments." Start looking at default rate by segment during EDA so
  the modeling stage inherits a hypothesis instead of a blank page.
- **You do not need the supplementary tables.** The assignment says `application_{train|test}.csv`
  only. Your BPS mentions bureau and previous-application data — name that as scoped future work in
  your results section. That is a strength, not a gap.

The stage-gated workflow from your slides is the right vehicle because it forces *you* to write the
question list. The gate between Chat 2 and Chat 3 is the part that makes the notebook yours.

---

## 1. Positron setup

### Step 1 — Open the right folder

`File → Open Folder…` → **`Home-Credit-Project`** (the repo root, not `Home_Credit_Data_export`).

*Why:* Positron sets the workspace root as the working directory for the Console, the notebook kernel,
and the terminal all at once. Open the repo root and `data/raw/application_train.csv` resolves
identically everywhere — no `../..` fragility when you move a notebook. It also activates the Source
Control pane against your existing git repo, and your `.gitignore` already excludes `data/raw/*`,
which matters because `installments_payments.csv` alone is 723 MB and GitHub rejects files over 100 MB.

You have the CSVs in **both** `Home_Credit_Data_export/` and `data/raw/`. Use `data/raw/` — it is the
path your slides assume and the one your `.gitignore` covers. Consider deleting the duplicate later to
save ~3 GB.

### Step 2 — Pin one interpreter

Click the interpreter picker in the **top-right** of the Positron window. Pick one Python environment
and do not change it mid-project.

*Why:* Positron runs the Console and the notebook kernel through the same session. If the notebook
kernel and the Console disagree about which environment they are in, you get the classic "it works in
the Console but the notebook says ModuleNotFoundError." Pinning one environment also means the HTML
you submit is reproducible.

You have Anaconda installed. Either use a conda env or create a project venv from the Positron
terminal:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy matplotlib seaborn jupyter nbformat nbclient pyarrow
```

Then reselect `.venv` in the interpreter picker. Add `.venv/` to `.gitignore` — it is already there.

### Step 3 — Cache the data as Parquet

`application_train.csv` is 166 MB and 307,511 × 122. Reading it from CSV takes ~20 seconds every time
the kernel restarts, and you will restart a lot.

Run once in the Console:

```python
import pandas as pd
pd.read_csv("data/raw/application_train.csv", low_memory=False).to_parquet("data/interim/app_train.parquet")
pd.read_csv("data/raw/application_test.csv",  low_memory=False).to_parquet("data/interim/app_test.parquet")
```

*Why:* Parquet loads in about a second and preserves dtypes, so you stop fighting `low_memory`
warnings. `data/interim/*` is already gitignored. Your final render still runs top-to-bottom from a
clean kernel — it just runs fast enough that you will actually do it more than once.

### Step 4 — Learn three panes

This is where Positron earns its place over a plain notebook:

- **Variables pane** — every object in the session with its shape and type. Click the table icon next
  to a DataFrame to open it in the Data Explorer.
- **Data Explorer** — sort, filter, and a per-column summary strip showing **% missing**, min/max, and
  distribution. Use it to *find* problems interactively, then write the code that *documents* them.
  Exploration in the Explorer, evidence in the notebook.
- **Plots pane** — keeps a history of every figure. Use the back/forward arrows to compare two
  versions of a plot without re-running code.

*Why this matters for your grade:* the rubric rewards a notebook that reads as a deliberate argument.
If you explore in the Data Explorer and only commit the checks that turned out to be informative, your
notebook stays tight instead of becoming a log of everything you tried.

### Step 5 — Create the notebook

`notebooks/01_eda.ipynb`. Positron opens `.ipynb` in its native notebook editor.

Make the **first cell a Raw cell** (change the cell type from Python to Raw) containing:

```yaml
---
title: "Home Credit Default Risk — Exploratory Data Analysis"
subtitle: "IS 6812 Practice Project"
author: "Ivania Zavidich-Fearnow"
date: today
format:
  html:
    toc: true
    toc-depth: 3
    toc-location: left
    number-sections: true
    embed-resources: true
    code-fold: false
    theme: cosmo
execute:
  echo: true
  warning: false
  message: false
---
```

*Why each line:*

| Setting | Rubric requirement it satisfies |
|---|---|
| `toc: true`, `toc-depth: 3` | "headings and table of contents" |
| `echo: true`, `code-fold: false` | "code chunks revealed" |
| `warning: false`, `message: false` | "warning ... output suppressed" |
| `embed-resources: true` | Single self-contained `.html` — no `_files/` folder to lose when you upload to Canvas |
| `number-sections: true` | Makes the TOC navigable and lets you cite "see §4.2" in your results section |

---

## 2. The six standing checks

These open the notebook, before your questions. Each one below is confirmed against your actual
`application_train.csv` — run them yourself and you should reproduce these numbers.

### 1. Target distribution

`TARGET = 1` is **8.07%** of 307,511 rows.

**Why it matters to your BPS:** a model that predicts "everyone repays" is **91.93% accurate** and
completely useless. This is the "transparent baseline" your BPS promises to beat, and it is exactly
why your success metric is written as approval-rate/difficulty-rate trade-off rather than accuracy.
State this in the notebook — it justifies your whole metric choice.

### 2. Missingness, and does it relate to the target?

**67 of 122 columns** have missing values. The worst are the building-characteristic block
(`COMMONAREA_*`, `NONLIVINGAPARTMENTS_*`, `FONDKAPREMONT_MODE`, `LIVINGAPARTMENTS_*`) at **68–70%**.

The second half of the check is the one people skip. Test whether missingness itself predicts default:

```python
for col in ["EXT_SOURCE_1", "OWN_CAR_AGE", "COMMONAREA_AVG"]:
    miss = df.loc[df[col].isna(), "TARGET"].mean()
    have = df.loc[df[col].notna(), "TARGET"].mean()
    print(f"{col:20s} missing {miss:.2%}  present {have:.2%}")
```

`EXT_SOURCE_1` missing → **8.52%** default; present → **7.50%**. Not dramatic, but real.

**Why it matters:** the difference tells you whether to impute (missingness is noise) or add a
missing-indicator feature (missingness is signal). It also feeds your BPS's availability screen — a
70%-missing column cannot be part of a "focused set of reliable, readily available variables."

### 3. Impossible values

Four confirmed findings:

- **`DAYS_EMPLOYED = 365243`** on **55,374 rows** (18%). That is +1,000 years, and positive where
  every other `DAYS_*` column is negative. It is a sentinel for "not employed." The group's default
  rate is **5.40%** vs **8.66%** for everyone else — these are pensioners, and they are *lower* risk.
  Recode to `NaN` plus a `FLAG_NOT_EMPLOYED` indicator; leaving it in would poison any model.
- **`CODE_GENDER = 'XNA'`** on **4 rows**. Trivial to drop, but say so.
- **`AMT_INCOME_TOTAL` max = 117,000,000** against a 99.9th percentile of **900,000**. A 130× outlier.
- **`OBS_60_CNT_SOCIAL_CIRCLE` max = 344.** An applicant with 344 observable social-circle contacts is
  a data-entry artifact, not a person.

**Why it matters:** this is the "what would an underwriter refuse to believe?" check, and it is the
single most persuasive thing you can put in an EDA notebook. It demonstrates you looked at the data
rather than at a correlation matrix.

### 4. Keys, duplicates, and grain

`SK_ID_CURR` is **unique** in train (307,511 rows = 307,511 ids). **Zero** fully duplicated rows.
**Zero** id overlap between train and test.

**Grain:** one row = one loan application. Not one customer, not one loan. Say this explicitly — it is
the sentence that prevents you from making a customer-level claim you cannot support.

### 5. Temporal direction — what was known at application time?

`DAYS_EMPLOYED` is **the only** `DAYS_*` column with positive values, and every one of those is the
365243 sentinel. So the convention holds: all `DAYS_*` are negative = days *before* application.

Everything in `application_train` is as-of the application date, so there is no leakage inside this
table. Two things to note anyway:

- `AMT_REQ_CREDIT_BUREAU_*` (hour/day/week/mon/qrt/year) are lookups *before* the decision — legitimate.
- This is the check that becomes load-bearing if you later add `installments_payments` or
  `bureau_balance`, where rows exist *after* the application date. Flag it now as the reason those
  tables need careful aggregation.

**Why it matters:** your BPS excludes causal claims and promises a decision usable at application
time. A feature known only after approval would break that promise silently.

### 6. Train/test consistency

Columns: train has **122**, test has **121**. The only difference is **`TARGET`** — clean.

Category levels present in train but **not** in test:

- `NAME_INCOME_TYPE`: `'Maternity leave'`
- `NAME_FAMILY_STATUS`: `'Unknown'`
- `CODE_GENDER`: `'XNA'`

**Why it matters:** these are the levels that will blow up one-hot encoding or produce a column
mismatch at predict time. Decide now — collapse into "Other," or drop the handful of rows. Document
the decision; it belongs in `decisions.md`.

---

## 3. The five chats, adapted

Your slides are written for a generic project. Here they are with your paths, your Python stack, and
your BPS substituted in. Run each as a **separate conversation** in the Positron Assistant — that
isolation is the whole point of the stage gate.

One setup step first: your Chat 1 prompt references `docs/`, but your dictionary lives at
`data/raw/HomeCredit_columns_description.csv`. It has been copied to `docs/` for you.

### Chat 1 → `docs/data_map.md`

> Read the data dictionary in `docs/` and the CSVs in `data/raw/`. Do not do EDA. For each table,
> report: (1) the grain — what one row represents — and the row count; (2) the primary key and whether
> it is unique; (3) foreign keys, the table they join to, and the cardinality; (4) the share of
> `application_train` applicants with at least one row; (5) any column whose values contradict the
> dictionary. Write the result to `docs/data_map.md`.

**Do all nine tables** even though you will only analyze `application_{train,test}`. Item (4) is the
one that pays off: knowing that only ~86% of applicants have a bureau record is the fact that justifies
scoping them out of this stage in your results section.

### Chat 2 → `docs/feature_report.md`

> What features do consumer lenders use to assess repayment risk for thin-file applicants? Cite
> sources — regulator guidance, lender disclosures, or peer-reviewed work; not Kaggle notebooks or
> blog posts. Then read `docs/data_map.md` and, for each feature, state whether this data can build
> it, with the exact columns and tables, or why it cannot. Write to `docs/feature_report.md`. Do not
> compute anything.

Good sources to steer toward: CFPB research on credit invisibility, Basel/OCC model risk guidance
(SR 11-7), the FDIC small-dollar lending guidelines, and the academic literature on alternative data
in credit scoring. **Check every citation it gives you** — this is the step where an assistant is most
likely to invent a plausible-sounding paper.

### The gate — you write `docs/questions.md`

Do not skip this and do not delegate it. Five to eight questions, each traceable to your BPS. Suggested
starting set, derived from your own document:

1. Which applicant characteristics separate repayment difficulty most strongly, and are any of them
   strong enough to matter on their own?
2. Which of those characteristics are both well-populated and easy for an underwriter to verify?
3. Does the relationship between income/credit size and default behave the way an underwriter would
   expect, or is it non-monotonic?
4. Does the default rate vary enough across major segments (gender, education, income type, region
   rating, housing) to threaten the stability requirement?
5. How much redundancy is there among the 122 columns — how many are effectively measuring one thing?
6. What does the distribution of the strongest predictors imply about where a decision threshold could
   sit without collapsing the approval rate?

### Chat 3 → `docs/eda_plan.md`

> Here is my question list. First, check it against the six standing checks (target distribution;
> missingness and its relation to the target; impossible values; keys, duplicates and grain; temporal
> direction — what was known at application time; train/test consistency in columns and category
> levels); for any check not covered, add a question and mark it `[added]`. Then, for each question,
> propose the method, the tables needed, and what the output will look like. Order the questions so
> that nothing depends on a result that comes later. Flag any question this data cannot answer and any
> that is description rather than a question. Write the plan to `docs/eda_plan.md`. Do not run code
> until I approve it.

Read this one closely. The "description rather than a question" flag is the assistant telling you
which of your items will produce a table nobody learns anything from.

### Chat 4 → `notebooks/01_eda.ipynb`

> Execute `docs/eda_plan.md` in `notebooks/01_eda.ipynb`. Open with a "Standing checks" section, one
> subsection per check, then one section per question in the approved order. For each: the question as
> a heading, the code, the output, and one sentence stating what the output shows — no interpretation
> of what it means. Stop and ask if a result makes a later question moot. Use pandas, matplotlib and
> seaborn; every plot gets a title and axis labels. Keep printed output short — `.head()`, not whole
> frames. When finished: is there anything you have missed — against the BPS, the dictionary, the six
> standing checks, and what an underwriter would refuse to believe? Add those as a final section.

The "no interpretation" constraint is doing real work. Rubric item 4 wants *your* interpretive writing
between code chunks. If the assistant writes it, you are submitting its reasoning about a business
problem it did not define.

### The second gate — you write `docs/decisions.md`

For every problem the notebook surfaced, write the decision and the reason. Recode `DAYS_EMPLOYED`.
Drop or keep `XNA`. Impute or flag `EXT_SOURCE_1`. Cap or keep the 117M income. Collapse the
train-only category levels. This file becomes the backbone of your results section and it is what you
hand the modeling stage.

### Chat 5 → critique

> Read `notebooks/01_eda.ipynb` and my interpretations and decisions in `docs/decisions.md`. For each
> decision, argue against it: what in the notebook contradicts it, what was never checked, and what a
> skeptical credit-risk reviewer would ask. Then check each question in `docs/questions.md` was
> actually answered. Do not soften the critique and do not change any decision — list the objections
> and let me rule on them.

Then **you** rule on each objection. Fixing two or three well-chosen ones and writing one line on why
you rejected the rest is a stronger submission than a notebook with no visible disagreement in it.

---

## 4. Rendering to HTML

From the Positron terminal:

```bash
quarto render notebooks/01_eda.ipynb --output-dir ../reports/generated
```

Quarto ships with Positron, and it reads the YAML raw cell from Step 5 — that is how you get the table
of contents. Plain `jupyter nbconvert` produces HTML with **no TOC**, which costs you rubric item 1.

Before you render for submission: **Restart kernel and run all cells.** A notebook that only works
because of a variable defined in a cell you later deleted will render as a wall of tracebacks.

Check the rendered file: TOC on the left, code visible, no yellow warning boxes, every plot has a
title and axis labels, no cell dumping hundreds of rows.

---

## 5. Rubric crosswalk

| # | Requirement | Where it lives |
|---|---|---|
| 1 | Headings + TOC | YAML raw cell; `##`/`###` headings per standing check and question |
| 2 | Business problem statement | Section 1 of the notebook, condensed from your BPS — two paragraphs, not the whole document |
| 3 | Plots + summary tables, titled and labeled | Chat 4 output, one per question |
| 4 | Interpretive writing **between** chunks | **You write this.** One markdown cell after each result: what it means for approval decisions |
| 5 | Code annotation **within** chunks | Comments explaining *why* the step exists, not what the line does |
| 6 | Results section | Final section: data problems found, strongest relationships, how EDA changed your analytics approach |
| 7 | Error-free writing | Read the rendered HTML, not the notebook |
| — | AI-use description | Short final section — see below |

For item 6, you already have three concrete answers from the standing checks: the 365243 sentinel is a
data problem you found; the 8.07% base rate is why your metric is a threshold trade-off rather than
accuracy; and the 68–70% missing block is why your BPS's "readily available variables" criterion
eliminates candidates before any model is fit.

---

## 6. The AI-use section

Be specific — specificity reads as competence. Something close to:

> I used Claude in Positron under a stage-gated workflow: separate conversations for data mapping,
> industry-feature research, planning, execution, and adversarial critique, with my own written
> approval required between stages. I wrote the question list and all interpretive claims myself; the
> assistant produced the data map, the code, and a critique of my decisions. It was most useful for
> mechanical enumeration — cataloguing 122 columns, checking category levels across train and test —
> and least reliable when asked to cite industry sources, where I verified each reference. The
> adversarial critique stage changed [N] of my decisions; I rejected the rest for reasons recorded in
> `docs/decisions.md`.

Fill in `[N]` honestly. A number is more convincing than a claim.

---

## Start here

1. Open `Home-Credit-Project` in Positron.
2. Pin an interpreter; install the packages.
3. Run the Parquet caching snippet.
4. Open a new Assistant conversation and paste the Chat 1 prompt.
