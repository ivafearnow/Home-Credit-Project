# Interpretation Notes — `01_eda.ipynb`

For each result: what the output actually says, why it matters, and a plain version to rewrite in your
own words. Read the first two parts until the finding makes sense to you. Then write the paragraph
yourself. Don't paste the third part as-is.

---

## S1 — Grain, keys, duplicates

**What I found:** 307,511 training rows, 48,744 test rows. Every row has its own ID. No duplicate
rows. Nobody appears in both files.

**Why it matters:** One row is one *application*, not one *person*. Someone who applied twice shows up
twice, and nothing links those two rows.

**Write something like:**
> Each row here is a loan application, not a customer. If someone applied twice they appear twice, and
> nothing in this file connects those rows. So every rate I report below is a rate per application. I
> should not describe any of them as customer default rates.

---

## S2 — Target distribution

**What I found:** 24,825 of 307,511 applications had payment trouble. That's 8.07%.

**Why it matters:** If you guess "no trouble" for everyone, you're right 91.93% of the time — and you
approve every single person who defaults. High accuracy, zero value.

**Write something like:**
> Only 8.07% of applicants had payment trouble. That means a model that approved everyone would be
> 91.93% accurate and would still approve every borrower who defaults. Accuracy is the wrong way to
> judge this problem. That is why my business problem statement measures success as a trade-off between
> how many people we approve and how many of them run into trouble.

---

## S3 — Train/test consistency

**What I found:** Test has the same columns as train except `TARGET`. Three categories appear only in
train: `XNA` gender (4 rows), `Unknown` family status (2), `Maternity leave` (5). Eleven
`FLAG_DOCUMENT_*` columns have the same value for every test applicant.

**Why it matters:** A category the model never saw in training will break it at scoring time. A column
that never changes can't separate anybody.

**Write something like:**
> The two files line up well. The only column missing from test is `TARGET`, which is expected. But
> three categories show up in training and never in test, so I need to handle them or the model will
> hit a value it has never seen when it scores a real applicant. Eleven of the document flag columns
> have the same value for every test applicant, which means they cannot tell anyone apart in the
> population I actually have to score.

---

## S4 — Impossible values

**What I found:** `DAYS_EMPLOYED = 365243` in 55,374 training rows — 18% of the data. That's about
1,000 years. Also: 4 rows with `XNA` gender, 1 test row with an impossible region rating of −1, income
up to 117 million, social-circle counts up to 348.

**Why it matters:** 365243 isn't a measurement, it's a placeholder meaning "no job record." Left alone
it poisons every average. Deleted, you lose 18% of your data — and specifically the retirees, who
default *less*.

**Write something like:**
> My biggest data problem is `DAYS_EMPLOYED`. In 55,374 rows, about 18% of the data, it equals 365243,
> which is roughly 1,000 years. That is not a real number. It is a placeholder for applicants with no
> employment record. I replaced it with a missing value and added a flag column, so I lose the fake
> number but keep the useful fact that the person is not employed. I did not delete these rows. They
> are 18% of my data and they behave differently from everyone else, so dropping them would change the
> population I am studying. I did remove the 4 rows with an `XNA` gender, since 4 out of 307,511
> cannot affect anything. But I repaired the one test row with an impossible region rating instead of
> deleting it, because every test applicant has to get a prediction.

---

## S5 — Temporal direction

**What I found:** After fixing the sentinel, every `DAYS_*` column is negative. The three `EXT_SOURCE`
columns have no documentation at all.

**Why it matters:** Negative means "before the application," so nothing here leaks information from
after the loan was made. Except we can't check that for `EXT_SOURCE`.

**Write something like:**
> After fixing `DAYS_EMPLOYED`, every date column is negative, which matches the dictionary. Negative
> means the event happened before the application, so nothing in this table describes what happened
> after the loan was approved. The exception is the three `EXT_SOURCE` columns. The dictionary only
> says they are normalized scores from an outside source. I do not know who built them, what goes into
> them, or when they were measured, so I cannot confirm they were known at application time.

---

## S6 — Missingness

**What I found:** Applicants missing `HOUSETYPE_MODE` default at 9.15%. Applicants who have it default
at 6.99%. The building columns are all missing on the *same* rows, about 68–70% of them.

**Why it matters:** The missingness itself predicts. Fill those blanks with an average and you delete
the signal.

**Write something like:**
> Missing values here are not random. Applicants missing `HOUSETYPE_MODE` default at 9.15%, while
> applicants who have it default at 6.99%. The fact that a value is missing tells me something on its
> own. The building columns are also all missing on the same rows, which suggests one property data
> source that only covers some applicants. If I filled these in with an average I would erase that
> signal, so I would add a flag marking the value as missing instead. It also means these columns fail
> the availability test in my business problem statement, since they are empty for about 70% of
> applicants.

---

## Q2b — Redundancy

**What I found:** 71 pairs of numeric columns correlate above 0.90. They collapse into 15 groups,
cutting 121 columns down to 85 usable ones. The biggest group has ten members.

**Why it matters:** You don't have 122 pieces of evidence. You have about 85.

**Write something like:**
> The 122 columns describe far fewer than 122 different things. Seventy-one pairs of numeric columns
> correlate above 0.90 and collapse into 15 groups. The largest group has ten members, all versions of
> the same building measurements ending in `_AVG`, `_MODE` and `_MEDI`. This matters for two reasons. A
> simple logistic model produces unstable coefficients when I feed it three copies of the same
> measurement. And if I listed ten correlated building columns as ten separate risk factors, I would be
> overstating how much independent evidence I actually have.

---

## Q1 — Single-variable separation

**What I found:** Nothing passed. Best is `EXT_SOURCE_2` at AUC 0.654. Best one you can actually
explain is `DAYS_BIRTH` at 0.585. Your bar was 0.70.

**Why it matters:** This is the main conclusion of the whole notebook. No single fact about an
applicant is enough.

**Write something like:**
> Before running this I set the bar for a variable standing on its own: an AUC of at least 0.70, plus a
> riskiest-tenth to safest-tenth default ratio of at least 2.0. Nothing passed. AUC is the chance that
> if I pick one applicant who defaulted and one who did not, the variable ranks the defaulter as
> riskier — 0.5 would be a coin flip. My strongest variable, `EXT_SOURCE_2`, reached 0.654. The
> strongest one I can actually explain, `DAYS_BIRTH`, reached 0.585. I predicted this before I ran the
> analysis and the result matched. This is the most important thing I learned: no single fact about an
> applicant is enough to make a decision, so my recommendation has to be a model that combines several
> weak signals rather than a rule based on one column.

**The surprise, in plain terms:** all three `EXT_SOURCE` columns passed your decile test easily — the
riskiest tenth defaults about six times as often as the safest tenth — but failed on AUC. That looks
contradictory and isn't. The decile ratio only compares the two extremes. AUC checks how well the
variable orders *everybody*. So these scores spot the very safest and very riskiest applicants well,
and can't rank the big middle group.

> One result surprised me. All three `EXT_SOURCE` columns passed my decile test easily, with the
> riskiest tenth defaulting about six times as often as the safest tenth, but they still failed on AUC.
> The two measures disagree because they look at different things. The decile ratio only compares the
> extremes, while AUC checks how well the variable ranks everyone. So these scores are good at spotting
> the very safest and very riskiest applicants, and poor at ordering the large middle group. That makes
> them useful for sending borderline cases to manual review, but not good enough to score everyone.

---

## Q1b — Money variables

**What I found:** You predicted default would rise with both annuity-to-income and credit-to-income.
Annuity-to-income did (rank correlation 0.87). Credit-to-income didn't — it came out flat (−0.04).
Income falls from 8.19% default in the lowest tenth to 6.14% in the highest. Bigger loans default
*less*.

**Why it matters:** The prediction you got wrong is the most interesting thing in this section.

**Write something like:**
> I wrote down what I expected before I looked. I expected default to rise as both annuity-to-income
> and credit-to-income went up. Annuity-to-income did exactly that, with a rank correlation of 0.87.
> Credit-to-income did not — it came out flat at −0.04. I think this means the monthly payment compared
> to income is what predicts trouble, not the total size of the loan compared to income. A large loan
> spread over a long term has a small monthly payment, so grouping loans by total size hides the burden
> that actually causes missed payments. I would carry the annuity measure into the modeling stage and
> drop credit-to-income.
>
> Income itself behaves as I expected but weakly, falling from 8.19% default in the lowest tenth to
> 6.14% in the highest. Loan amount runs the opposite way, with larger loans defaulting less. I do not
> read that as large loans being safer. It more likely reflects who qualifies for a large loan in the
> first place. Everyone in this data was already approved, so these are patterns inside an approved
> population, not causes.

---

## Q2c — The three-way screen

> **Numbers below are predicted, not confirmed.** The cell hasn't been re-run since the threshold fix.
> Check each figure against your own output before you write anything.

**What I found:** With "meaningful signal" now defined as AUC ≥ 0.55, four variables should pass —
`DAYS_BIRTH`, `DAYS_EMPLOYED`, `ORGANIZATION_TYPE`, `NAME_INCOME_TYPE`. All three `EXT_SOURCE` columns
fail on verifiability despite being the strongest predictors in the dataset. Two variables just miss:
`OCCUPATION_TYPE` on coverage (68.7%), `NAME_EDUCATION_TYPE` on AUC (0.546).

**Why it matters:** This is the shortlist your whole business problem statement was pointing at — and
the things it excludes cost you something you can put a number on: 0.654 versus 0.585.

**Write something like:**
> Putting all three tests together, four variables pass: `DAYS_BIRTH`, `DAYS_EMPLOYED`,
> `ORGANIZATION_TYPE` and `NAME_INCOME_TYPE`. Each is something an underwriter can check against a
> document, each is present for at least 80% of applicants, and each separates the two groups by a
> clear margin. None of them is strong on its own. The best reaches an AUC of 0.585.
>
> What failed is the more interesting half. All three `EXT_SOURCE` columns failed, and they are the
> strongest predictors in this data. They did not fail on performance. They failed because I cannot say
> where they come from, `EXT_SOURCE_1` is present for only 43.6% of applicants, and I could not give a
> declined applicant a real reason if that reason is a score nobody can explain. I can measure what the
> exclusion costs: `EXT_SOURCE_2` reaches 0.654 and my best passing variable reaches 0.585. That gap is
> what Home Credit gives up by restricting itself to predictors it can defend.
>
> Two variables came close and I left them in Review rather than moving the line to admit them.
> `OCCUPATION_TYPE` has enough signal but is missing for 31% of applicants. `NAME_EDUCATION_TYPE` has
> full coverage but an AUC of 0.546, just under my threshold. Education is worth a note, because its
> segment spread is large — 10.93% default at lower secondary against 5.36% for higher education —
> while its AUC stays low. The reason is that the high-risk category is only 1.2% of applicants, so a
> big gap affecting very few people does not help me rank the population.
>
> I should be clear about where the 0.55 threshold came from. I set it after seeing the results, not
> before, so it is a judgment call rather than a pre-registered rule. I chose a round number clear of
> the noise floor and well below the 0.70 bar I had set for a variable standing on its own. It is also
> worth saying that the confidence intervals here are only about ±0.008 wide, because the sample has
> over 300,000 rows. At that size almost any small relationship is statistically significant, so
> significance is not a useful filter here. What I am judging is the size of the effect, and 0.55 is my
> judgment about size.

**If your output differs from the four names above,** don't rewrite the numbers to match mine — tell me
what you got and we'll work out why.

---

## Q3 — Segments

**What I found:** Renters default at 12.31%, people living with parents at 11.70%, people in their own
home at 7.80%. Worst-rated regions 11.40% versus 4.84% in the best. Education runs 10.93% down to
5.36%. Five segments crossed your 1.25× line, none hit 2.0×. **No predictor reversed direction
anywhere.**

**Why it matters:** The reversal check is the real result. Nothing flipped, so your predictors are
stable across groups.

**Write something like:**
> Default rates differ a lot by segment. People renting default at 12.31% and people living with
> parents at 11.70%, against 7.80% for people in their own home. The lowest-rated regions default at
> 11.40% and the highest-rated at 4.84%. Education runs from 10.93% at lower secondary down to 5.36%
> for higher education. Five segments crossed the 1.25 times line I set in advance, but none reached
> 2.0.
>
> The more useful result is that nothing reversed direction. `EXT_SOURCE_2`, `EXT_SOURCE_3` and
> `DAYS_BIRTH` point the same way in every segment I checked. I treated a reversal as the serious
> problem, because a variable that means "riskier" for one group and "safer" for another cannot be used
> under a single rule. That did not happen anywhere. This is evidence that these relationships are
> stable inside this market, which is the closest I can get to my BPS's concern about whether they
> would transfer to a new market. It is not the same as testing a new market.
>
> One segment shows why I set a minimum group size. `NAME_INCOME_TYPE = Other` has the highest default
> rate in the table at 18.18%, but only 55 applicants, and its confidence interval runs from 10.2% to
> 30.3%. I left it unflagged because 55 people cannot support a conclusion.

**[YOUR CALL]** `CODE_GENDER` crossed your threshold (men 10.14%, women 7.00%). Sex is a protected
characteristic under U.S. lending law and cannot be used in underwriting. Say in one sentence whether
you're reporting it as a fairness check or excluding it. Leaving it in a risk-factor table without
comment is what a reviewer will catch.

---

## Results section

Your closing section needs three things. You have all of them:

**Data problems I found** — the 1,000-year employment code in 18% of rows; a block of building columns
missing for about 70% of applicants, where the missingness itself predicts default; eleven document
flags that never vary in the test set; three categories that exist only in training; one impossible
region rating in test.

**What turned out to be strongly related** — nothing, on its own. Best single variable reached 0.654
against a 0.70 bar. You predicted that in advance and it held.

**How EDA changed your approach** — **[YOUR CALL]**, pick the one you actually believe:

1. No single predictor works, so the transparent baseline has to be a multivariable model, not a
   one-column rule.
2. The verifiability screen removes your three strongest predictors, so the real question becomes how
   much performance responsible lending costs — a different and better question than maximizing AUC.
3. The annuity result redirects feature engineering toward payment burden and away from loan size.

Option 3 is the narrowest and best-evidenced. Option 2 is the one that makes your project different
from a Kaggle notebook.
