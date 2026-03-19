# Debugging Log

## Bug 1: IndexError in `Clusters_IndividualDescription` (Notebook 06)

**File:** `jupyter_notebooks/06 - ModelingEvaluation - Cluster.ipynb` — Cell 31

**Error:**
```
IndexError: index 0 is out of bounds for axis 0 with size 0
```
at line `ClusterDescription[column] = [ClusterDescription[column].values[0]]`

**Root Cause:**
The function `Clusters_IndividualDescription` initializes an empty DataFrame (`pd.DataFrame()` with 0 rows), then iterates over columns assigning values. The **numeric branch** assigned a bare scalar string (e.g., `"25 -- 75"`), but assigning a scalar to a column of an empty DataFrame does **not** create a row — the DataFrame stays at 0 rows. When the **object branch** (triggered by `SalaryBand`) later tried to access `.values[0]`, it failed because the column had 0 elements.

**Fix:**
Wrap all column assignments in a list `[value]` so that each assignment consistently creates exactly 1 row, and simplify the object branch to avoid the redundant `.values[0]` re-assignment.

**Before:**
```python
if EDA_ClusterSubset[column].dtype == 'object':
    ClusterDescription[column] = (
        EDA_ClusterSubset[column]
        .value_counts(normalize=True)
        .head(3)
        .apply(lambda x: f"{100*x:.{decimal_points}f}%")
        .reset_index()
        .apply(lambda row: f"'{row.iloc[0]}': {row.iloc[1]} ", axis=1)
        .str.cat(sep=', ')
    )
    ClusterDescription[column] = [ClusterDescription[column].values[0]]
else:
    ClusterDescription[column] = (
        f"{EDA_ClusterSubset[column].quantile(0.25).round(decimal_points)}"
        f" -- {EDA_ClusterSubset[column].quantile(0.75).round(decimal_points)}"
    )
```

**After:**
```python
if EDA_ClusterSubset[column].dtype == 'object':
    description = (
        EDA_ClusterSubset[column]
        .value_counts(normalize=True)
        .head(3)
        .apply(lambda x: f"{100*x:.{decimal_points}f}%")
        .reset_index()
        .apply(lambda row: f"'{row.iloc[0]}': {row.iloc[1]} ", axis=1)
        .str.cat(sep=', ')
    )
    ClusterDescription[column] = [description if description else 'N/A']
else:
    ClusterDescription[column] = [
        f"{EDA_ClusterSubset[column].quantile(0.25).round(decimal_points)}"
        f" -- {EDA_ClusterSubset[column].quantile(0.75).round(decimal_points)}"
    ]
```

**Key Takeaway:** When building a DataFrame row-by-row via column assignment, always use `[value]` (a list) instead of a bare scalar to ensure a row is actually created.


[BACK TO README.md](Readme.md).