# Data Access Instructions

The raw data used in this project come from the **U.S. Small Business Administration 7(a) & 504 FOIA dataset**.

Official source:

https://data.sba.gov/dataset/7a-504-foia

## Files Used

The analysis uses the SBA 7(a) files covering:

- FY2010–FY2019
- FY2020–Present

It also uses the official SBA 7(a)/504 data dictionary.

The project uses the data snapshot dated **June 30, 2026**.

## Why the Raw Data Are Not Stored Here

The SBA files are large and are already publicly available from the official source. To keep the GitHub repository smaller and easier to reproduce, the raw CSV files are not committed to this repository.

## Reproducing the Analysis

The notebook can download the required files automatically from the SBA source. If downloading manually, place the source files in:

```text
data/raw/
```

The expected filenames in the final notebook are:

```text
FOIA_7a_FY2010_FY2019_asof_260630.csv
FOIA_7a_FY2020_Present_asof_260630.csv
7a_504_foia_data_dictionary.xlsx
```

After the files are available, run the main notebook from top to bottom.

## Modeling Cutoff

Although the source includes loans after 2021, the prediction model only uses loans that have a complete five-year outcome window.

With a June 30, 2026 snapshot, the latest eligible approval date is:

**June 30, 2021**

Loans approved after this date are not included in the predictive modeling sample.
