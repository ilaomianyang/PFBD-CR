# PFBD-CR

**English** | [简体中文](README.zh-CN.md)

Research code for physics-informed, false-alarm-bounded defense and causal
recovery of meteorological sensor streams.

> The project and repository name were unified to **PFBD-CR**; earlier releases
> and links refer to this code as **TrustMeteo-QC**.

This is a code-only repository. Restricted observations, generated result
tables, model weights, testbed binaries, manuscripts, and working notes are not
included. The repository is an archival implementation release, not a
turnkey dataset distribution.

## Evidence boundary

The frozen V6 evaluation is the primary evidence layer for the associated
study. It uses controlled attack injections on real meteorological observation
backgrounds; these attacks are not production incidents or third-party attack
logs.

The V6 Trust component uses a centered Hampel rolling window. V6 is therefore a
batch evaluation with future information in that component and must not be
described as a strictly causal online detector.

The V9 future-window study was not evaluated and does not contribute to the
reported performance claims. Stage500 remains exploratory and does not replace
the frozen V6 evidence. Mixed and negative transfer results must remain visible.

See [README.zh-CN.md](README.zh-CN.md) for the Chinese project guide.
See [DATA_SOURCE_PAPERS.md](DATA_SOURCE_PAPERS.md) for direct data-source
papers, official product documentation, and the evidence boundary. BibTeX
entries are provided in
[DATA_SOURCE_REFERENCES.bib](DATA_SOURCE_REFERENCES.bib).

## Repository layout

```text
src/                              detector, fusion, recovery, evaluation, and audit code
config/                           historical protocol configuration
confirmatory_package/             frozen V4/V6 protocol and access records
scripts/                          evidence packaging utilities
tests/                            data-independent core tests
vendor/USAD/                      third-party USAD code and BSD license
requirements.txt                  tested core Python dependencies
requirements-optional.txt         optional ERA5 download dependency
README.zh-CN.md                   Chinese project guide
DATA_SOURCE_PAPERS.md             data-source papers and official documentation
DATA_SOURCE_REFERENCES.bib        BibTeX entries for data sources
```

The source archive retains stage numbers because they identify the order and
purpose of the experiment scripts. Many higher-stage scripts consume external
data or frozen intermediate artifacts and are not part of the minimal core
test path.

## Environment

The release environment recorded by the project is Python 3.12 on Windows.
Create an isolated environment and install the pinned core dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The tested package versions are listed in `requirements.txt`. External
download scripts may require additional packages or provider accounts.

## Core tests

Run the data-independent detector, fusion, and recovery tests:

```powershell
python tests\run_tests.py
```

These tests check detector interfaces, spatial outlier response, validation-only
fusion calibration, and the rule that recovery does not rewrite clean rows.

## Reproduction entry points

The final V6 evidence rebuild entry point is:

```powershell
python src\stage470_final_v6_submission_rebuild.py
```

Post-review aggregation, uncertainty, provenance, resource, and package audits
are implemented by the corresponding `stage478` to `stage485` scripts.

These scripts expect a study workspace containing the original provider data
and frozen intermediate artifacts. Those files are intentionally not
redistributed. Protocol JSON files retain machine-specific historical paths as
provenance records; adapt local copies of those paths before running outside
the original workspace.

## Data and credentials

Data remain subject to the terms of ECCC, KNMI, MeteoSwiss, NOAA/NCEI,
Copernicus, FMI, SMHI, MET Norway, iTrust, and the other original providers.
Obtain each dataset from its official source; see
[DATA_SOURCE_PAPERS.md](DATA_SOURCE_PAPERS.md).

Never commit provider credentials to this repository. The ERA5 download path
expects the user's own Copernicus CDS configuration. Other provider adapters
must receive credentials from local configuration or environment variables.

## Citation

Citation metadata is provided in `CITATION.cff`. Replace the software record
with the associated journal article details once the paper is published.
The associated article has not been published yet, so no article DOI is
claimed. Data-source papers and official citations are listed in
[DATA_SOURCE_PAPERS.md](DATA_SOURCE_PAPERS.md).

## License

The code in this repository is released under the MIT License. Third-party
components retain their original licenses; see `THIRD_PARTY_NOTICES.md`.
