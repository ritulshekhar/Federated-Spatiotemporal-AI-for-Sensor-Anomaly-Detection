"""CPCB (India) acquisition — Phase 3 stretch goal, deliberately not implemented.

CLAUDE.md §3.2, §9 Phase 3, §11.

This file is a placeholder with the access routes recorded, not a stub waiting
to be filled in casually. Three constraints have to be satisfied before any of
it runs:

1. **The export schema is unverified** (§11). Column names, units, and whether
   ``pollutant_avg`` is an hourly mean or some other aggregation have not been
   checked against a real downloaded sample. ``harmonize_cpcb`` raises for
   exactly this reason.
2. **No Validity equivalent exists** (§2.6, §3.2). AQICN's own notice: data are
   unvalidated at publication. Every fault label on CPCB data is synthetic,
   tagged ``label_source="synthetic"``, and never blended into a real-fault
   metric.
3. **Phase 1 and Phase 2 must produce real reported numbers first** (§9). Do
   not start this before the full baseline ladder has results.

Access routes, in descending order of usefulness:

* ``app.cpcbccr.com`` — official bulk export. 15-minute data, capped at roughly
  one week per request, so a multi-year pull means hundreds of requests.
* ``data.gov.in`` API — AQI-level min/max/avg. Coarser than the per-pollutant
  hourly structure this project needs; verify the granularity actually matches
  before relying on it.
* OpenAQ — free API but only a 90-day rolling window; the full historical
  archive lives on S3.

When this is implemented, it is framed as a **generalization experiment**
(train on Catalonia, evaluate transfer to India, report the threshold shift as
a result in itself — §5.8), not as a dataset merge for primary training.
"""

from __future__ import annotations

import pandas as pd

CPCB_BULK_PORTAL = "https://app.cpcbccr.com/ccr/#/caaqm-dashboard-all/caaqm-landing"
DATA_GOV_IN_API = "https://api.data.gov.in/resource/3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69"
OPENAQ_API = "https://api.openaq.org/v3"
OPENAQ_S3_ARCHIVE = "s3://openaq-data-archive"


def fetch(*_args, **_kwargs) -> pd.DataFrame:
    raise NotImplementedError(
        "CPCB acquisition is a Phase 3 stretch goal (CLAUDE.md §9). It must not "
        "run until (a) Phase 1 and Phase 2 have reported numbers for every "
        "baseline rung, and (b) the export schema has been verified against a "
        "real downloaded sample (§11). See this module's docstring for the "
        "access routes."
    )
