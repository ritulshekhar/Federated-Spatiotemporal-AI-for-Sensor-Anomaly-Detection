"""FedAirGuard — Privacy-Preserving Federated Spatiotemporal AI for Distributed
Air-Quality Monitoring and Sensor Anomaly Detection.

See CLAUDE.md for the full design contract. The short version:

  * Each station is a virtual AIoT edge node that trains a local GRU forecaster.
  * Only weight updates and encoder hidden states ever leave a node — never raw
    readings and never local normalization statistics (§2.4, §7.1).
  * A server-side graph network learns spatial structure from those hidden
    states alone, and a decision layer turns (residual, spatial agreement) into
    a trustworthy / sensor-faulty call for every anomalous reading (§7.4).
  * Nothing is ever reported as a single pooled number (§2.2, §8.3).
"""

__version__ = "0.1.0"
