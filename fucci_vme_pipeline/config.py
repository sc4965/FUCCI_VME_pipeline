"""Central configuration: channel mapping and all tunable thresholds.

Every parameter that the design doc marked "adjustable later" or "placeholder
until validated" lives here as a single source of truth, so re-tuning never
means touching the stage code itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ChannelRole:
    """Maps a physical acquisition channel to its biological role."""

    wavelength_nm: int
    color_name: str
    fluorophore: str
    role: str  # e.g. "cdt1", "geminin", "slbp", "nuclear_infection"


# Default lookup for the current mNeonGreen-H1.0 construct. This is a
# starting point, not a hardcoded truth: Stage 1 must cross-check each ND2
# file's own channel metadata (name/wavelength) against this table rather
# than assuming channel order, since acquisition setups can vary.
DEFAULT_CHANNEL_MAP: dict[int, ChannelRole] = {
    405: ChannelRole(405, "blue", "mTagBFP2-SLBP(18-126)", "slbp"),
    488: ChannelRole(488, "green", "mNeonGreen-H1.0 / surface-eGFP", "nuclear_infection"),
    561: ChannelRole(561, "red", "mScarlet3-Cdt1(30-120)", "cdt1"),
    640: ChannelRole(640, "magenta", "emiRFP670-Geminin(1-110)", "geminin"),
}


@dataclass
class PipelineConfig:
    """All tunable parameters. Defaults marked PLACEHOLDER need validation
    against real data / annotations before being trusted at scale.
    """

    channel_map: dict[int, ChannelRole] = field(default_factory=lambda: dict(DEFAULT_CHANNEL_MAP))

    # --- Stage 2: segmentation ---
    min_object_area_px: int = 20  # PLACEHOLDER: tune against annotated mitotic frames
    max_nuclear_candidate_area_px: int = 2000  # coarse pre-filter, generous by design
    cellpose_pretrained_model: str = "cpsam_v2"  # current Cellpose-SAM default (as of the models actually installed)
    cellpose_diameter: float | None = None  # None => Cellpose-SAM auto-estimates
    cellpose_flow_threshold: float = 0.4
    cellpose_cellprob_threshold: float = 0.0  # PLACEHOLDER: lower to catch small/dim objects
    require_gpu: bool = True  # fail loudly if no GPU rather than silently running ~minutes/frame on CPU

    # --- Stage 3: tracking ---
    btrack_max_search_radius_um: float = 30.0  # max frame-to-frame displacement for FUCCI-4 tracking
    infected_link_max_distance_um: float = 30.0  # simple-linker gate for the infected population
    track_merge_max_drop_fraction: float = 0.02  # tolerated fraction of objects btrack may reject as false positives

    # --- Stage 4: cell-cycle classification ---
    min_track_coverage: float = 0.6  # fraction of movie frames a track must span to be analyzed
    normal_mphase_reference_min: float = 60.0  # PLACEHOLDER: replace with mock-condition empirical value
    mphase_arrest_multiplier: float = 2.0  # arrested if m_phase_duration > multiplier * reference
    phase_gate_threshold: float = 0.5  # normalized (0-1) cutoff for "high" vs "low" per channel
    mitosis_condensation_threshold: float = 0.6  # PLACEHOLDER: tune against annotated mitotic frames

    # --- Stage 5: infection classification ---
    infection_gate_method: str = "otsu"  # "otsu" | "gmm" | "manual"
    infection_manual_threshold: float | None = None  # required if infection_gate_method == "manual"
    # Otsu/GMM both assume a genuinely bimodal population; on a true mock/
    # uninfected sample (no real second population), they will still force
    # SOME split, which can mislabel the brightest tail of ordinary cells
    # as "infected". "manual" bypasses that assumption entirely with a
    # fixed, user-supplied cutoff.

    # --- Stage 6: neighbor / exposure ---
    neighbor_mode: str = "radius"  # "radius" | "delaunay"
    neighbor_radius_um: float = 50.0
    exposure_decay: str = "inverse_square"  # "inverse_square" | "exponential"
    exposure_max_radius_um: float = 150.0

    # --- Stage 8: dimensionality reduction ---
    n_pca_components: int = 5
    random_state: int = 0

    def channel_role(self, wavelength_nm: int) -> ChannelRole:
        try:
            return self.channel_map[wavelength_nm]
        except KeyError as exc:
            raise KeyError(
                f"No role mapping for wavelength {wavelength_nm} nm. "
                f"Known wavelengths: {sorted(self.channel_map)}. "
                "Check the ND2 file's own channel metadata and update channel_map."
            ) from exc
