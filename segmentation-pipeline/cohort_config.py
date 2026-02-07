"""Cohort-specific class remapping for Dice coefficient analysis.

When evaluating a segmentation model against ground truth annotations from a
specific training cohort, the model's raw class indices must be remapped to the
cohort's target class scheme.  For example, the breast model's raw class 3
(DCIS) maps to target class 1 (Tumor) when evaluated against the breast cohort
ground truth.

The target scheme is deliberately simplified to two non-background classes --
**Tumor** (index 1) and **Tumor_Stroma** (index 2) -- because the downstream
R analysis (``figures-for-pub.qmd``, Figure 3) expects exactly the CSV columns
``DICE_Tumor`` and ``DICE_Tumor_Stroma``.

For the prostate cohort, target class 2 represents *Benign* tissue rather than
tumour stroma, but the CSV column is still named ``DICE_Tumor_Stroma`` to keep
the output schema uniform.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Cohort(StrEnum):
    """Training cohorts with ground truth annotations available."""

    BREAST = "BRCA"
    COADREAD = "COADREAD"
    LUNG = "LUNG"
    PROSTATE = "PRAD"


class ModelKey(StrEnum):
    """Model identifiers matching ``segment.py``'s ``Model`` enum names."""

    BREAST = "BREAST"
    COLON = "COLON"
    LUNG = "LUNG"
    KIDNEY = "KIDNEY"
    PROSTATE = "PROSTATE"


# ---------------------------------------------------------------------------
# Cohort specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CohortSpec:
    """Target class schema and per-model remapping for one evaluation cohort.

    Attributes
    ----------
    dir_name:
        Directory name used in the analysis output tree, e.g.
        ``"03_BRCA_COHORT"``.  Must match the pattern the R code uses to
        extract cohort names (``[A-Z]+(?=_COHORT)``).
    class_names:
        Mapping of target class index to human-readable name.
    model_maps:
        For each :class:`ModelKey`, a dict mapping the model's *raw* class
        index (as produced by ``segment.py``) to the cohort's *target* class
        index.
    """

    dir_name: str
    class_names: dict[int, str]
    model_maps: dict[ModelKey, dict[int, int]] = field(repr=False)


# ---------------------------------------------------------------------------
# Breast cohort  (target: 0=BG, 1=Tumor, 2=TumorStroma)
# ---------------------------------------------------------------------------

_BREAST_MAPS: dict[ModelKey, dict[int, int]] = {
    ModelKey.BREAST: {
        0: 0,  # Background
        1: 1,  # TUMOR -> Tumor
        2: 2,  # TUMOR STROMA -> Tumor_Stroma
        3: 1,  # DCIS -> Tumor
        4: 1,  # LCIS -> Tumor
        5: 0,  # NECROSIS -> Background
        6: 0,  # MUCIN -> Background
        7: 0,  # INFLAM -> Background
        8: 0,  # FAT -> Background
        9: 0,  # STROMA -> Background
        10: 0,  # BLOOD -> Background
        11: 0,  # SKIN -> Background
        12: 0,  # BEN EPIT / SKIN ADNEX -> Background
        13: 0,  # BACK -> Background
    },
    ModelKey.LUNG: {
        0: 0,  # Background
        1: 1,  # TUMOR -> Tumor
        2: 2,  # TUMOR STROMA -> Tumor_Stroma
        3: 0,  # NECROSIS -> Background
        4: 0,  # MUCIN -> Background
        5: 0,  # BENIGN LUNG -> Background
        6: 0,  # STROMA etc. -> Background
        7: 0,  # BLOOD -> Background
        8: 0,  # BRONCHUS -> Background
        9: 0,  # CARTILAGE -> Background
        10: 0,  # GLAND_BRONCH -> Background
        11: 0,  # LYMPH -> Background
        12: 0,  # BACK -> Background
    },
    ModelKey.COLON: {
        0: 0,  # Background
        1: 1,  # TUMOR / ADENOM_HG -> Tumor
        2: 0,  # MUC / ADENOM_LG -> Background
        3: 2,  # TU_STROMA -> Tumor_Stroma
        4: 0,  # SUBMUC -> Background
        5: 0,  # MUSC_PROP / MUSC_MUC -> Background
        6: 0,  # ADVENT / VESSEL -> Background
        7: 0,  # LYMPH -> Background
        8: 0,  # ULCUS / NECROSIS -> Background
        9: 0,  # BLOOD -> Background
        10: 0,  # MUCIN -> Background
        11: 0,  # BACK -> Background
    },
    ModelKey.KIDNEY: {
        0: 0,  # Background
        1: 1,  # TUMOR -> Tumor
        2: 1,  # TUMOR_REGRESS -> Tumor
        3: 0,  # NECROSIS -> Background
        4: 0,  # KIDNEY_BENIGN -> Background
        5: 0,  # UROTHEL -> Background
        6: 0,  # FAT -> Background
        7: 2,  # STROMA -> Tumor_Stroma
        8: 0,  # BLOOD -> Background
        9: 0,  # ADRENAL -> Background
        10: 0,  # BACK -> Background
    },
    ModelKey.PROSTATE: {
        0: 0,  # Background
        1: 1,  # TUMOR -> Tumor
        2: 0,  # N (normal epithelium) -> Background
        3: 0,  # N_STR (normal stroma) -> Background
        4: 0,  # BACK -> Background
    },
}

# ---------------------------------------------------------------------------
# Colorectal (COADREAD) cohort  (target: 0=BG, 1=Tumor, 2=TumorStroma)
# ---------------------------------------------------------------------------

_COADREAD_MAPS: dict[ModelKey, dict[int, int]] = {
    ModelKey.BREAST: {
        0: 0,
        1: 1,  # TUMOR -> Tumor
        2: 2,  # TUMOR STROMA -> Tumor_Stroma
        3: 1,  # DCIS -> Tumor
        4: 1,  # LCIS -> Tumor
        5: 0,  # NECROSIS -> Background
        6: 0,  # MUCIN -> Background
        7: 0,  # INFLAM -> Background
        8: 0,  # FAT -> Background
        9: 0,  # STROMA -> Background
        10: 0,  # BLOOD -> Background
        11: 0,  # SKIN -> Background
        12: 0,  # BEN EPIT / SKIN ADNEX -> Background
        13: 0,  # BACK -> Background
    },
    ModelKey.LUNG: {
        0: 0,
        1: 1,  # TUMOR -> Tumor
        2: 2,  # TUMOR STROMA -> Tumor_Stroma
        3: 0,  # NECROSIS -> Background
        4: 0,  # MUCIN -> Background
        5: 0,  # BENIGN LUNG -> Background
        6: 0,  # STROMA etc. -> Background
        7: 0,  # BLOOD -> Background
        8: 0,  # BRONCHUS -> Background
        9: 0,  # CARTILAGE -> Background
        10: 0,  # GLAND_BRONCH -> Background
        11: 0,  # LYMPH -> Background
        12: 0,  # BACK -> Background
    },
    ModelKey.COLON: {
        0: 0,
        1: 1,  # TUMOR / ADENOM_HG -> Tumor
        2: 0,  # MUC / ADENOM_LG -> Background
        3: 2,  # TU_STROMA -> Tumor_Stroma
        4: 0,  # SUBMUC -> Background
        5: 0,  # MUSC_PROP / MUSC_MUC -> Background
        6: 0,  # ADVENT / VESSEL -> Background
        7: 0,  # LYMPH -> Background
        8: 0,  # ULCUS / NECROSIS -> Background
        9: 0,  # BLOOD -> Background
        10: 0,  # MUCIN -> Background
        11: 0,  # BACK -> Background
    },
    ModelKey.KIDNEY: {
        0: 0,
        1: 1,  # TUMOR -> Tumor
        2: 1,  # TUMOR_REGRESS -> Tumor
        3: 0,  # NECROSIS -> Background
        4: 0,  # KIDNEY_BENIGN -> Background
        5: 0,  # UROTHEL -> Background
        6: 0,  # FAT -> Background
        7: 2,  # STROMA -> Tumor_Stroma
        8: 0,  # BLOOD -> Background
        9: 0,  # ADRENAL -> Background
        10: 0,  # BACK -> Background
    },
    ModelKey.PROSTATE: {
        0: 0,
        1: 1,  # TUMOR -> Tumor
        2: 0,  # N -> Background
        3: 0,  # N_STR -> Background
        4: 0,  # BACK -> Background
    },
}

# ---------------------------------------------------------------------------
# Lung cohort  (target: 0=BG, 1=Tumor, 2=TumorStroma)
# ---------------------------------------------------------------------------

_LUNG_MAPS: dict[ModelKey, dict[int, int]] = {
    ModelKey.BREAST: {
        0: 0,
        1: 1,  # TUMOR -> Tumor
        2: 2,  # TUMOR STROMA -> Tumor_Stroma
        3: 1,  # DCIS -> Tumor
        4: 1,  # LCIS -> Tumor
        5: 0,  # NECROSIS -> Background
        6: 0,  # MUCIN -> Background
        7: 0,  # INFLAM -> Background
        8: 0,  # FAT -> Background
        9: 0,  # STROMA -> Background
        10: 0,  # BLOOD -> Background
        11: 0,  # SKIN -> Background
        12: 0,  # BEN EPIT / SKIN ADNEX -> Background
        13: 0,  # BACK -> Background
    },
    ModelKey.LUNG: {
        0: 0,
        1: 1,  # TUMOR -> Tumor
        2: 2,  # TUMOR STROMA -> Tumor_Stroma
        3: 0,  # NECROSIS -> Background
        4: 0,  # MUCIN -> Background
        5: 0,  # BENIGN LUNG -> Background
        6: 0,  # STROMA etc. -> Background
        7: 0,  # BLOOD -> Background
        8: 0,  # BRONCHUS -> Background
        9: 0,  # CARTILAGE -> Background
        10: 0,  # GLAND_BRONCH -> Background
        11: 0,  # LYMPH -> Background
        12: 0,  # BACK -> Background
    },
    ModelKey.COLON: {
        0: 0,
        1: 1,  # TUMOR / ADENOM_HG -> Tumor
        2: 0,  # MUC / ADENOM_LG -> Background
        3: 2,  # TU_STROMA -> Tumor_Stroma
        4: 0,  # SUBMUC -> Background
        5: 0,  # MUSC_PROP / MUSC_MUC -> Background
        6: 0,  # ADVENT / VESSEL -> Background
        7: 0,  # LYMPH -> Background
        8: 0,  # ULCUS / NECROSIS -> Background
        9: 0,  # BLOOD -> Background
        10: 0,  # MUCIN -> Background
        11: 0,  # BACK -> Background
    },
    ModelKey.KIDNEY: {
        0: 0,
        1: 1,  # TUMOR -> Tumor
        2: 1,  # TUMOR_REGRESS -> Tumor
        3: 0,  # NECROSIS -> Background
        4: 0,  # KIDNEY_BENIGN -> Background
        5: 0,  # UROTHEL -> Background
        6: 0,  # FAT -> Background
        7: 2,  # STROMA -> Tumor_Stroma
        8: 0,  # BLOOD -> Background
        9: 0,  # ADRENAL -> Background
        10: 0,  # BACK -> Background
    },
    ModelKey.PROSTATE: {
        0: 0,
        1: 1,  # TUMOR -> Tumor
        2: 0,  # N -> Background
        3: 0,  # N_STR -> Background
        4: 0,  # BACK -> Background
    },
}

# ---------------------------------------------------------------------------
# Prostate cohort  (target: 0=BG, 1=Tumor, 2=Benign)
#
# The prostate model only has 4 classes; other models' non-tumour classes
# are collapsed into class 2 (Benign).  The CSV column is still called
# "Tumor_Stroma" for schema uniformity.
# ---------------------------------------------------------------------------

_PROSTATE_MAPS: dict[ModelKey, dict[int, int]] = {
    ModelKey.BREAST: {
        0: 0,
        1: 1,  # TUMOR -> Tumor
        2: 2,  # TUMOR STROMA -> Benign
        3: 1,  # DCIS -> Tumor
        4: 1,  # LCIS -> Tumor
        5: 2,  # NECROSIS -> Benign
        6: 2,  # MUCIN -> Benign
        7: 2,  # INFLAM -> Benign
        8: 2,  # FAT -> Benign
        9: 2,  # STROMA -> Benign
        10: 2,  # BLOOD -> Benign
        11: 2,  # SKIN -> Benign
        12: 2,  # BEN EPIT / SKIN ADNEX -> Benign
        13: 0,  # BACK -> Background
    },
    ModelKey.LUNG: {
        0: 0,
        1: 1,  # TUMOR -> Tumor
        2: 2,  # TUMOR STROMA -> Benign
        3: 2,  # NECROSIS -> Benign
        4: 2,  # MUCIN -> Benign
        5: 2,  # BENIGN LUNG -> Benign
        6: 2,  # STROMA etc. -> Benign
        7: 2,  # BLOOD -> Benign
        8: 2,  # BRONCHUS -> Benign
        9: 2,  # CARTILAGE -> Benign
        10: 2,  # GLAND_BRONCH -> Benign
        11: 2,  # LYMPH -> Benign
        12: 0,  # BACK -> Background
    },
    ModelKey.COLON: {
        0: 0,
        1: 1,  # TUMOR / ADENOM_HG -> Tumor
        2: 2,  # MUC / ADENOM_LG -> Benign
        3: 2,  # TU_STROMA -> Benign
        4: 2,  # SUBMUC -> Benign
        5: 2,  # MUSC_PROP / MUSC_MUC -> Benign
        6: 2,  # ADVENT / VESSEL -> Benign
        7: 2,  # LYMPH -> Benign
        8: 2,  # ULCUS / NECROSIS -> Benign
        9: 2,  # BLOOD -> Benign
        10: 2,  # MUCIN -> Benign
        11: 0,  # BACK -> Background
    },
    ModelKey.KIDNEY: {
        0: 0,
        1: 1,  # TUMOR -> Tumor
        2: 1,  # TUMOR_REGRESS -> Tumor
        3: 2,  # NECROSIS -> Benign
        4: 2,  # KIDNEY_BENIGN -> Benign
        5: 2,  # UROTHEL -> Benign
        6: 2,  # FAT -> Benign
        7: 2,  # STROMA -> Benign
        8: 2,  # BLOOD -> Benign
        9: 2,  # ADRENAL -> Benign
        10: 0,  # BACK -> Background
    },
    ModelKey.PROSTATE: {
        0: 0,
        1: 1,  # TUMOR -> Tumor
        2: 2,  # N (normal epithelium) -> Benign
        3: 2,  # N_STR (normal stroma) -> Benign
        4: 0,  # BACK -> Background
    },
}

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_STANDARD_CLASS_NAMES = {0: "Background", 1: "Tumor", 2: "Tumor_Stroma"}
_PROSTATE_CLASS_NAMES = {0: "Background", 1: "Tumor", 2: "Tumor_Stroma"}  # "Benign" internally

COHORT_SPECS: dict[Cohort, CohortSpec] = {
    Cohort.BREAST: CohortSpec(
        dir_name="03_BRCA_COHORT",
        class_names=_STANDARD_CLASS_NAMES,
        model_maps=_BREAST_MAPS,
    ),
    Cohort.COADREAD: CohortSpec(
        dir_name="04_COADREAD_COHORT",
        class_names=_STANDARD_CLASS_NAMES,
        model_maps=_COADREAD_MAPS,
    ),
    Cohort.LUNG: CohortSpec(
        dir_name="01_LUNG_COHORT",
        class_names=_STANDARD_CLASS_NAMES,
        model_maps=_LUNG_MAPS,
    ),
    Cohort.PROSTATE: CohortSpec(
        dir_name="02_PRAD_COHORT",
        class_names=_PROSTATE_CLASS_NAMES,
        model_maps=_PROSTATE_MAPS,
    ),
}


def get_remap_lut(cohort: Cohort, model: ModelKey) -> dict[int, int]:
    """Return the class remapping dict for a (cohort, model) pair.

    Raises :class:`KeyError` if the combination is not defined.
    """
    return COHORT_SPECS[cohort].model_maps[model]
