from enum import Enum
from dataclasses import dataclass
from typing import Optional

CVT_CRITERIA = (
    "pre_cc_max_cvt>0.4 and post_cc_max_cvt>0.4 "
    "and pre_cc_abs_cvt>0.2 and post_cc_abs_cvt>0.2"
)
ORI_CRITERIA_CVT_FULL = "pre_gosi_cvt_monet_full>0.25 and post_gosi_cvt_monet_full>0.25"
ORI_CRITERIA_IV = "pre_osi_iv>0.4 and post_osi_iv>0.4"


@dataclass
class Variable:
    name_latex: str
    col_name: str
    inverse: Optional[bool] = False  # whether to invert the variable axis when plotting
    criteria: Optional[str] = "index == index"  # criteria for filtering the data


class Variables(Enum):
    IN_SILICO_SIG_CORR_CVT = Variable(
        name_latex=r"Digital twin signal correlation",
        col_name="in_silico_sig_corr_cvt",
        inverse=False,
        criteria=CVT_CRITERIA,
    )
    IN_VIVO_SIG_CORR = Variable(
        name_latex=r"In vivo signal correlation",
        col_name="in_vivo_sig_corr",
        inverse=False,
        criteria="pre_cc_max_cvt>0.4 and post_cc_max_cvt>0.4",
    )
    READOUT_SIMILARITY_CVT = Variable(
        name_latex=r"Feature similarity",
        col_name="readout_similarity_cvt",
        inverse=False,
        criteria=CVT_CRITERIA,
    )
    READOUT_LOCATION_DISTANCE_CVT = Variable(
        name_latex=r"RF distance",
        col_name="readout_location_distance_cvt",
        inverse=True,
        criteria=CVT_CRITERIA,
    )
    READOUT_LOCATION_DISTANCE_STA = Variable(
        name_latex=r"RF distance (in silico STA)",
        col_name="readout_location_distance_sta",
        inverse=True,
        criteria=CVT_CRITERIA,
    )
    DELTA_ORI_CVT_MONET_FULL = Variable(
        name_latex=r"Digital twin (fine tuned) $\Delta$Ori",
        col_name="delta_ori_cvt_monet_full",
        inverse=True,
        criteria=ORI_CRITERIA_CVT_FULL,
    )
    DELTA_ORI_IV = Variable(
        name_latex=r"In vivo $\Delta$Ori",
        col_name="delta_ori_iv",
        inverse=True,
        criteria=ORI_CRITERIA_IV,
    )

    @classmethod
    def get_name(cls, variable: Variable) -> str:
        for item in cls:
            if item.value == variable:
                return item.name
        raise ValueError(f"Variable {variable} not found in Variables")
