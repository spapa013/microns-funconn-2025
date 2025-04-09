from enum import Enum
from dataclasses import dataclass
from typing import Optional


@dataclass
class Quantity:
    """
    Abstract dataclass for a quantity. The quantity is defined by a numerator divided by an optional denominator.
    If the denominator is not provided, it is assumed to be 1.
    """

    numerator: str
    denominator: Optional[str] = None
    query: Optional[str] = None
    family: str = "poisson"
    name_latex: Optional[str] = None

    def __post_init__(self):
        if self.name_latex is None:
            self.name_latex = (
                self.numerator.replace("_", " ")
                + " / "
                + (self.denominator.replace("_", " ") if self.denominator else "")
            )
        if self.query is None:
            self.query = "index == index"


class Quantities(Enum):
    LD_DENSITY = Quantity(
        numerator="dend_len",
        family="tweedie",
        name_latex=r"$L_d$ / neuron pair",
    )
    LD_DENSITY_CONTROL = Quantity(
        numerator="dend_len",
        query="n_synapses == 0",
        family="tweedie",
        name_latex=r"$L_d$ / neuron pair (synapses excluded)",
    )
    N_SYNAPSES = Quantity(
        numerator="n_synapses",
        family="poisson",
        name_latex=r"$N_{syn}$",
    )
    SYNAPSE_SIZE = Quantity(
        numerator="synapse_size",
        family="tweedie",
        name_latex=r"Total cleft volume",
    )
    SYNAPSE_DENSITY = Quantity(
        numerator="n_synapses",
        denominator="dend_len",
        query="dend_len > 0",
        family="poisson",
        name_latex=r"$N_{syn} / mm\ L_d$",
    )
    SYNAPSE_SIZE_DENSITY = Quantity(
        numerator="synapse_size",
        denominator="dend_len",
        query="dend_len > 0",
        family="tweedie",
        name_latex=r"Total cleft volume / mm $L_d$",
    )
    MEAN_SYNAPSE_SIZE = Quantity(
        numerator="synapse_size",
        denominator="n_synapses",
        query="n_synapses > 0",
        family="tweedie",
        name_latex=r"mean cleft volume",
    )
    N_SYNAPSES_POSITIVE = Quantity(
        numerator="n_synapses",
        query="n_synapses > 0",
        family="poisson",
        name_latex=r"$N_{syn}$",
    )

    @classmethod
    def get_name(cls, value: Quantity) -> str:
        for q in cls:
            if q.value == value:
                return q.name
        raise ValueError(f"No quantity found with value: {value}")
