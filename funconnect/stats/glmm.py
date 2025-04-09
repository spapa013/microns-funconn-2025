import gc
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd
import rpy2.robjects as ro
import statsmodels.api as sm
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

from funconnect.stats.quantity import Quantity, Quantities
from funconnect.stats.variable import Variable, Variables
from funconnect.utility.R import model_diagnostics
from funconnect.utility.connectomics import summarize_edge_data

logger = logging.getLogger("funconnect")

pandas2ri.activate()
base = importr("base")
glmmTMB = importr("glmmTMB")
tidyverse = importr("tidyverse")
broom = importr("broom.mixed")
stats = importr("stats")
emmeans = importr("emmeans")
performance = importr("performance")


@dataclass
class GlmmRslt:
    """
    Dataclass for storing
    GLMM results.
    """

    model_performance: pd.Series
    model_coef: pd.DataFrame
    emtrends: pd.DataFrame
    pairwise: pd.DataFrame
    meta_info: pd.DataFrame

    def save(self, rslt_dir: Path) -> None:
        """
        Save the results to a directory.
        """
        # Create directory if it does not exist
        os.makedirs(rslt_dir, exist_ok=True)
        # Save model performance
        self.model_performance.to_csv(
            rslt_dir / "glmm_model_performance.csv",
            index=False,
        )
        # Save model coefficients
        self.model_coef.to_csv(
            rslt_dir / "glmm_model_coef.csv",
            index=False,
        )
        # Save emtrends
        self.emtrends.to_csv(rslt_dir / "glmm_emtrends.csv", index=False)
        # Save pairwise comparisons
        self.pairwise.to_csv(rslt_dir / "glmm_pairwise.csv", index=False)
        # Save meta info
        self.meta_info.to_csv(rslt_dir / "glmm_meta_info.csv", index=False)
        logger.info(f"Saved GLMM results to {rslt_dir}")


@dataclass
class Glmm:
    quantity: Quantity
    data: pd.DataFrame
    variable: Variable
    proj: str = "proj_hva"
    link: str = "log"

    def __post_init__(self):
        """
        Verify the quantity is compatible with the data.
        """
        necessary_cols = [
            self.quantity.numerator,
            "pre_nucleus_id",
            self.proj,
            self.variable.col_name,
        ]
        if self.quantity.denominator:
            necessary_cols.append(self.quantity.denominator)
        missing_cols = set(necessary_cols) - set(self.data.columns)
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        self.cols = necessary_cols
        self.quantity_name = Quantities.get_name(self.quantity)
        self.variable_name = Variables.get_name(self.variable)

    @property
    def formula(self) -> str:
        """
        Return the formula string for the GLM model.

        Args:
            variable_name: Variable name to include in formula
            proj: Type of projection ('hva' or 'hva_layer')

        Returns:
            str: Complete formula string for model fitting
        """

        # Build formula components
        response = self.quantity.numerator
        predictor = f"{self.variable.col_name}*{self.proj}"
        random_effect = f"+ (1|pre_nucleus_id:{self.proj})"

        # Add offset term if denominator exists
        offset = (
            f"+ offset({self.link}({self.quantity.denominator}))"
            if self.quantity.denominator
            else ""
        )

        # Combine all parts with proper spacing
        parts = [response, "~", predictor]
        if offset:
            parts.append(offset)
        parts.append(random_effect)

        return " ".join(parts)

    def _fit_glmm(
        self, rslt_dir: str, n_parallel: int = 4
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Fit a glmmTMB model to the data.
        """
        data = (
            self.data.loc[:, self.cols].copy()
        )  # select necessary columns to reduce data transfer between Python and R
        r_model = glmmTMB.glmmTMB(
            formula=ro.Formula(self.formula),
            data=data,
            family=getattr(ro.r, self.quantity.family)(link=self.link),
            control=glmmTMB.glmmTMBControl(
                parallel=n_parallel, optCtrl=ro.r("list(iter.max=1e3,eval.max=1e3)")
            ),
        )
        model_check = model_diagnostics(r_model, rslt_dir / "model_diagnostics.png")
        if not model_check["converged"]:
            logger.warning(
                f"Model did not converge for {self.quantity_name} and {self.variable_name}."
            )

        # Get performance metrics and immediately convert to Python
        r_performance = performance.performance(r_model)
        model_performance = ro.conversion.rpy2py(r_performance).to_dict(
            orient="records"
        )[0]
        del r_performance  # Explicitly delete R reference

        r_coef = ro.r["tidy"](
            r_model,
            conf_int=True,
            conf_level=0.95,
            effects=["fixed", "ran_pars", "ran_vals"],
        )

        model_coef = ro.conversion.rpy2py(r_coef).assign(
            variable_name=self.variable_name,
            quantity_name=self.quantity_name,
        )
        del r_coef  # Explicitly delete R reference

        model_performance = pd.Series({
            **model_check,
            **model_performance,
            "variable_name": self.variable_name,
            "quantity_name": self.quantity_name,
        })
        # Run garbage collection to free up memory
        ro.r("gc()")
        return r_model, model_performance, model_coef

    def _fit_emtrends(self, model) -> pd.DataFrame:
        """
        Fit emtrends to the glmmTMB model.
        """
        r_emtrends = emmeans.emtrends(
            model,
            ro.Formula(f" ~ {self.proj}"),
            var=self.variable.col_name,
            infer=True,
            adjust="none",
        )
        r_tidy = ro.r["tidy"](r_emtrends, **{"conf.int": True})
        trend = ro.conversion.rpy2py(r_tidy)
        del r_tidy  # Explicitly delete R reference
        trend = (
            (
                trend.rename(
                    columns={
                        self.variable_name + ".trend": "coef",
                        "conf.low": "coef_ci_low",
                        "conf.high": "coef_ci_high",
                        "p.value": "p",
                        "statistic": "z",
                    }
                )
            )
            .set_index(self.proj)
            .assign(
                variable_name=self.variable_name,
                quantity_name=self.quantity_name,
                p_adj=lambda df: sm.stats.multipletests(df["p"], method="fdr_bh")[1],
            )
            .reset_index()
        )
        # Run garbage collection to free up memory
        ro.r("gc()")
        return r_emtrends, trend

    def _fit_pairwise(self, r_emtrends) -> pd.DataFrame:
        """
        Fit pairwise comparisons to the emmeans results.
        """
        r_pair_trend = ro.r.tidy(ro.r.pairs(r_emtrends, adjust="none"))
        pair_trend = ro.conversion.rpy2py(r_pair_trend)
        del r_pair_trend  # Explicitly delete R reference
        pair_trend = (
            pair_trend.drop(labels=["term", "null.value", "std.error", "df"], axis=1)
            .rename(
                columns={
                    "p.value": "p",
                    "conf.low": "coef_ci_low",
                    "conf.high": "coef_ci_high",
                    "statistic": "z",
                    "estimate": "coef_diff",
                }
            )
            .assign(
                variable_name=self.variable_name,
                quantity_name=self.quantity_name,
                p_adj=lambda df: sm.stats.multipletests(df["p"], method="fdr_bh")[1],
            )
        )
        # Run garbage collection to free up memory
        ro.r("gc()")
        return pair_trend

    def _meta_info(self):
        """
        Return meta info for the model.
        """
        meta_info = []
        # get meta information
        for proj, proj_data in self.data.groupby(self.proj):
            _meta_info = pd.Series({
                "variable_name": self.variable_name,
                "quantity_name": self.quantity_name,
                self.proj: proj,
                **summarize_edge_data(proj_data),
            })
            meta_info.append(_meta_info)
        return pd.DataFrame(meta_info)

    def fit(self, rslt_dir: str, n_parallel: int = 4, save: bool = True) -> GlmmRslt:
        """
        Fit the GLMM model and return the results.
        """
        # create rslt_dir if it does not exist
        os.makedirs(rslt_dir, exist_ok=True)
        try:
            r_model, model_performance, model_coef = self._fit_glmm(
                rslt_dir, n_parallel
            )
            r_emtrends, emtrends_fit = self._fit_emtrends(r_model)
            pairwise_fit = self._fit_pairwise(r_emtrends)
            meta_info = self._meta_info()
            del r_model, r_emtrends  # Explicitly delete R references
            glmm_rslt = GlmmRslt(
                model_performance=model_performance,
                model_coef=model_coef,
                emtrends=emtrends_fit,
                pairwise=pairwise_fit,
                meta_info=meta_info,
            )
            if save:
                glmm_rslt.save(rslt_dir)
            return glmm_rslt
        finally:
            # Clean up all R objects
            ro.r("rm(list=ls())")
            ro.r("gc()")
            # Force Python garbage collection
            gc.collect()
