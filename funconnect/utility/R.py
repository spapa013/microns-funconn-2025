import rpy2.robjects as ro
from rpy2.robjects.packages import importr
import gc
import numpy as np

performance = importr("performance")
DHARMa = importr("DHARMa")
grDevices = importr("grDevices")


def model_diagnostics(model, filename):
    filename = str(filename)
    # model checks
    converged = performance.check_convergence(model)
    singular = performance.check_singularity(model)
    simulated_residuals = DHARMa.simulateResiduals(model)
    p_uniformity = float(performance.check_residuals(simulated_residuals))
    p_overdispersion = float(performance.check_overdispersion(model).rx2("p_value"))
    grDevices.png(
        file=filename,
    )
    ro.r.plot(simulated_residuals)
    grDevices.dev_off()
    return {
        "p_uniformity": p_uniformity,
        "p_overdispersion": p_overdispersion,
        "converged": bool(np.array(converged)),
        "singular": bool(np.array(singular)),
    }


def r2py_clear():
    gc.collect()
    ro.r("rm(list=ls())")
    ro.r("gc()")
    gc.collect()


def export_to_r_global(df):
    for col in df.columns:
        if df[col].dtype.kind in "biufc":
            ro.globalenv[col] = ro.FloatVector(df[col].to_numpy())
        else:
            ro.globalenv[col] = ro.StrVector(df[col].to_numpy())