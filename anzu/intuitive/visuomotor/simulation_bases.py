class SimFailure(Exception):
    """Indicates simulation failed."""

    pass


def translate_drake_simulation_failures(func):
    """
    Catches any specific exceptions that may arise from Drake simulations
    and converts them to an instance of the ``SimFailure`` class.
    """
    convergence_substr = (
        "MultibodyPlant's discrete update solver failed to converge"
    )
    sap_substr = "condition '!std::isnan(ExtractDoubleOrThrow(contact_configuration.vn))' failed."  # noqa

    hessian_nan_substr = "The Hessian of the momentum cost along the search direction is NaN."  # noqa
    line_search_nan_substr = "The initial guess for line search is NaN."

    def decorated(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (RuntimeError, BaseException) as e:
            err_str = str(e)
            if convergence_substr in err_str:
                raise SimFailure(convergence_substr)
            elif sap_substr in err_str:
                raise SimFailure(err_str)
            elif hessian_nan_substr in err_str:
                raise SimFailure(err_str)
            elif line_search_nan_substr in err_str:
                raise SimFailure(err_str)
            else:
                raise

    return decorated
