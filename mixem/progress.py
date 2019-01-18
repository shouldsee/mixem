# coding=utf-8


def simple_progress(iteration, weights, distributions, log_likelihood, log_proba):
    """A simple default progress callback to use with mixem.em"""

    print("iteration {iteration:4d} (log-likelihood={log_likelihood:.5e}): p(x|Φ) = {formatted_distributions}".format(
        iteration=iteration,
        log_likelihood=log_likelihood,
        formatted_distributions=" + ".join("{w:.3g}*{d}".format(w=w, d=d) for w, d in zip(weights, distributions))
    ))
    
def very_simple_progress(iteration, weights, distributions, log_likelihood, log_proba):
    """A simple default progress callback to use with mixem.em"""

    print("iteration {iteration:4d} (log-likelihood={log_likelihood:.5e}): p(x|Φ) = {formatted_distributions}".format(
        iteration=iteration,
        log_likelihood=log_likelihood,
        formatted_distributions=" + ".join("{w:.3g}*{d}".format(w=w, d=d) for w, d in zip(weights, distributions))[:10] 
        
    ))