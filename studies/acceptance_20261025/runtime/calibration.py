"""Evaluation-only count calibration with analytic joint gradients.

The primary fixed-rho scale fit is unchanged. The secondary joint fit uses
the same objective and bounds as likelihood.calibrate, with boundary checks
and an analytic Jacobian to avoid finite-difference noise near rho=1e-4.
No training, selection, count likelihood or partition code imports this file.
"""
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import digamma, expit
from likelihood import nll_numpy

LOW=np.asarray([0.,np.log(1e-4)])
HIGH=np.asarray([3.,np.log(.8)])


def joint_value_gradient(v,k,n,tau):
    scale,log_rho=np.asarray(v,np.float64)
    rho=np.exp(log_rho);c=1/rho-1
    raw=scale*tau;p=expit(np.clip(raw,-20,20));a=p*c;b=(1-p)*c
    da=digamma(a)-digamma(k+a)+digamma(n+c)-digamma(c)
    db=digamma(b)-digamma(n-k+b)+digamma(n+c)-digamma(c)
    dp=tau*p*(1-p)*(np.abs(raw)<20)
    jac=np.asarray([np.mean(c*(da-db)*dp),np.mean(-(p*da+(1-p)*db)/rho)])
    value=float(nll_numpy(k,n,raw,rho).mean())
    if not np.isfinite(value) or not np.isfinite(jac).all():raise RuntimeError("Nonfinite calibration objective/gradient")
    return value,jac


def projected_gradient(v,jac):
    v,jac=np.asarray(v),np.asarray(jac)
    free=jac.copy()
    free[(v<=LOW+1e-7)&(jac>=0)]=0
    free[(v>=HIGH-1e-7)&(jac<=0)]=0
    return float(np.max(np.abs(free)))


def fit_joint(k,n,tau,rho,scale):
    candidates=[];statuses=[]
    def add(v,label):
        v=np.clip(np.asarray(v,np.float64),LOW,HIGH)
        value,jac=joint_value_gradient(v,k,n,tau)
        candidates.append(dict(v=v,value=value,label=label,gradient=projected_gradient(v,jac)))
    # Every edge and corner is considered, including the binomial-limit floor.
    # These are optimization candidates on calibration data, not test selection.
    for fixed in (0,1):
        free=1-fixed
        for bound in (LOW[fixed],HIGH[fixed]):
            def vector(x):
                v=np.zeros(2);v[fixed]=bound;v[free]=x;return v
            opt=minimize_scalar(lambda x:joint_value_gradient(vector(x),k,n,tau)[0],
                bounds=(LOW[free],HIGH[free]),method="bounded",options={"xatol":1e-8})
            if not opt.success:raise RuntimeError(f"Joint boundary fit failed: {opt.message}")
            for x in (opt.x,LOW[free],HIGH[free]):add(vector(x),f"edge_{fixed}_{bound:.8g}")
    best_edge=min(candidates,key=lambda c:c["value"])
    starts=[np.asarray([scale,np.log(rho)]),best_edge["v"]]
    for index,start in enumerate(starts):
        opt=minimize(lambda v:joint_value_gradient(v,k,n,tau),start,method="L-BFGS-B",jac=True,
            bounds=list(zip(LOW,HIGH)),options={"ftol":1e-12,"gtol":1e-7,"maxiter":300,"maxls":50})
        add(opt.x,f"analytic_joint_{index}")
        statuses.append(dict(success=bool(opt.success),message=str(opt.message),iterations=int(opt.nit)))
    acceptable=[c for c in candidates if c["gradient"]<=2e-5]
    if not acceptable:
        raise RuntimeError("Joint calibration did not meet projected-gradient tolerance; no result accepted")
    best=min(acceptable,key=lambda c:c["value"])
    # A candidate with materially better objective but bad stationarity means
    # the search needs review; do not silently return a worse boundary solution.
    if best["value"]>min(c["value"] for c in candidates)+1e-7:
        raise RuntimeError("Nonstationary calibration candidate has a better objective; review required")
    return best["v"],dict(method="analytic_gradient_with_four_edges",candidate=best["label"],
        projected_gradient=best["gradient"],objective=best["value"],optimizer_attempts=statuses,
        rho_at_boundary=bool(abs(best["v"][1]-LOW[1])<1e-6 or abs(best["v"][1]-HIGH[1])<1e-6),
        acceptance="Feasible candidate with projected gradient<=2e-5; no claim of a global optimum")


def calibrate_single(k,n,tau,rho):
    k,n,tau=[np.asarray(v,np.float64) for v in (k,n,tau)]
    if len(k)<30 or k.ndim!=1 or k.shape!=n.shape or k.shape!=tau.shape or not np.isfinite(tau).all():
        raise ValueError("Invalid or insufficient calibration support")
    if not (np.isfinite(k).all() and np.isfinite(n).all() and np.all(n>0) and np.all(k>=0) and np.all(k<=n) and 1e-4<=rho<=.8):
        raise ValueError("Invalid calibration count or shared rho")
    objective=lambda a:float(nll_numpy(k,n,a*tau,rho).mean())
    opt=minimize_scalar(objective,bounds=(0,3),method="bounded")
    candidates=[(float(opt.x),float(opt.fun)),(0.,objective(0.)),(3.,objective(3.))]
    if not opt.success:raise RuntimeError(f"Primary scale fit failed: {opt.message}")
    scale=min(candidates,key=lambda item:item[1])[0]
    joint,diagnostics=fit_joint(k,n,tau,rho,scale)
    return dict(scale=scale,shared_rho=float(rho),joint_scale=float(joint[0]),joint_rho=float(np.exp(joint[1])),
        rows=len(k),scale_at_boundary=scale<.001 or scale>2.999,joint_diagnostics=diagnostics)


def calibrate(frame,rows,tau,rho):
    x=frame.iloc[rows];assays=x.assay_id.to_numpy();results=[]
    for assay in range(5):
        mask=assays==assay
        results.append(calibrate_single(x.n_alt.to_numpy()[mask],x.total.to_numpy()[mask],tau[mask],rho[assay]))
    return dict(scales=[r["scale"] for r in results],joint_scales=[r["joint_scale"] for r in results],
        joint_rho=[r["joint_rho"] for r in results],
        diagnostics=[dict(assay=a,rows=r["rows"],scale_at_boundary=r["scale_at_boundary"],joint=r["joint_diagnostics"]) for a,r in enumerate(results)])
