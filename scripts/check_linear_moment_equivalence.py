"""Same-design Ridge is linear in labels: splitting a quadratic target adds no signal."""
from pathlib import Path
import json,sys
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.born_scattering import sha256

def main():
    root=ROOT/'data/m2_response_pilot_v1';mom=ROOT/'data/m2_residual_moments_v1'
    split=json.loads((root/'split.json').read_text());tr=split['train'];te=split['assessment']
    scale=np.array(json.loads((root/'manifest.json').read_text())['query_scale'])
    base=np.load(root/'feature_queries.npy').astype(float)/scale
    c=np.load(mom/'cross.npy').astype(float)/scale;u=np.load(mom/'energy.npy').astype(float)/scale
    x=StandardScaler().fit(base[tr]).transform(base)
    cfit=Ridge(alpha=1).fit(x[tr],c[tr]).predict(x[te]);ufit=Ridge(alpha=1).fit(x[tr],u[tr]).predict(x[te])
    direct=Ridge(alpha=1).fit(x[tr],c[tr]+u[tr]).predict(x[te])
    error=float(np.max(np.abs(cfit+ufit-direct)));assert error<1e-10
    q=np.load(root/'queries.npy').astype(float)/scale
    native=Ridge(alpha=1).fit(x[tr],q[tr]-base[tr]).predict(x[te])
    output=dict(passed=True,same_alpha=1,identical_design_linear_target_equivalence_error=error,
        difference_from_native_query_labels_due_to_float32_decomposition=float(np.max(np.abs(direct-native))),
        fraction_signed_cross_negative=float(np.mean(c[tr]<0)),
        interpretation='The sum of independently Ridge-fitted c and u equals the Ridge fit of c+u for identical design/penalty/intercept. This does not cover nonlinear constrained heads.',
        script_sha256=sha256(Path(__file__)))
    out=ROOT/'audit/m2_residual_20261101';out.mkdir(exist_ok=True);(out/'linear_equivalence.json').write_text(json.dumps(output,indent=2));print(output)

if __name__=='__main__':main()
