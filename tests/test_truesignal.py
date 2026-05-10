"""
Unit tests for TrueSignal core components.

Run with:
    pytest tests/test_truesignal.py -v

Coverage:
  - MAS algorithm (identical/opposite/boundary/symmetric)
  - Incongruence labeling rule (all conflict combinations)
  - MLP model (output shape, range, wrong input dimension)
  - Feature vector (parse, 24d shape, NaN check)
"""

import pytest, torch, numpy as np, sys
sys.path.append(".")
from models.mlp_fusion import IncongruenceMLP, parse_vector
from scipy.spatial.distance import cosine


# ── Helpers ───────────────────────────────────────────────────────
def mas(a, b):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    if np.linalg.norm(a)==0 or np.linalg.norm(b)==0:
        return 0.0
    return float(1.0 - cosine(a, b))

def apply_rule(fl,fc,vl,vc,tl,tc,th=0.6):
    pairs=[("face","voice",fl,fc,vl,vc),
           ("face","text", fl,fc,tl,tc),
           ("voice","text",vl,vc,tl,tc)]
    c=[]
    for m1,m2,l1,c1,l2,c2 in pairs:
        if l1!=l2 and c1>=th and c2>=th: c.append(f"{m1}-{m2}")
    return len(c)>0, c


# ── MAS Tests ─────────────────────────────────────────────────────
class TestMAS:
    def test_identical_vectors_score_one(self):
        v = [0,0,0,1,0,0,0]
        assert abs(mas(v,v) - 1.0) < 1e-6

    def test_opposite_vectors_score_zero(self):
        a = [1,0,0,0,0,0,0]
        b = [0,0,0,1,0,0,0]
        assert abs(mas(a,b) - 0.0) < 1e-6

    def test_range_always_0_to_1(self):
        rng = np.random.default_rng(42)
        for _ in range(200):
            a = rng.dirichlet(np.ones(7))
            b = rng.dirichlet(np.ones(7))
            assert 0.0 <= mas(a,b) <= 1.0+1e-6

    def test_symmetric(self):
        a = [0.5,0.2,0,0.3,0,0,0]
        b = [0,0,0,0.7,0.3,0,0]
        assert abs(mas(a,b) - mas(b,a)) < 1e-6

    def test_zero_vector_returns_zero(self):
        assert mas([0]*7, [1,0,0,0,0,0,0]) == 0.0


# ── Incongruence Rule Tests ────────────────────────────────────────
class TestIncongruenceRule:
    def test_all_agree_is_congruent(self):
        inc, pairs = apply_rule("joy",0.9,"joy",0.85,"joy",0.92)
        assert not inc and pairs == []

    def test_face_voice_conflict(self):
        inc, pairs = apply_rule("anger",0.8,"joy",0.75,"anger",0.85)
        assert inc and "face-voice" in pairs

    def test_low_confidence_not_flagged(self):
        inc, _ = apply_rule("anger",0.4,"joy",0.3,"anger",0.9)
        assert not inc

    def test_three_way_conflict(self):
        inc, pairs = apply_rule("anger",0.9,"joy",0.8,"neutral",0.85)
        assert inc and len(pairs)==3

    def test_exactly_at_threshold_included(self):
        inc, pairs = apply_rule("anger",0.6,"joy",0.6,"anger",0.9)
        assert inc and "face-voice" in pairs

    def test_just_below_threshold_excluded(self):
        inc, _ = apply_rule("anger",0.59,"joy",0.59,"anger",0.9)
        assert not inc


# ── MLP Model Tests ───────────────────────────────────────────────
class TestMLP:
    @pytest.fixture
    def model(self):
        return IncongruenceMLP(input_dim=24).eval()

    def test_output_shape_batch(self, model):
        assert model(torch.randn(32,24)).shape == (32,)

    def test_sigmoid_output_in_range(self, model):
        with torch.no_grad():
            p = torch.sigmoid(model(torch.randn(100,24)))
        assert p.min()>0.0 and p.max()<1.0

    def test_single_sample(self, model):
        with torch.no_grad():
            out = torch.sigmoid(model(torch.randn(1,24)))
        assert out.shape==(1,)

    def test_wrong_input_raises(self, model):
        with pytest.raises(Exception):
            model(torch.randn(8,21))   # missing MAS scores


# ── Feature Vector Tests ──────────────────────────────────────────
class TestFeatureVector:
    def test_parse_list(self):
        v = parse_vector([0.1,0.2,0.3,0.1,0.1,0.1,0.1])
        assert len(v)==7

    def test_parse_string(self):
        v = parse_vector("[0.1,0.2,0.3,0.1,0.1,0.1,0.1]")
        assert len(v)==7

    def test_corrupt_returns_uniform(self):
        v = parse_vector("not_valid")
        assert len(v)==7 and abs(v.sum()-1.0)<0.01

    def test_full_24d_shape(self):
        fv  = np.random.dirichlet(np.ones(7)).astype(np.float32)
        vv  = np.random.dirichlet(np.ones(7)).astype(np.float32)
        tv  = np.random.dirichlet(np.ones(7)).astype(np.float32)
        mas = np.array([0.5,0.3,0.8], dtype=np.float32)
        feat = np.concatenate([fv,vv,tv,mas])
        assert feat.shape==(24,)
        assert not np.isnan(feat).any()
        assert not np.isinf(feat).any()
