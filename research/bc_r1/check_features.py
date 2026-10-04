"""Real RGB forward equivalence before allowing C to reuse LoRAT features."""
import cv2
from .common import *
from .tracker_c import ResearchCore

def main():
    setup();folder=RUN/'C_feature_check';folder.mkdir(exist_ok=True)
    case=read(RUN/'PROTOCOL.json')['cases'][0];cap=cv2.VideoCapture(case['video'])
    ok,first=cap.read();assert ok;ok,second=cap.read();assert ok;cap.release()
    z=cv2.cvtColor(first,cv2.COLOR_BGR2RGB);x=cv2.cvtColor(second,cv2.COLOR_BGR2RGB)
    core=ResearchCore(folder/'MODEL.json',features=False);core.initialize(z,case['initial_bbox'])
    original=core.propose(x)
    from types import MethodType
    from .tracker_c import feature_track
    adapter=core.native.adapter;adapter.track=MethodType(feature_track,adapter);adapter._descriptors={}
    exposed=core.propose(x)
    assert original==exposed,(original,exposed)
    f=core.descriptor(exposed);assert f.shape==(4,1024) and np.isfinite(f).all()
    save(folder/'RESULT.json',dict(status='PASS',native_and_feature_forward_exactly_equal=True,
        descriptor_shape=list(f.shape),source_frame_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
        source_hashes={str(Path(__file__).with_name(n)):sha(Path(__file__).with_name(n)) for n in ['model_c.py','tracker_c.py','check_features.py']}))
    print('C_FEATURE_CHECK_PASS',flush=True)
if __name__=='__main__':main()
