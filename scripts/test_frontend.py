import os

from streamlit.testing.v1 import AppTest

APP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend", "app.py")
at = AppTest.from_file(os.path.abspath(APP), default_timeout=90).run()
assert not at.exception, at.exception
assert not at.error, [e.value for e in at.error]
print("loaded; first selectboxes:", [s.label for s in at.selectbox][:4])


def run_state(state, with_act=False):
    at.selectbox(key="state").set_value(state).run()
    assert not at.exception, at.exception
    if with_act:
        for f in ["primary_act", "primary_section"]:
            box = at.selectbox(key=f"{state}_{f}")
            box.set_value(box.options[1])
        at.run()
    at.button[0].click().run()
    assert not at.exception, at.exception
    assert not at.error, [e.value for e in at.error]
    html = " ".join(m.value for m in at.markdown)
    assert "RISK" in html, f"no result shown for {state}"
    caps = " ".join(c.value for c in at.caption)
    print(f"{state}{' + act/section' if with_act else ''}: result shown | {caps[:90]}")
    return caps


run_state("delhi")
caps = run_state("delhi", with_act=True)
assert "delhi_acts" in caps, "act-aware model was not used"
run_state("odisha")
run_state("bihar")
print("frontend test passed")
