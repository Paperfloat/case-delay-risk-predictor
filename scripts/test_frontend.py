import os

from streamlit.testing.v1 import AppTest

APP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend", "app.py")
at = AppTest.from_file(os.path.abspath(APP), default_timeout=90).run()
assert not at.exception, at.exception
assert not at.error, [e.value for e in at.error]
print("loaded; first selectboxes:", [s.label for s in at.selectbox][:4])


def blob():
    return " ".join(e.value for e in list(at.markdown) + list(at.caption))


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
    text = blob()
    assert "RISK" in text, f"no result shown for {state}"
    assert "How long did similar cases take" in text, f"no similar-case history for {state}"
    assert "What pushed this result" in text
    assert "log-days" not in text, "old technical wording is still shown"
    assert "Female" not in text and "Male" not in text, "gender wording is still shown"
    first = next((m.value for m in at.markdown if "similar past cases" in m.value), "")
    print(f"{state}{' + act/section' if with_act else ''}: ok | {first[:100]}")
    return text


run_state("delhi")
text = run_state("delhi", with_act=True)
assert "delhi_acts" in text, "act-aware model was not used"
run_state("odisha")
run_state("bihar")
print("frontend test passed")
