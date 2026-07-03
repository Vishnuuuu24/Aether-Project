"""De-identification egress filter is default-deny (docs/06 §6, §9)."""

from __future__ import annotations

import pytest

from ai.llm.deid import EgressBlocked, assert_clean_for_egress, scan_for_identifiers


@pytest.mark.parametrize(
    "text",
    [
        "contact jane.doe@example.com",
        "call 415-555-2671 tomorrow",
        "SSN 123-45-6789 on file",
        "DOB 1984-02-11",
        "seen on 02/11/1984",
        "seen on Feb 11, 2024",
        "record 004839201 pulled",
        "lives at 123 Main Street",
        "see https://hospital.example/patient",
    ],
)
def test_direct_identifiers_are_detected(text: str) -> None:
    report = scan_for_identifiers(text)
    assert not report.clean
    with pytest.raises(EgressBlocked):
        assert_clean_for_egress(text)


def test_clean_clinical_text_passes() -> None:
    text = "Resting heart rate rose about 12 bpm above the personal baseline overnight."
    assert scan_for_identifiers(text).clean
    assert_clean_for_egress(text)  # does not raise


def test_blocked_error_lists_kinds() -> None:
    with pytest.raises(EgressBlocked) as exc:
        assert_clean_for_egress("email a@b.com and phone 415-555-2671")
    assert "email" in exc.value.kinds
    assert "phone" in exc.value.kinds
