"""Undo a global truststore injection before the HTTP stack is imported.

Some Python installs (corporate Windows images especially) call
`pip._vendor.truststore.inject_into_ssl()` at interpreter startup so the
Windows certificate store is honoured. That replaces `ssl.SSLContext`
globally, for every process.

If a standalone `truststore` is *also* installed - httpx pulls it in - it
captures `_original_super_SSLContext` at import time, which by then is pip's
injected class. Setting `verify_mode` then calls that class's setter, which
delegates via `super(SSLContext, SSLContext)`, which resolves back to itself:

    File "ssl.py", line 680, in verify_mode
      super(SSLContext, SSLContext).verify_mode.__set__(self, value)
      [Previous line repeated 494 more times]
    RecursionError: maximum recursion depth exceeded

Every HTTPS request through httpx dies this way, surfacing as
`anthropic.APIConnectionError: Connection error.` - which looks like a network
problem and is not one.

Import this module BEFORE anthropic/httpx so the standalone truststore
captures the real `ssl.SSLContext`. Undoing the injection does not disable
certificate verification: the standalone truststore still reads the Windows
store, and certifi remains the fallback.
"""

import ssl


def _is_truststore_context(cls) -> bool:
    return "truststore" in getattr(cls, "__module__", "")


def undo_pip_truststore_injection() -> bool:
    """Restore the stdlib ssl.SSLContext if pip's vendored copy replaced it.

    Returns True if an injection was found and undone. Safe to call always -
    a no-op when nothing was injected.
    """
    if not _is_truststore_context(ssl.SSLContext):
        return False
    for mod_name in ("pip._vendor.truststore", "truststore"):
        try:
            mod = __import__(mod_name, fromlist=["extract_from_ssl"])
            extract = getattr(mod, "extract_from_ssl", None)
            if extract is None:
                continue
            extract()
            if not _is_truststore_context(ssl.SSLContext):
                return True
        except Exception:
            continue
    return False


def use_local_ca_bundle():
    """Point every TLS stack at a repo-local CA bundle, if one exists.

    Networks that inspect TLS need their corporate root CA, which lives in the
    Windows certificate store rather than in certifi. Once
    scripts/export_windows_ca_bundle.ps1 has written corp-ca-bundle.pem, wire it
    up automatically so it does not have to be re-exported into environment
    variables in every new shell - a step that is easy to forget and surfaces as
    a confusing CERTIFICATE_VERIFY_FAILED.

    Explicit environment variables always win; this only fills in the blanks.
    Returns the bundle path if one was applied, else None.
    """
    import os
    import pathlib as _p

    bundle = _p.Path(__file__).resolve().parent.parent / "corp-ca-bundle.pem"
    if not bundle.exists():
        return None
    for var in ("SSL_CERT_FILE", "AWS_CA_BUNDLE", "REQUESTS_CA_BUNDLE"):
        os.environ.setdefault(var, str(bundle))
    return str(bundle)


CA_BUNDLE = use_local_ca_bundle()

# Escape hatch: set AUTOFAB_KEEP_TRUSTSTORE=1 to leave the injection alone
# (for example on a machine where the Windows store is the only source of a
# corporate root CA and httpx happens to work anyway).
import os as _os

if _os.getenv("AUTOFAB_KEEP_TRUSTSTORE", "").strip() in ("1", "true", "yes"):
    UNDONE = False
else:
    UNDONE = undo_pip_truststore_injection()
