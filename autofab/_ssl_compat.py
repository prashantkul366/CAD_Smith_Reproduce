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


# Escape hatch: set AUTOFAB_KEEP_TRUSTSTORE=1 to leave the injection alone
# (for example on a machine where the Windows store is the only source of a
# corporate root CA and httpx happens to work anyway).
import os as _os

if _os.getenv("AUTOFAB_KEEP_TRUSTSTORE", "").strip() in ("1", "true", "yes"):
    UNDONE = False
else:
    UNDONE = undo_pip_truststore_injection()
