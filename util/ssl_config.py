import logging

_injected = False


def configure_system_truststore() -> None:
    global _injected

    if _injected:
        return

    try:
        import truststore
    except ImportError:
        raise RuntimeError("Dependencies missing about truststore")

    try:
        truststore.inject_into_ssl()
        _injected = True
    except Exception:
        logging.getLogger(__name__).exception("Failed to configure system truststore")
