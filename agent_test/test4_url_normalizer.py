import urllib.parse

def normalize_url(raw_url: str) -> str:
    if not raw_url or not raw_url.strip():
        raise ValueError("URL cannot be empty.")

    url_clean = raw_url.strip()
    parsed = urllib.parse.urlparse(url_clean)

    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https"):
        raise ValueError(f"Unsupported or invalid URL scheme: '{scheme}'")

    if not parsed.netloc:
        raise ValueError("URL must have a valid host netloc.")

    # Parse host and port
    host = parsed.hostname.lower() if parsed.hostname else ""
    port = parsed.port

    # Strip default ports (80 for http, 443 for https)
    netloc = host
    if port is not None:
        if not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
            netloc = f"{host}:{port}"

    # Normalize path (strip trailing slash unless path is just '/')
    path = parsed.path.rstrip("/") if parsed.path != "/" else "/"

    # Sort and deduplicate query parameters
    query = ""
    if parsed.query:
        query_dict = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
        # Deduplicate and sort items
        sorted_pairs = []
        for key in sorted(query_dict.keys()):
            for val in sorted(list(set(query_dict[key]))):
                sorted_pairs.append((key, val))
        query = urllib.parse.urlencode(sorted_pairs)

    reconstructed = urllib.parse.urlunparse((scheme, netloc, path, "", query, ""))
    return reconstructed
