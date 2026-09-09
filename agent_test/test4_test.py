from test4_url_normalizer import normalize_url
import pytest

def test_1_lowercase_and_trailing_slash():
    assert normalize_url("HTTP://Example.COM/path/") == "http://example.com/path"

def test_2_default_ports_stripped():
    assert normalize_url("http://example.com:80/api") == "http://example.com/api"
    assert normalize_url("https://example.com:443/api") == "https://example.com/api"

def test_3_custom_port_preserved():
    assert normalize_url("http://example.com:8080/api") == "http://example.com:8080/api"

def test_4_query_param_alphabetical_sorting():
    assert normalize_url("http://example.com?beta=2&alpha=1") == "http://example.com?alpha=1&beta=2"

def test_5_query_param_deduplication():
    assert normalize_url("http://example.com?tag=b&tag=a&tag=b") == "http://example.com?tag=a&tag=b"

def test_6_ipv4_host_preserved():
    assert normalize_url("http://192.168.1.1:5000/metrics") == "http://192.168.1.1:5000/metrics"

def test_7_empty_string_rejected():
    with pytest.raises(ValueError):
        normalize_url("")

def test_8_invalid_scheme_rejected():
    with pytest.raises(ValueError):
        normalize_url("ftp://unsupported.com")
