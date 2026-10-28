import app as module


def test_hello_escapes_the_name():
    r = module.app.test_client().get("/hello/<b>")
    assert r.status_code == 200 and b"&lt;b&gt;" in r.data


def test_dates_are_json():
    assert module.app.test_client().get("/day").get_json() == {"day": "2024-05-17"}
