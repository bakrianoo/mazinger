"""Guards for the YouTube player-client workaround in ``mazinger.download``.

YouTube periodically breaks the InnerTube clients yt-dlp reaches for by
default; extraction then dies with "The page needs to be reloaded" before a
single format is listed.  ``download`` pins a known-good client set and falls
back to yt-dlp's own defaults, so these tests pin down three things: the
options actually handed to yt-dlp, which errors trigger a retry, and that
errors a different client cannot fix are not retried pointlessly.
"""

import pytest

import mazinger.download as D


# The exact message from the report, ANSI colour codes and all.
RELOAD_ERROR = (
    "\x1b[0;31mERROR:\x1b[0m [youtube] Ybf8o-dt0AE: The page needs to be reloaded."
)


# -- The options handed to yt-dlp ----------------------------------------


def test_common_opts_pin_the_working_player_clients():
    opts = D._yt_dlp_common_opts()
    assert opts["extractor_args"] == {
        "youtube": {
            "player_client": [
                "web_safari",
                "web_embedded",
                "visionos",
                "-tv_downgraded",
            ]
        }
    }


def test_tv_downgraded_is_excluded():
    # tv_downgraded is what returns "The page needs to be reloaded", and it is
    # in yt-dlp's default set for *authenticated* (cookie-carrying) sessions --
    # exactly the path Mazinger takes once a user supplies cookies.
    clients = D._yt_dlp_common_opts()["extractor_args"]["youtube"]["player_client"]
    assert "-tv_downgraded" in clients
    assert "tv_downgraded" not in clients


def test_a_jsless_client_is_always_offered():
    # web_safari and web_embedded both need a JS runtime.  Without visionos,
    # a box with no Node would extract zero formats where plain yt-dlp works.
    clients = D._yt_dlp_common_opts()["extractor_args"]["youtube"]["player_client"]
    assert "visionos" in clients


def test_empty_client_tuple_leaves_yt_dlp_defaults_alone():
    # The fallback tier must not send extractor_args at all, otherwise it is
    # not really falling back to yt-dlp's own client selection.
    assert "extractor_args" not in D._yt_dlp_common_opts(())


def test_common_opts_keep_the_pre_existing_settings():
    opts = D._yt_dlp_common_opts()
    assert opts["ignoreconfig"] is True
    assert opts["noplaylist"] is True
    assert opts["js_runtimes"] == {"node": {}}


# -- Which errors are worth another client -------------------------------


@pytest.mark.parametrize(
    "message",
    [
        RELOAD_ERROR,
        "ERROR: [youtube] abc: no video formats found",
        "ERROR: [youtube] abc: Failed to extract any player response",
        "ERROR: [youtube] abc: This content isn't available.",
        # Raised when every requested client came back with nothing usable --
        # e.g. JS-requiring clients on a box with no JavaScript runtime.
        "ERROR: [youtube] abc: Requested format is not available. Use --list-formats",
    ],
)
def test_player_client_errors_are_retryable(message):
    assert D.is_player_client_error(message)


@pytest.mark.parametrize(
    "message",
    [
        # Cookie problems are the other recovery path; switching clients does
        # not substitute for authentication, so these must not be retried.
        "ERROR: [youtube] abc: Sign in to confirm you're not a bot. Use --cookies",
        "ERROR: [youtube] abc: Private video. Sign in if you've been granted access",
        "ERROR: [youtube] abc: Video unavailable",
    ],
)
def test_non_player_client_errors_are_not_retryable(message):
    assert not D.is_player_client_error(message)


# -- The retry ladder ----------------------------------------------------


def test_fallback_retries_with_yt_dlp_defaults_after_a_reload_error():
    seen = []

    def call(opts):
        seen.append(opts.get("extractor_args"))
        if len(seen) == 1:
            raise RuntimeError(RELOAD_ERROR)
        return "ok"

    result = D._run_with_player_fallback(D._yt_dlp_common_opts, call, what="Test")

    assert result == "ok"
    assert len(seen) == 2
    assert seen[0]["youtube"]["player_client"][0] == "web_safari"  # pinned set
    assert seen[1] is None  # second attempt used yt-dlp's defaults


def test_first_attempt_wins_without_a_retry():
    calls = []

    def call(opts):
        calls.append(opts)
        return "ok"

    assert D._run_with_player_fallback(D._yt_dlp_common_opts, call, what="Test") == "ok"
    assert len(calls) == 1


def test_cookie_errors_raise_immediately():
    calls = []

    def call(opts):
        calls.append(opts)
        raise RuntimeError("ERROR: Sign in to confirm you're not a bot. Use --cookies")

    with pytest.raises(RuntimeError, match="not a bot"):
        D._run_with_player_fallback(D._yt_dlp_common_opts, call, what="Test")

    # No point burning a second request on an error cookies alone can fix.
    assert len(calls) == 1


def test_the_last_attempts_error_propagates():
    def call(opts):
        raise RuntimeError(RELOAD_ERROR)

    with pytest.raises(RuntimeError, match="needs to be reloaded"):
        D._run_with_player_fallback(D._yt_dlp_common_opts, call, what="Test")


# -- The environment override --------------------------------------------


def test_env_override_replaces_the_whole_ladder(monkeypatch):
    monkeypatch.setenv(D.PLAYER_CLIENT_ENV, "tv, mweb")
    assert D._player_client_attempts() == (("tv", "mweb"),)


def test_env_override_can_hand_control_back_to_yt_dlp(monkeypatch):
    monkeypatch.setenv(D.PLAYER_CLIENT_ENV, "default")
    attempts = D._player_client_attempts()
    assert attempts == (("default",),)
    # "default" is yt-dlp's own keyword for its built-in client set.
    assert D._yt_dlp_common_opts(attempts[0])["extractor_args"] == {
        "youtube": {"player_client": ["default"]}
    }


def test_blank_env_override_is_ignored(monkeypatch):
    monkeypatch.setenv(D.PLAYER_CLIENT_ENV, "   ")
    assert D._player_client_attempts() == D._YT_PLAYER_CLIENT_ATTEMPTS
