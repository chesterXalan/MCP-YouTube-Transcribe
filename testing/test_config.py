"""Unit tests for Whisper configuration and model download logic."""

import asyncio
import os
import sys
from unittest.mock import AsyncMock, patch

import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from youtube_tool import VALID_MODELS, WHISPER_CPP_FILENAMES, _download_model, _get_config


# --- _get_config() ---


class TestGetConfig:
    def test_defaults(self, monkeypatch):
        monkeypatch.delenv("WHISPER_MODEL", raising=False)
        monkeypatch.delenv("WHISPER_MODEL_DIR", raising=False)
        monkeypatch.delenv("WHISPER_LANGUAGE", raising=False)

        config = _get_config()

        assert config["model_name"] == "tiny"
        assert config["model_dir"].endswith("models")
        assert config["language"] == "auto"

    @pytest.mark.parametrize("model", VALID_MODELS)
    def test_valid_models(self, monkeypatch, model):
        monkeypatch.setenv("WHISPER_MODEL", model)
        config = _get_config()
        assert config["model_name"] == model

    def test_invalid_model_raises(self, monkeypatch):
        monkeypatch.setenv("WHISPER_MODEL", "nonexistent")
        with pytest.raises(ValueError, match="Invalid WHISPER_MODEL"):
            _get_config()

    def test_custom_model_dir(self, monkeypatch):
        monkeypatch.setenv("WHISPER_MODEL_DIR", "/tmp/my-models")
        config = _get_config()
        assert config["model_dir"] == "/tmp/my-models"

    def test_custom_language(self, monkeypatch):
        monkeypatch.setenv("WHISPER_LANGUAGE", "ja")
        config = _get_config()
        assert config["language"] == "ja"

    def test_whitespace_stripped(self, monkeypatch):
        monkeypatch.setenv("WHISPER_MODEL", "  base  ")
        monkeypatch.setenv("WHISPER_LANGUAGE", " zh ")
        config = _get_config()
        assert config["model_name"] == "base"
        assert config["language"] == "zh"

    def test_empty_model_dir_uses_default(self, monkeypatch):
        monkeypatch.setenv("WHISPER_MODEL_DIR", "   ")
        config = _get_config()
        assert config["model_dir"].endswith("models")


# --- _download_model() ---


class TestDownloadModel:
    def test_existing_model_returns_path(self, tmp_path):
        model_file = tmp_path / "ggml-tiny.bin"
        model_file.write_bytes(b"fake model data")

        result = _download_model("tiny", str(tmp_path))

        assert result == str(model_file)

    def test_missing_model_triggers_download(self, tmp_path):
        target = tmp_path / "ggml-base.bin"

        def fake_urlretrieve(url, path, reporthook=None):
            with open(path, "wb") as f:
                f.write(b"downloaded model")

        with patch("youtube_tool.urllib.request.urlretrieve", side_effect=fake_urlretrieve):
            result = _download_model("base", str(tmp_path))

        assert result == str(target)
        assert target.exists()
        assert not (tmp_path / "ggml-base.bin.downloading").exists()

    def test_download_failure_cleans_up(self, tmp_path):
        with patch("youtube_tool.urllib.request.urlretrieve", side_effect=ConnectionError("network down")):
            with pytest.raises(RuntimeError, match="Failed to download model"):
                _download_model("small", str(tmp_path))

        assert not (tmp_path / "ggml-small.bin.downloading").exists()
        assert not (tmp_path / "ggml-small.bin").exists()

    def test_creates_model_dir_if_missing(self, tmp_path):
        new_dir = tmp_path / "sub" / "models"

        def fake_urlretrieve(url, path, reporthook=None):
            with open(path, "wb") as f:
                f.write(b"data")

        with patch("youtube_tool.urllib.request.urlretrieve", side_effect=fake_urlretrieve):
            _download_model("tiny", str(new_dir))

        assert new_dir.exists()


# --- WHISPER_CPP_FILENAMES ---


class TestFilenameMapping:
    def test_all_models_have_filenames(self):
        for model in VALID_MODELS:
            assert model in WHISPER_CPP_FILENAMES
            assert WHISPER_CPP_FILENAMES[model] == f"ggml-{model}.bin"


# --- MCP handler language passthrough ---


class TestMcpLanguagePassthrough:
    def test_language_passed_to_function(self):
        with patch("mcp_server.get_youtube_transcript") as mock_fn:
            mock_fn.return_value = {"status": "success", "title": "T", "url": "U", "source": "S", "transcript": "text"}

            from mcp_server import handle_call_tool

            asyncio.run(handle_call_tool("get_youtube_transcript", {
                "query": "test",
                "language": "zh",
            }))

            mock_fn.assert_called_once_with(query="test", force_whisper=False, language="zh")

    def test_language_none_when_omitted(self):
        with patch("mcp_server.get_youtube_transcript") as mock_fn:
            mock_fn.return_value = {"status": "success", "title": "T", "url": "U", "source": "S", "transcript": "text"}

            from mcp_server import handle_call_tool

            asyncio.run(handle_call_tool("get_youtube_transcript", {
                "query": "test",
            }))

            mock_fn.assert_called_once_with(query="test", force_whisper=False, language=None)
