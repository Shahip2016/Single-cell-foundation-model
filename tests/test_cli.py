"""
Unit Tests for CellFM CLI
============================

Tests for the command-line interface argument parsing and
the info command execution.

Run with: python -m pytest tests/test_cli.py -v
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cellfm.cli import create_parser, cmd_info, main


class TestCLIParser:
    """Test CLI argument parsing."""

    def test_create_parser(self):
        """Parser should be created without errors."""
        parser = create_parser()
        assert parser is not None

    def test_info_command(self):
        """Should parse 'info' command."""
        parser = create_parser()
        args = parser.parse_args(["info", "--config", "80M"])
        assert args.command == "info"
        assert args.config == "80M"

    def test_info_800m(self):
        """Should parse 800M config."""
        parser = create_parser()
        args = parser.parse_args(["info", "--config", "800M"])
        assert args.config == "800M"

    def test_train_command(self):
        """Should parse 'train' command with required args."""
        parser = create_parser()
        args = parser.parse_args([
            "train", "--data", "cells.h5ad",
            "--epochs", "10", "--lr", "0.001",
        ])
        assert args.command == "train"
        assert args.data == "cells.h5ad"
        assert args.epochs == 10
        assert args.lr == 0.001

    def test_train_defaults(self):
        """Train should have sensible defaults."""
        parser = create_parser()
        args = parser.parse_args(["train", "--data", "cells.h5ad"])
        assert args.epochs == 5
        assert args.batch_size == 16
        assert args.lr == 1e-4
        assert args.lora_rank == 0
        assert args.device == "auto"

    def test_predict_command(self):
        """Should parse 'predict' command."""
        parser = create_parser()
        args = parser.parse_args([
            "predict",
            "--checkpoint", "best.pt",
            "--data", "test.h5ad",
            "--output", "preds.npy",
        ])
        assert args.command == "predict"
        assert args.checkpoint == "best.pt"
        assert args.data == "test.h5ad"
        assert args.output == "preds.npy"

    def test_embed_command(self):
        """Should parse 'embed' command."""
        parser = create_parser()
        args = parser.parse_args([
            "embed",
            "--checkpoint", "best.pt",
            "--data", "cells.h5ad",
            "--output", "emb.npy",
        ])
        assert args.command == "embed"
        assert args.output == "emb.npy"

    def test_no_command_exits(self):
        """No command should exit cleanly."""
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code == 0

    def test_version(self):
        """Version flag should exit."""
        with pytest.raises(SystemExit):
            main(["--version"])


class TestCmdInfo:
    """Test the info command execution."""

    def test_info_80m(self, capsys):
        """Info command should display 80M config."""
        parser = create_parser()
        args = parser.parse_args(["info", "--config", "80M"])
        cmd_info(args)

        captured = capsys.readouterr()
        assert "80M" in captured.out
        assert "Layers" in captured.out

    def test_info_800m(self, capsys):
        """Info command should display 800M config."""
        parser = create_parser()
        args = parser.parse_args(["info", "--config", "800M"])
        cmd_info(args)

        captured = capsys.readouterr()
        assert "800M" in captured.out

    def test_main_info(self, capsys):
        """Main function should dispatch to info."""
        main(["info", "--config", "80M"])
        captured = capsys.readouterr()
        assert "CellFM" in captured.out


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
