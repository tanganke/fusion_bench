"""
Tests for PyinstrumentProfilerMixin.
"""

import os
import tempfile
import unittest
from unittest.mock import Mock, patch

from fusion_bench.mixins.pyinstrument import PyinstrumentProfilerMixin


class TestPyinstrumentProfilerMixin(unittest.TestCase):
    """Test cases for PyinstrumentProfilerMixin."""

    def setUp(self):
        """Set up test fixtures."""
        self.mixin = PyinstrumentProfilerMixin()

    @patch("fusion_bench.mixins.pyinstrument.Profiler")
    def test_mixin_with_pyinstrument_mock(self, mock_profiler_class):
        """Test the mixin with a mocked pyinstrument profiler."""
        mock_profiler = Mock()
        mock_profiler_class.return_value = mock_profiler

        # Create a new instance to get the mocked profiler
        mixin = PyinstrumentProfilerMixin()

        # Test start_profile
        mixin.start_profile("test_action")
        mock_profiler.start.assert_called_once()
        self.assertTrue(mixin._is_profiling)

        # Test stop_profile
        mixin.stop_profile("test_action")
        mock_profiler.stop.assert_called_once()
        self.assertFalse(mixin._is_profiling)

        # Test context manager
        with mixin.profile("context_test"):
            pass

        # Should have been called twice more (once for context manager)
        self.assertEqual(mock_profiler.start.call_count, 2)
        self.assertEqual(mock_profiler.stop.call_count, 2)

    @patch("fusion_bench.mixins.pyinstrument.Profiler")
    def test_save_profile_report_mock(self, mock_profiler_class):
        """Test saving profile report with mocked profiler."""
        mock_profiler = Mock()
        mock_profiler.output_html.return_value = "<html>test report</html>"
        mock_profiler.output_json.return_value = '{"test": "data"}'
        mock_profiler.output_text.return_value = "test text report"
        mock_profiler_class.return_value = mock_profiler

        mixin = PyinstrumentProfilerMixin()

        with tempfile.TemporaryDirectory() as temp_dir:
            # Test HTML output
            html_path = os.path.join(temp_dir, "test_report.html")
            mixin.save_profile_report(html_path, format="html")

            with open(html_path, "r") as f:
                content = f.read()
            self.assertEqual(content, "<html>test report</html>")

            # Test text output
            text_path = os.path.join(temp_dir, "test_report.txt")
            mixin.save_profile_report(text_path, format="text")

            with open(text_path, "r") as f:
                content = f.read()
            self.assertEqual(content, "test text report")

    @patch("fusion_bench.mixins.pyinstrument.Profiler")
    def test_invalid_format_raises_error(self, mock_profiler_class):
        """Test that invalid format raises ValueError."""
        mock_profiler = Mock()
        mock_profiler_class.return_value = mock_profiler

        mixin = PyinstrumentProfilerMixin()

        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "test.invalid")

            # Should raise ValueError for invalid format
            with self.assertRaises(ValueError):
                mixin.save_profile_report(path, format="invalid")

    def test_reset_profile(self):
        """Test resetting the profiler."""
        # Start profiling
        self.mixin.start_profile("test")
        self.assertTrue(self.mixin._is_profiling)

        # Reset should stop profiling and clear profiler
        self.mixin.reset_profile()
        self.assertFalse(self.mixin._is_profiling)
        self.assertIsNone(self.mixin._profiler)

    def test_double_start_stop(self):
        """Test that double start/stop doesn't cause issues."""
        # Multiple starts should be safe
        self.mixin.start_profile("test1")
        self.mixin.start_profile("test2")  # Should be ignored

        # Multiple stops should be safe
        self.mixin.stop_profile("test1")
        self.mixin.stop_profile("test2")  # Should be ignored


if __name__ == "__main__":
    unittest.main()
