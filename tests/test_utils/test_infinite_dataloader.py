import unittest

from torch.utils.data import DataLoader, Dataset

from fusion_bench.utils.data import InfiniteDataLoader
from fusion_bench.utils.validation import ValidationError


class SimpleDataset(Dataset):
    """Simple dataset for testing"""

    def __init__(self, size=10):
        self.size = size
        self.data = list(range(size))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


class TestInfiniteDataLoader(unittest.TestCase):
    """Test cases for InfiniteDataLoader"""

    def setUp(self):
        """Set up test fixtures"""
        self.dataset = SimpleDataset(size=5)
        self.dataloader = DataLoader(self.dataset, batch_size=2)

    def test_initialization_with_valid_dataloader(self):
        """Test that InfiniteDataLoader initializes correctly with valid DataLoader"""
        infinite_loader = InfiniteDataLoader(self.dataloader)
        self.assertIsNotNone(infinite_loader)
        self.assertEqual(infinite_loader.data_loader, self.dataloader)
        self.assertEqual(infinite_loader.iteration_count, 0)

    def test_initialization_with_none(self):
        """Test that initialization with None raises ValidationError"""
        with self.assertRaises(ValidationError) as cm:
            InfiniteDataLoader(None)
        self.assertIn("cannot be None", str(cm.exception))

    def test_initialization_with_invalid_type(self):
        """Test that initialization with non-DataLoader raises ValidationError"""
        with self.assertRaises(ValidationError) as cm:
            InfiniteDataLoader("not a dataloader")
        self.assertIn("must be a DataLoader instance", str(cm.exception))

        with self.assertRaises(ValidationError) as cm:
            InfiniteDataLoader([1, 2, 3])
        self.assertIn("must be a DataLoader instance", str(cm.exception))

    def test_iteration_basic(self):
        """Test basic iteration through the dataset once"""
        infinite_loader = InfiniteDataLoader(self.dataloader)
        items = []
        for i, batch in enumerate(infinite_loader):
            items.extend(batch.tolist())
            if i >= 2:  # Get 3 batches (covers all 5 items)
                break

        self.assertEqual(len(items), 5)
        self.assertEqual(sorted(items), [0, 1, 2, 3, 4])

    def test_infinite_iteration(self):
        """Test that iteration continues beyond dataset length"""
        infinite_loader = InfiniteDataLoader(self.dataloader)
        items = []
        for i, batch in enumerate(infinite_loader):
            items.extend(batch.tolist())
            if i >= 5:  # Get more batches than dataset has
                break

        # Should have more items than the original dataset
        self.assertGreater(len(items), 5)
        # Should have at least one complete iteration
        self.assertGreaterEqual(infinite_loader.iteration_count, 1)

    def test_iteration_count(self):
        """Test that iteration_count is tracked correctly"""
        infinite_loader = InfiniteDataLoader(self.dataloader)

        # Initial count should be 0
        self.assertEqual(infinite_loader.iteration_count, 0)

        # Iterate through entire dataset once
        for _ in range(len(self.dataloader)):
            next(infinite_loader)

        # Should still be 0 (not completed a full cycle yet)
        self.assertEqual(infinite_loader.iteration_count, 0)

        # One more iteration triggers reset
        next(infinite_loader)
        self.assertEqual(infinite_loader.iteration_count, 1)

        # Continue for another full cycle
        for _ in range(len(self.dataloader)):
            next(infinite_loader)

        next(infinite_loader)
        self.assertEqual(infinite_loader.iteration_count, 2)

    def test_reset_method(self):
        """Test that reset() method works correctly"""
        infinite_loader = InfiniteDataLoader(self.dataloader)

        # Iterate a bit
        for i in range(10):
            next(infinite_loader)

        initial_count = infinite_loader.iteration_count
        self.assertGreater(initial_count, 0)

        # Reset
        infinite_loader.reset()

        # Count should be back to 0
        self.assertEqual(infinite_loader.iteration_count, 0)

    def test_iter_reset(self):
        """Test that calling iter() resets the loader"""
        infinite_loader = InfiniteDataLoader(self.dataloader)

        # Iterate a bit
        for i in range(10):
            next(infinite_loader)

        self.assertGreater(infinite_loader.iteration_count, 0)

        # Call iter() again
        iter(infinite_loader)

        # Count should be reset
        self.assertEqual(infinite_loader.iteration_count, 0)

    def test_len(self):
        """Test that __len__ returns correct length"""
        infinite_loader = InfiniteDataLoader(self.dataloader)
        self.assertEqual(len(infinite_loader), len(self.dataloader))

    def test_for_loop_usage(self):
        """Test that InfiniteDataLoader works in a for loop with break"""
        infinite_loader = InfiniteDataLoader(self.dataloader)

        count = 0
        for batch in infinite_loader:
            count += 1
            if count >= 20:  # Much more than dataset length
                break

        self.assertEqual(count, 20)
        self.assertGreater(infinite_loader.iteration_count, 0)

    def test_multiple_iterations_with_iter(self):
        """Test multiple separate iterations with iter()"""
        infinite_loader = InfiniteDataLoader(self.dataloader)

        # First iteration
        items1 = []
        for i, batch in enumerate(infinite_loader):
            items1.extend(batch.tolist())
            if i >= 2:
                break

        # Second iteration (should reset)
        items2 = []
        for i, batch in enumerate(infinite_loader):
            items2.extend(batch.tolist())
            if i >= 2:
                break

        # Both should produce the same sequence
        self.assertEqual(items1, items2)

    def test_empty_dataloader(self):
        """Test behavior with empty dataset"""
        empty_dataset = SimpleDataset(size=0)
        empty_loader = DataLoader(empty_dataset)

        # This should raise an error when trying to iterate
        infinite_loader = InfiniteDataLoader(empty_loader, max_retries=1)

        with self.assertRaises(RuntimeError) as cm:
            next(infinite_loader)
        error_msg = str(cm.exception)
        self.assertIn("Failed to retrieve data from data loader", error_msg)
        self.assertIn("appears to be empty", error_msg)

    def test_max_retries_parameter(self):
        """Test that max_retries parameter is respected"""
        infinite_loader = InfiniteDataLoader(self.dataloader, max_retries=5)
        self.assertEqual(infinite_loader.max_retries, 5)

    def test_dataloader_with_shuffle(self):
        """Test InfiniteDataLoader with shuffled DataLoader"""
        shuffled_loader = DataLoader(self.dataset, batch_size=2, shuffle=True)
        infinite_loader = InfiniteDataLoader(shuffled_loader)

        # Should still work even with shuffling
        items = []
        for i, batch in enumerate(infinite_loader):
            items.extend(batch.tolist())
            if i >= 5:
                break

        self.assertGreater(len(items), 5)

    def test_with_larger_dataset(self):
        """Test with a larger dataset to ensure robustness"""
        large_dataset = SimpleDataset(size=100)
        large_loader = DataLoader(large_dataset, batch_size=10)
        infinite_loader = InfiniteDataLoader(large_loader)

        # Iterate through multiple epochs worth of data
        total_items = 0
        for i, batch in enumerate(infinite_loader):
            total_items += len(batch)
            if i >= 50:  # 51 batches, should be multiple epochs
                break

        self.assertGreater(total_items, 100)
        self.assertGreater(infinite_loader.iteration_count, 1)


if __name__ == "__main__":
    unittest.main()
