import os
import unittest
import yaml
import shutil
from pathlib import Path
from unittest.mock import MagicMock

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys_path_added = False
if str(_PROJECT_ROOT / "src") not in os.sys.path:
    os.sys.path.append(str(_PROJECT_ROOT / "src"))
    sys_path_added = True

from novelty_eval.ablation.merge_ablation_runs import _derive_pointwise_partner_excluded

class TestMergeFilteringRobustness(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("temp_test_filtering").resolve()
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        self.test_dir.mkdir(exist_ok=True)
        
        # 1. Create mock pairwise source
        self.source_pairwise = self.test_dir / "source_pairwise.yaml"
        self.pairwise_data = {
            '3': {
                'context': 'deep learning safety',
                'ideas': {
                    '0': 'Original Abstract for Blocked Paper',
                    '1': 'Original Abstract for Partner Idea'
                },
                'metadata': {
                    '0': {'title': 'Blocked Paper Title'},
                    '1': {'title': 'Partner Idea Title'}
                },
                'expected_winners': [0]
            }
        }
        with open(self.source_pairwise, 'w') as f:
            yaml.dump(self.pairwise_data, f)

        # 2. Create mock pointwise instances with transformed text
        self.pointwise_instances = self.test_dir / "pointwise_instances.yaml"
        self.pointwise_data = {
            '6': {
                'context': 'deep learning safety',
                'idea': 'PLAN: Transformed text for Blocked Paper',
                'metadata': {'title': 'Blocked Paper Title'}
            },
            '7': {
                'context': 'deep learning safety',
                'idea': 'PLAN: Transformed text for Partner Idea',
                'metadata': {'title': 'Partner Idea Title'}
            }
        }
        with open(self.pointwise_instances, 'w') as f:
            yaml.dump(self.pointwise_data, f)

        # 3. Create mock _instance_creation_config.yaml
        self.config_path = self.test_dir / "_instance_creation_config.yaml"
        self.config_data = {
            'derive_pointwise_from_pairwise': str(self.source_pairwise)
        }
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config_data, f)

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_pointwise_partner_fallback_matching(self):
        """
        Verify that _derive_pointwise_partner_excluded correctly identifies 
        partners even when idea texts have changed, using metadata fallback.
        """
        blocked_titles = {"blocked paper title"}
        
        # This should trigger the fallback because 'PLAN: Transformed text...' 
        # won't match 'Original Abstract for Partner Idea'
        excluded = _derive_pointwise_partner_excluded(
            str(self.pointwise_instances),
            blocked_titles
        )
        
        # We expect index '7' to be identified as the partner of index '6'
        self.assertIn('7', excluded)
        self.assertEqual(excluded['7'], 'Partner Idea Title')
        self.assertEqual(len(excluded), 1)

    def test_generic_title_collision_disambiguation(self):
        """
        Verify that the neighbor rule correctly identifies the partner even if 
        the same generic title exists at a distant index.
        """
        # Add a distant colliding title
        with open(self.pointwise_instances, 'r') as f:
            data = yaml.safe_load(f)
        
        data['100'] = {
            'context': 'different context',
            'idea': 'Some unrelated idea',
            'metadata': {'title': 'Partner Idea Title'} # Same title as the partner at index 7
        }
        with open(self.pointwise_instances, 'w') as f:
            yaml.dump(data, f)
            
        excluded = _derive_pointwise_partner_excluded(
            str(self.pointwise_instances),
            {"blocked paper title"}
        )
        
        # Should still only pick '7' because it's the neighbor. '100' should be ignored.
        self.assertIn('7', excluded)
        self.assertNotIn('100', excluded)
        self.assertEqual(len(excluded), 1)

    def test_metadata_mismatch_fails_gracefully(self):
        """
        Verify that if the neighbor exists but its metadata doesn't match the 
        expected partner, it is not excluded.
        """
        # Change the metadata of the partner at index 7
        with open(self.pointwise_instances, 'r') as f:
            data = yaml.safe_load(f)
        data['7']['metadata']['title'] = "Completely Different Title"
        with open(self.pointwise_instances, 'w') as f:
            yaml.dump(data, f)

        excluded = _derive_pointwise_partner_excluded(
            str(self.pointwise_instances),
            {"blocked paper title"}
        )
        
        # No partner should be found
        self.assertEqual(excluded, {})

    def test_warning_on_missing_partner(self):
        """
        Verify that a warning is logged when a partner is identified in the 
        source but not found in the pointwise data.
        """
        # Remove the partner from pointwise data entirely
        with open(self.pointwise_instances, 'r') as f:
            data = yaml.safe_load(f)
        del data['7']
        with open(self.pointwise_instances, 'w') as f:
            yaml.dump(data, f)

        mock_logger = MagicMock()
        excluded = _derive_pointwise_partner_excluded(
            str(self.pointwise_instances),
            {"blocked paper title"},
            logger_inst=mock_logger
        )
        
        # Partner not found
        self.assertEqual(excluded, {})
        
        # Warning should have been logged
        mock_logger.warning.assert_called()
        args, _ = mock_logger.warning.call_args
        self.assertIn("Pointwise partner matching failed", args[0])
        self.assertIn("Partner Idea Title", args[0])

    def test_pointwise_partner_no_config(self):
        """Verify that it returns empty dict if config is missing."""
        os.remove(self.config_path)
        excluded = _derive_pointwise_partner_excluded(
            str(self.pointwise_instances),
            {"blocked paper title"}
        )
        self.assertEqual(excluded, {})

    def test_pointwise_partner_no_source(self):
        """Verify that it returns empty dict if source file is missing."""
        os.remove(self.source_pairwise)
        excluded = _derive_pointwise_partner_excluded(
            str(self.pointwise_instances),
            {"blocked paper title"}
        )
        self.assertEqual(excluded, {})

if __name__ == "__main__":
    unittest.main()
