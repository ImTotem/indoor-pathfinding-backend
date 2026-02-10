# slam_engines/rtabmap/config_generator.py
from pathlib import Path
from typing import Optional, Dict
import yaml

from . import constants


class ConfigGenerator:
    """RTAB-Map configuration file generator for standalone mode"""
    
    def generate(
        self,
        output_path: str,
        camera_intrinsics: Optional[Dict] = None,
        custom_params: Optional[Dict] = None
    ) -> str:
        """
        Generate RTAB-Map parameter YAML configuration file.
        
        Args:
            output_path: Path where the YAML config file will be written
            camera_intrinsics: Optional dict with camera calibration parameters
                              (fx, fy, cx, cy, k1, k2, p1, p2, width, height)
            custom_params: Optional dict of custom RTAB-Map parameters to override defaults
        
        Returns:
            str: Path to the generated configuration file
        """
        # Start with default parameters
        params = constants.DEFAULT_PARAMS.copy()
        
        # Merge custom parameters if provided
        if custom_params:
            params.update(custom_params)
        
        # Build configuration structure
        config = {
            'rtabmap_params': params,
            'camera_info': camera_intrinsics if camera_intrinsics else {}
        }
        
        # Create output directory if it doesn't exist
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Write YAML configuration file
        with open(output_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        print(f"[ConfigGenerator] RTAB-Map config generated: {output_path}")
        return str(output_file)
    
    def params_to_cli_args(self, params: Dict) -> list:
        """
        Convert RTAB-Map parameters dictionary to command-line arguments.
        
        Converts a dict like {'Mem/IncrementalMemory': 'true', 'Kp/MaxFeatures': '400'}
        to a list like ['-param', 'Mem/IncrementalMemory', 'true', '-param', 'Kp/MaxFeatures', '400']
        
        Args:
            params: Dictionary of RTAB-Map parameters
        
        Returns:
            list: Command-line arguments in format ['-param', 'Key', 'value', '-param', 'Key2', 'value2', ...]
        """
        args = []
        for key, value in params.items():
            args.extend(['-param', key, str(value)])
        return args
