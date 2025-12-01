"""
Scene Understanding Module

Provides scene description capabilities for Scene Aria System
using FastVLM vision-language model.

Components:
- FastVLMWrapper: Main interface for scene description
- describe_image: Convenience function for quick usage

Performance (RTX 2060):
- Model: 372ms average latency
- E2E: 752ms (industry standard compliant)
- VRAM: 1.19 GB

Author: Roberto Teleng
Project: Scene Aria System
"""

from .fastvlm_wrapper import FastVLMWrapper, describe_image

__all__ = ['FastVLMWrapper', 'describe_image']
__version__ = '0.1.0'