"""PixelPeeper — screenshot pixel-grid analyser."""
from .analysis import AnalysisResult, AxisEvidence, ResolutionCandidate, analyse_image, analyse_region

__version__ = "1.5.0"
__all__ = ["AnalysisResult", "AxisEvidence", "ResolutionCandidate", "analyse_image", "analyse_region"]
