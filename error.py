class EmotionDetectionError(Exception):
    """Base exception class for Emotion Detection project."""

    def __init__(self, message="An error occurred in the Emotion Detection project."):
        self.message = message
        super().__init__(self.message)


class FacialDetectionError(EmotionDetectionError):
    """Exception raised for errors in facial detection."""

    def __init__(self, message="Error occurred in facial detection."):
        super().__init__(message)


class EmotionAnalysisError(EmotionDetectionError):
    """Exception raised for errors in emotion analysis."""

    def __init__(self, message="Error occurred in emotion analysis."):
        super().__init__(message)
