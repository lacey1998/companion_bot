"""
Simple emotion detection module for companion bot.
Detects emotion from user input to match training data format.
"""

import re
from typing import Dict, List, Optional


class EmotionDetector:
    """
    Detects emotion from user input using keyword matching.
    
    Matches EXACTLY the 32 emotions from the Empathetic Dialogues dataset:
    afraid, angry, annoyed, anticipating, anxious, apprehensive, ashamed, caring,
    confident, content, devastated, disappointed, disgusted, embarrassed, excited,
    faithful, furious, grateful, guilty, hopeful, impressed, jealous, joyful,
    lonely, nostalgic, prepared, proud, sad, sentimental, surprised, terrified, trusting
    
    Can be replaced with a trained classifier later for better accuracy.
    """
    
    def __init__(self):
        # Emotion keywords matching EXACTLY the 32 emotions from Empathetic Dialogues dataset
        # Dataset emotions: afraid, angry, annoyed, anticipating, anxious, apprehensive, 
        # ashamed, caring, confident, content, devastated, disappointed, disgusted, 
        # embarrassed, excited, faithful, furious, grateful, guilty, hopeful, impressed, 
        # jealous, joyful, lonely, nostalgic, prepared, proud, sad, sentimental, 
        # surprised, terrified, trusting
        self.emotion_keywords = {
            "afraid": ["afraid", "frightened", "scared", "fear", "fearful"],
            "angry": ["angry", "mad", "rage", "irritated", "frustrated"],
            "annoyed": ["annoyed", "irritated", "bothered", "frustrated", "irked"],
            "anticipating": ["anticipating", "expecting", "looking forward", "awaiting"],
            "anxious": ["anxious", "worried", "nervous", "stressed", "tense", "uneasy"],
            "apprehensive": ["apprehensive", "worried", "concerned", "uneasy", "doubtful"],
            "ashamed": ["ashamed", "humiliated", "embarrassed", "disgraced"],
            "caring": ["caring", "concerned", "compassionate", "sympathetic"],
            "confident": ["confident", "sure", "certain", "assured", "self-assured"],
            "content": ["content", "satisfied", "pleased", "fulfilled", "at ease"],
            "devastated": ["devastated", "crushed", "destroyed", "shattered", "heartbroken"],
            "disappointed": ["disappointed", "let down", "discouraged", "disheartened"],
            "disgusted": ["disgusted", "revolted", "sickened", "repulsed", "appalled"],
            "embarrassed": ["embarrassed", "awkward", "self-conscious", "uncomfortable"],
            "excited": ["excited", "thrilled", "eager", "enthusiastic", "pumped"],
            "faithful": ["faithful", "loyal", "devoted", "committed", "dedicated"],
            "furious": ["furious", "enraged", "livid", "outraged", "incensed"],
            "grateful": ["grateful", "thankful", "appreciative", "blessed"],
            "guilty": ["guilty", "remorseful", "regretful", "sorry", "ashamed"],
            "hopeful": ["hopeful", "optimistic", "positive", "expectant"],
            "impressed": ["impressed", "amazed", "awed", "admiring"],
            "jealous": ["jealous", "envious", "covetous", "resentful"],
            "joyful": ["joyful", "happy", "glad", "delighted", "cheerful", "ecstatic"],
            "lonely": ["lonely", "alone", "isolated", "abandoned", "lonesome"],
            "nostalgic": ["nostalgic", "homesick", "wistful", "yearning"],
            "prepared": ["prepared", "ready", "organized", "set"],
            "proud": ["proud", "accomplished", "achieved", "successful", "honored"],
            "sad": ["sad", "depressed", "down", "unhappy", "miserable", "upset", "crying"],
            "sentimental": ["sentimental", "remembering", "reminiscing", "nostalgic", "emotional"],
            "surprised": ["surprised", "shocked", "astonished", "amazed", "startled"],
            "terrified": ["terrified", "horrified", "petrified", "frightened", "scared"],
            "trusting": ["trusting", "confident in", "relying on", "faithful"],
        }
        
        # Exact 32 emotions from Empathetic Dialogues dataset (for validation)
        self.emotion_categories = {
            "afraid", "angry", "annoyed", "anticipating", "anxious", "apprehensive",
            "ashamed", "caring", "confident", "content", "devastated", "disappointed",
            "disgusted", "embarrassed", "excited", "faithful", "furious", "grateful",
            "guilty", "hopeful", "impressed", "jealous", "joyful", "lonely",
            "nostalgic", "prepared", "proud", "sad", "sentimental", "surprised",
            "terrified", "trusting"
        }
    
    def detect_emotion(self, text: str) -> str:
        """
        Detect emotion from user input text.
        
        Args:
            text: User input text
            
        Returns:
            Detected emotion label (e.g., "sad", "happy", "anxious")
        """
        text_lower = text.lower()
        
        # Count matches for each emotion
        emotion_scores = {}
        for emotion, keywords in self.emotion_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                emotion_scores[emotion] = score
        
        # Return emotion with highest score
        if emotion_scores:
            detected_emotion = max(emotion_scores, key=emotion_scores.get)
            return detected_emotion
        
        # Default to neutral/supportive if no emotion detected
        return "neutral"
    
    def detect_emotion_with_confidence(self, text: str) -> tuple[str, float]:
        """
        Detect emotion with confidence score.
        
        Returns:
            (emotion, confidence) tuple where confidence is 0.0-1.0
        """
        text_lower = text.lower()
        
        emotion_scores = {}
        for emotion, keywords in self.emotion_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                emotion_scores[emotion] = score
        
        if emotion_scores:
            detected_emotion = max(emotion_scores, key=emotion_scores.get)
            max_score = emotion_scores[detected_emotion]
            total_keywords = len(self.emotion_keywords[detected_emotion])
            confidence = min(max_score / total_keywords, 1.0)
            return detected_emotion, confidence
        
        return "neutral", 0.0
    
    def is_valid_emotion(self, emotion: str) -> bool:
        """
        Check if an emotion is one of the 32 valid emotions from Empathetic Dialogues.
        
        Args:
            emotion: Emotion label to validate
            
        Returns:
            True if emotion is valid, False otherwise
        """
        return emotion.lower() in self.emotion_categories
    
    def get_all_emotions(self) -> List[str]:
        """
        Get list of all 32 valid emotions from Empathetic Dialogues dataset.
        
        Returns:
            Sorted list of emotion labels
        """
        return sorted(list(self.emotion_categories))


# Simple function interface for easy import
def detect_emotion(text: str) -> str:
    """Simple function to detect emotion from text."""
    detector = EmotionDetector()
    return detector.detect_emotion(text)

