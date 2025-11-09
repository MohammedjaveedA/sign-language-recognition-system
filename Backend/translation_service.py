import os
import logging
import requests
import json
import glob
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class TranslationService:
    def __init__(self):
        self.supported_languages = {
            'en': 'English',
            'ka': 'Kannada',
            'hi': 'Hindi',
            'ta': 'Tamil',
            'te': 'Telugu',
            'ml': 'Malayalam',
            'bn': 'Bengali',
        }
        
        # Get default language from .env or use English as fallback
        default_lang = os.getenv('DEFAULT_LANGUAGE', 'en')
        self.target_language = default_lang if default_lang in self.supported_languages else 'en'
        
        # Load or create fallback translations
        self.fallback_file = "fallback_translations.json"
        self.fallback_translations = self._load_fallback_translations()
        
        # Auto-detect words from dataset and update fallback translations
        self._auto_update_fallback_from_dataset()
        
        print(f"🌐 Translation service initialized. Default language: {self.supported_languages[self.target_language]}")
        print(f"📚 Loaded {len(self.fallback_translations.get('en', {}))} words in fallback dictionary")
    
    def _auto_update_fallback_from_dataset(self):
        """Automatically detect new words from dataset and add to fallback translations"""
        dataset_words = self._scan_dataset_for_words()
        new_words_added = False
        
        for word in dataset_words:
            word_lower = word.lower()
            if word_lower not in self.fallback_translations.get('en', {}):
                # Add new word to English fallback
                self.fallback_translations.setdefault('en', {})[word_lower] = word_lower
                new_words_added = True
                print(f"➕ Added new word to fallback: '{word_lower}'")
        
        if new_words_added:
            self._save_fallback_translations()
            print(f"💾 Saved {len(dataset_words)} words to fallback dictionary")
    
    def _scan_dataset_for_words(self):
        """Scan the dataset directory for sign language words/folders"""
        dataset_words = set()
        data_dirs = ["sign_data", "dataset", "data", "train_data"]
        
        for data_dir in data_dirs:
            if os.path.exists(data_dir):
                try:
                    # Look for folders (each folder represents a sign/word)
                    for item in os.listdir(data_dir):
                        item_path = os.path.join(data_dir, item)
                        if os.path.isdir(item_path):
                            # Add folder name as a word
                            dataset_words.add(item)
                    
                    # Also check for any model files or landmark files that might contain class names
                    model_files = glob.glob(os.path.join(data_dir, "*.pkl")) + \
                                 glob.glob(os.path.join(data_dir, "*.npy")) + \
                                 glob.glob(os.path.join(data_dir, "*.json"))
                    
                    for model_file in model_files:
                        # Try to extract words from filenames
                        filename = os.path.basename(model_file)
                        # Remove extensions and common prefixes
                        clean_name = filename.replace('_landmarks', '').replace('_model', '').replace('.pkl', '').replace('.npy', '').replace('.json', '')
                        if clean_name and not clean_name.startswith('.'):
                            dataset_words.add(clean_name)
                            
                except Exception as e:
                    logging.error(f"Error scanning dataset directory {data_dir}: {e}")
        
        # Also check the trained model for classes
        try:
            if os.path.exists("sign_language_model.pkl"):
                import pickle
                with open("sign_language_model.pkl", 'rb') as f:
                    model = pickle.load(f)
                    if hasattr(model, 'classes_'):
                        for cls in model.classes_:
                            dataset_words.add(str(cls))
        except Exception as e:
            logging.error(f"Error reading model classes: {e}")
        
        return sorted(list(dataset_words))
    
    def _load_fallback_translations(self):
        """Load fallback translations from JSON file or create default"""
        try:
            if os.path.exists(self.fallback_file):
                with open(self.fallback_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logging.error(f"Error loading fallback translations: {e}")
        
        # Return default fallback translations
        return self._get_default_fallback_translations()
    
    def _get_default_fallback_translations(self):
        """Comprehensive default fallback translations for all languages"""
        return {
            'en': {
    'hello': 'hello',
    'thank you': 'thank you',
    'yes': 'yes',
    'no': 'no',
    'water': 'water',
},
'ka': {
    'hello': 'ನಮಸ್ಕಾರ',
    'thank you': 'ಧನ್ಯವಾದಗಳು',
    'yes': 'ಹೌದು',
    'no': 'ಇಲ್ಲ',
    'water': 'ನೀರು',
},
'hi': {
    'hello': 'नमस्ते',
    'thank you': 'धन्यवाद',
    'yes': 'हाँ',
    'no': 'नहीं',
    'water': 'पानी',
},
'bn': {
    'hello': 'হ্যালো',
    'thank you': 'ধন্যবাদ',
    'yes': 'হ্যাঁ',
    'no': 'না',
    'water': 'পানি',
},
'ta': {
    'hello': 'வணக்கம்',
    'thank you': 'நன்றி',
    'yes': 'ஆம்',
    'no': 'இல்லை',
    'water': 'தண்ணீர்',
},
'te': {
    'hello': 'హలో',
    'thank you': 'ధన్యవాదాలు',
    'yes': 'అవును',
    'no': 'కాదు',
    'water': 'నీరు',
},
'ml': {
    'hello': 'ഹലോ',
    'thank you': 'നന്ദി',
    'yes': 'അതെ',
    'no': 'ഇല്ല',
    'water': 'വെള്ളം',
}

        }
    
    def _save_fallback_translations(self):
        """Save fallback translations to JSON file"""
        try:
            with open(self.fallback_file, 'w', encoding='utf-8') as f:
                json.dump(self.fallback_translations, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error(f"Error saving fallback translations: {e}")
    
    def set_target_language(self, language_code):
        """Set the target language for translation"""
        if language_code in self.supported_languages:
            self.target_language = language_code
            logging.info(f"Language changed to: {self.supported_languages[language_code]}")
            return True
        else:
            logging.warning(f"Unsupported language code: {language_code}")
            return False
    
    def get_supported_languages(self):
        """Get list of supported languages"""
        return self.supported_languages
    
    def translate_text(self, text, source_lang='auto'):
        """
        Translate text to target language using LibreTranslate (free API)
        """
        if not text or text.strip() == '':
            return text, 'en'
        
        # First check if we have a fallback translation
        text_lower = text.lower()
        if (self.target_language in self.fallback_translations and 
            text_lower in self.fallback_translations[self.target_language]):
            return self.fallback_translations[self.target_language][text_lower], 'en'
        
        # Try online translation
        try:
            # Using LibreTranslate free API (no API key needed)
            response = requests.post(
                'https://libretranslate.de/translate',
                json={
                    'q': text,
                    'source': source_lang,
                    'target': self.target_language,
                    'format': 'text'
                },
                headers={'Content-Type': 'application/json'},
                timeout=10  # 10 second timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                translated_text = result['translatedText']
                
                # Auto-add new translation to fallback for future use
                self._add_to_fallback(text_lower, translated_text)
                
                return translated_text, source_lang
            else:
                logging.error(f"Translation API error: {response.status_code}")
                return text, 'en'
                
        except requests.exceptions.Timeout:
            logging.error("Translation timeout")
            return text, 'en'
        except Exception as e:
            logging.error(f"Translation error: {e}")
            return text, 'en'
    
    def _add_to_fallback(self, english_word, translated_word):
        """Add a new translation to the fallback dictionary"""
        if english_word not in self.fallback_translations.get('en', {}):
            # Add to English first
            self.fallback_translations.setdefault('en', {})[english_word] = english_word
        
        # Add translation for current language
        self.fallback_translations.setdefault(self.target_language, {})[english_word] = translated_word
        
        # Save updated fallback translations
        self._save_fallback_translations()
        
        print(f"💾 Added new translation: '{english_word}' -> '{translated_word}' in {self.supported_languages[self.target_language]}")
    
    def detect_language(self, text):
        """Detect the language of the given text"""
        try:
            response = requests.post(
                'https://libretranslate.de/detect',
                json={'q': text},
                headers={'Content-Type': 'application/json'},
                timeout=5
            )
            if response.status_code == 200:
                result = response.json()
                if result and len(result) > 0:
                    return result[0]['language'], result[0]['confidence']
        except:
            pass
        return 'en', 0.0
    
    def get_available_words(self):
        """Get list of all available words in the fallback dictionary"""
        return list(self.fallback_translations.get('en', {}).keys())

# Global translation service instance
translation_service = TranslationService()