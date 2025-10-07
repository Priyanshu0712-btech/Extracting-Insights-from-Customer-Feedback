import nltk
import spacy
import re
import logging
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from nltk.corpus import stopwords, opinion_lexicon
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from collections import defaultdict, Counter
import json
import MySQLdb.cursors

# Initialize NLP tools
nltk.download("stopwords", quiet=True)
nltk.download("opinion_lexicon", quiet=True)
nltk.download("vader_lexicon", quiet=True)

stop_words = set(stopwords.words("english"))
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logging.warning("spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
    nlp = None

# Load sentiment lexicon
positive_words = set(opinion_lexicon.positive())
negative_words = set(opinion_lexicon.negative())

# Add specific words that should be highlighted
positive_words.update(['must', 'stood'])

# Negation words and phrases
NEGATION_WORDS = {
    'not', 'no', 'never', 'none', 'nothing', 'nowhere', 'neither', 'nor',
    'dont', "don't", 'doesnt', "doesn't", 'didnt', "didn't", 'cant', "can't",
    'cannot', 'wont', "won't", 'wouldnt', "wouldn't", 'shouldnt', "shouldn't",
    'couldnt', "couldn't", 'isnt', "isn't", 'arent', "aren't", 'wasnt', "wasn't",
    'werent', "weren't", 'hasnt', "hasn't", 'havent', "haven't", 'hadnt', "hadn't",
    'aint', "ain't", 'hardly', 'barely', 'scarcely', 'rarely', 'seldom'
}

# Intensifiers and diminishers
INTENSIFIERS = {'very', 'really', 'extremely', 'incredibly', 'absolutely', 'totally', 'completely', 'utterly', 'highly', 'so', 'quite', 'pretty', 'fairly', 'rather'}
DIMINISHERS = {'slightly', 'somewhat', 'kind of', 'sort of', 'a bit', 'a little', 'barely', 'hardly', 'scarcely'}

# Sarcasm indicators
SARCASM_INDICATORS = {
    'oh', 'sure', 'right', 'yeah', 'totally', 'absolutely', 'definitely', 'obviously',
    'clearly', 'apparently', 'evidently', 'obviously', 'naturally', 'of course',
    'as if', 'whatever', 'like', 'duh', 'wow', 'gee', 'gosh', 'oh boy', 'oh dear'
}

# Load sentiment models
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
MODEL_BERT = "nlptown/bert-base-multilingual-uncased-sentiment"
MODEL_IRONY = "cardiffnlp/twitter-roberta-base-irony"

# Initialize models (lazy loading to avoid issues)
_sentiment_analyzer = None
_bert_analyzer = None
_irony_analyzer = None
_tokenizer = None
_model = None

def detect_negation_scope(text: str) -> list:
    """Detect negation words and their scope in the text."""
    if nlp is None:
        return []
    doc = nlp(text)
    negations = []

    for i, token in enumerate(doc):
        if token.lemma_.lower() in NEGATION_WORDS:
            # Find the scope of negation (typically 3-5 words after negation)
            start_idx = i
            end_idx = min(i + 6, len(doc))  # Extended scope

            # Skip punctuation and conjunctions to find actual scope
            scope_tokens = []
            for j in range(start_idx + 1, end_idx):
                if not doc[j].is_punct and doc[j].text.lower() not in ['and', 'or', 'but', 'so', 'then']:
                    scope_tokens.append(doc[j])
                    if len(scope_tokens) >= 4:  # Extended scope to 4 words
                        break

            negations.append({
                'negation_word': token.text,
                'position': i,
                'scope_start': start_idx + 1,
                'scope_end': start_idx + 1 + len(scope_tokens),
                'scope_tokens': scope_tokens
            })

    return negations

def detect_sarcasm_indicators(text: str) -> dict:
    indicators_found = []
    if nlp is None:
        return {
            'indicators': [],
            'contradiction_score': 0.0,
            'has_sarcasm_potential': False
        }
    doc = nlp(text)

    # Check for sarcasm indicators
    for token in doc:
        if token.lemma_.lower() in SARCASM_INDICATORS:
            indicators_found.append(token.text)

    # Check for contradictory sentiment patterns
    positive_count = 0
    negative_count = 0
    contradiction_score = 0.0

    for token in doc:
        lemma = token.lemma_.lower()
        if lemma in positive_words:
            positive_count += 1
        elif lemma in negative_words:
            negative_count += 1

    # High contradiction score if both positive and negative words are present
    if positive_count > 0 and negative_count > 0:
        contradiction_score = min(positive_count, negative_count) / max(positive_count, negative_count)

    # Additional sarcasm patterns
    text_lower = text.lower()

    # Check for exaggerated politeness or obvious statements
    exaggerated_patterns = [
        r'oh.*so.*good', r'how.*wonderful', r'such.*amazing', r'too.*perfect',
        r'annoyingly.*good', r'frustratingly.*nice', r'irritatingly.*perfect',
        r'just.*fantastic', r'love.*it', r'great.*job', r'amazing.*work'
    ]

    for pattern in exaggerated_patterns:
        if re.search(pattern, text_lower):
            contradiction_score += 0.3

    # Check for mixed sentiments in the same sentence
    for sent in doc.sents:
        sent_text = sent.text.lower()
        sent_positive = sum(1 for token in sent if token.lemma_.lower() in positive_words)
        sent_negative = sum(1 for token in sent if token.lemma_.lower() in negative_words)
        if sent_positive > 0 and sent_negative > 0:
            contradiction_score += 0.2

    # Check for obvious positive words with negative context
    obvious_sarcasm_patterns = [
        r'fantastic.*not', r'wonderful.*but', r'amazing.*however',
        r'perfect.*except', r'excellent.*although', r'great.*unfortunately'
    ]

    for pattern in obvious_sarcasm_patterns:
        if re.search(pattern, text_lower):
            contradiction_score += 0.4
            indicators_found.append("obvious_sarcasm")

    return {
        'indicators': indicators_found,
        'contradiction_score': min(contradiction_score, 1.0),
        'has_sarcasm_potential': len(indicators_found) > 0 or contradiction_score > 0.3
    }

def lexicon_sentiment_with_negation(text: str) -> dict:
    """Perform lexicon-based sentiment analysis with negation handling."""
    if nlp is None:
        # Fallback: simple word-based analysis
        words = text.lower().split()
        positive_count = sum(1 for w in words if w in positive_words)
        negative_count = sum(1 for w in words if w in negative_words)
        if positive_count > negative_count:
            return {'sentiment': 'Positive', 'score': 0.5, 'confidence': 0.5, 'positive_words': [], 'negative_words': [], 'negated_words': [], 'negations_detected': 0}
        elif negative_count > positive_count:
            return {'sentiment': 'Negative', 'score': -0.5, 'confidence': 0.5, 'positive_words': [], 'negative_words': [], 'negated_words': [], 'negations_detected': 0}
        else:
            return {'sentiment': 'Neutral', 'score': 0.0, 'confidence': 0.5, 'positive_words': [], 'negative_words': [], 'negated_words': [], 'negations_detected': 0}
    doc = nlp(text)
    negations = detect_negation_scope(text)

    sentiment_score = 0.0
    positive_words_found = []
    negative_words_found = []
    negated_words = []

    for i, token in enumerate(doc):
        if token.is_punct or token.is_space:
            continue

        lemma = token.lemma_.lower()
        word_sentiment = 0.0

        # Check if word is in sentiment lexicons
        if lemma in positive_words:
            word_sentiment = 1.0
            positive_words_found.append(token.text)
        elif lemma in negative_words:
            word_sentiment = -1.0
            negative_words_found.append(token.text)

        # Apply intensifiers/diminishers
        if word_sentiment != 0.0:
            # Check previous words for intensifiers/diminishers
            for j in range(max(0, i-2), i):
                prev_lemma = doc[j].lemma_.lower()
                if prev_lemma in INTENSIFIERS:
                    word_sentiment *= 1.5
                    break
                elif prev_lemma in DIMINISHERS:
                    word_sentiment *= 0.5
                    break

        # Check if word is within negation scope
        is_negated = False
        for negation in negations:
            if negation['scope_start'] <= i < negation['scope_end']:
                is_negated = True
                negated_words.append(token.text)
                break

        if is_negated:
            word_sentiment *= -1  # Flip sentiment

        sentiment_score += word_sentiment

    # Normalize score
    total_sentiment_words = len(positive_words_found) + len(negative_words_found)
    if total_sentiment_words > 0:
        sentiment_score = sentiment_score / total_sentiment_words

    # Determine final sentiment
    if sentiment_score > 0.1:
        final_sentiment = "Positive"
    elif sentiment_score < -0.1:
        final_sentiment = "Negative"
    else:
        final_sentiment = "Neutral"

    return {
        'sentiment': final_sentiment,
        'score': sentiment_score,
        'confidence': min(abs(sentiment_score), 1.0),
        'positive_words': positive_words_found,
        'negative_words': negative_words_found,
        'negated_words': negated_words,
        'negations_detected': len(negations)
    }

def enhanced_sentiment_analysis(text: str) -> dict:
    """Enhanced sentiment analysis combining multiple approaches with negation and sarcasm handling."""
    if not text or len(text.strip()) < 3:
        return {
            'sentiment': 'Neutral',
            'confidence': 0.5,
            'methods_used': [],
            'negation_info': {},
            'sarcasm_info': {}
        }

    results = {}

    # Detect negation and sarcasm
    negation_info = detect_negation_scope(text)
    sarcasm_info = detect_sarcasm_indicators(text)

    # Lexicon-based analysis with negation
    lexicon_result = lexicon_sentiment_with_negation(text)
    results['lexicon'] = lexicon_result

    # Transformer-based analysis (existing)
    try:
        sentiment_analyzer = get_sentiment_analyzer()
        bert_analyzer = get_bert_analyzer()
        irony_analyzer = get_irony_analyzer()

        sent_roberta = sentiment_analyzer(text[:256])[0]
        sent_bert = bert_analyzer(text[:256])[0]
        irony_result = irony_analyzer(text[:256])[0]

        transformer_result = ensemble_sentiment(
            {'label': sent_roberta['label'], 'score': float(sent_roberta['score'])},
            {'label': sent_bert['label'], 'score': float(sent_bert['score'])},
            irony_result["label"], float(irony_result["score"])
        )
        results['transformer'] = {
            'sentiment': transformer_result[0],
            'confidence': transformer_result[1]
        }
    except Exception as e:
        logging.warning(f"Transformer analysis failed: {e}")
        results['transformer'] = {'sentiment': 'Neutral', 'confidence': 0.3}

    # Weighted ensemble with negation/sarcasm adjustments
    weights = {
        'lexicon': 0.4,  # Higher weight for lexicon with negation handling
        'transformer': 0.6
    }

    # Convert sentiments to scores
    sentiment_to_score = {'Positive': 1.0, 'Neutral': 0.0, 'Negative': -1.0}

    lexicon_score = sentiment_to_score.get(results['lexicon']['sentiment'], 0.0)
    transformer_score = sentiment_to_score.get(results['transformer']['sentiment'], 0.0)

    # Check for mixed sentiments (both positive and negative words present)
    has_mixed_sentiments = (
        len(results['lexicon']['positive_words']) > 0 and
        len(results['lexicon']['negative_words']) > 0
    )

    # If mixed sentiments detected, adjust weights to favor lexicon analysis
    if has_mixed_sentiments:
        weights['lexicon'] = 0.6
        weights['transformer'] = 0.4

    # Apply sarcasm adjustment
    if sarcasm_info['has_sarcasm_potential']:
        # Reduce confidence when sarcasm is detected
        weights['lexicon'] *= 0.8
        weights['transformer'] *= 0.8

        # If high contradiction, significantly flip the sentiment
        if sarcasm_info['contradiction_score'] > 0.6:
            lexicon_score *= -0.7  # Stronger flip for high contradiction
            transformer_score *= -0.7

    # Calculate weighted score
    final_score = (
        lexicon_score * weights['lexicon'] +
        transformer_score * weights['transformer']
    ) / sum(weights.values())

    # Determine final sentiment with adjusted thresholds for mixed sentiments
    if has_mixed_sentiments:
        # More conservative thresholds for mixed sentiments
        if final_score > 0.3:
            final_sentiment = "Positive"
        elif final_score < -0.3:
            final_sentiment = "Negative"
        else:
            final_sentiment = "Neutral"
    else:
        # Standard thresholds
        if final_score > 0.2:
            final_sentiment = "Positive"
        elif final_score < -0.2:
            final_sentiment = "Negative"
        else:
            final_sentiment = "Neutral"

    # Calculate confidence based on agreement and negation/sarcasm factors
    agreement_bonus = 1.0 if results['lexicon']['sentiment'] == results['transformer']['sentiment'] else 0.7
    negation_penalty = 0.9 if negation_info else 1.0
    sarcasm_penalty = 0.8 if sarcasm_info['has_sarcasm_potential'] else 1.0

    final_confidence = min(
        (abs(final_score) + results['lexicon']['confidence'] + results['transformer']['confidence']) / 3.0 *
        agreement_bonus * negation_penalty * sarcasm_penalty,
        1.0
    )

    return {
        'sentiment': final_sentiment,
        'confidence': final_confidence,
        'score': final_score,
        'methods_used': list(results.keys()),
        'negation_info': {
            'negations_detected': len(negation_info),
            'negated_words': results['lexicon'].get('negated_words', [])
        },
        'sarcasm_info': {
            'indicators_found': sarcasm_info['indicators'],
            'contradiction_score': sarcasm_info['contradiction_score'],
            'has_sarcasm_potential': sarcasm_info['has_sarcasm_potential']
        },
        'component_results': results
    }

def get_sentiment_analyzer():
    global _sentiment_analyzer
    if _sentiment_analyzer is None:
        try:
            _sentiment_analyzer = pipeline("sentiment-analysis",
                                        model=MODEL_NAME,
                                        tokenizer=MODEL_NAME,
                                        device=-1)
        except (OSError, ValueError) as e:
            logging.warning(f"Failed to load sentiment analyzer: {e}")
            _sentiment_analyzer = None
    return _sentiment_analyzer

def get_bert_analyzer():
    global _bert_analyzer
    if _bert_analyzer is None:
        try:
            _bert_analyzer = pipeline("sentiment-analysis", model=MODEL_BERT, device=-1)
        except (OSError, ValueError) as e:
            logging.warning(f"Failed to load BERT analyzer: {e}")
            _bert_analyzer = None
    return _bert_analyzer

def get_irony_analyzer():
    global _irony_analyzer
    if _irony_analyzer is None:
        try:
            _irony_analyzer = pipeline("text-classification", model=MODEL_IRONY, device=-1)
        except (OSError, ValueError) as e:
            logging.warning(f"Failed to load irony analyzer: {e}")
            _irony_analyzer = None
    return _irony_analyzer

# Sentiment mapping helper with irony adjustment
def map_sentiment(label, irony_label=None, irony_score=0.0):
    """Map sentiment labels and adjust for irony/sarcasm."""
    base_sentiment = "Neutral"
    if str(label).lower() in ["label_2", "positive", "5 star"]:
        base_sentiment = "Positive"
    elif str(label).lower() in ["label_1", "neutral", "4 star", "3 star"]:
        base_sentiment = "Neutral"
    elif str(label).lower() in ["label_0", "negative", "1 star", "2 star"]:
        base_sentiment = "Negative"

    # Adjust for irony/sarcasm
    if irony_label == "LABEL_1" and irony_score > 0.5:
        if base_sentiment == "Positive":
            return "Negative"
        elif base_sentiment == "Negative":
            return "Positive"
    return base_sentiment


def ensemble_sentiment(sent_roberta, sent_bert, irony_label=None, irony_score=0.0):
    """Combine sentiments from RoBERTa and BERT models."""
    # Map to scores: Positive=1, Neutral=0, Negative=-1
    def label_to_score(label):
        if label == "Positive":
            return 1.0
        elif label == "Neutral":
            return 0.0
        elif label == "Negative":
            return -1.0
        return 0.0

    roberta_score = label_to_score(map_sentiment(sent_roberta['label']))
    bert_score = label_to_score(map_sentiment(sent_bert['label']))

    # Weighted average (equal weights for simplicity)
    combined_score = (roberta_score + bert_score) / 2

    # Map back to sentiment
    if combined_score > 0.3:
        final_sentiment = "Positive"
    elif combined_score < -0.3:
        final_sentiment = "Negative"
    else:
        final_sentiment = "Neutral"

    # Adjust for irony
    if irony_label == "LABEL_1" and irony_score > 0.5:
        if final_sentiment == "Positive":
            final_sentiment = "Negative"
        elif final_sentiment == "Negative":
            final_sentiment = "Positive"

    # Combined confidence as average
    combined_confidence = (sent_roberta['score'] + sent_bert['score']) / 2

    return final_sentiment, combined_confidence

# Text preprocessing
def preprocess_text(text: str) -> str:
    """Clean raw review text before sentiment analysis."""
    if not text:
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs and HTML tags
    text = re.sub(r"http\S+|www\S+|<.*?>", " ", text)

    # Remove special characters / digits (keep words)
    text = re.sub(r"[^a-z\s]", " ", text)

    # Tokenize with spaCy
    if nlp is None:
        # Fallback: simple tokenization
        tokens = text.split()
        clean_tokens = [t for t in tokens if t not in stop_words]
        return " ".join(clean_tokens).strip()
    doc = nlp(text)

    # Remove stopwords + lemmatize
    clean_tokens = [
        token.lemma_ for token in doc
        if token.is_alpha and token.text not in stop_words
    ]

    return " ".join(clean_tokens).strip()

# Aspect extraction using spaCy NER and dependency parsing
def extract_aspects(text: str):
    """Extract key aspects from review text using NLP."""
    if not text:
        return []

    doc = nlp(text)

    aspects = []
    aspect_candidates = []

    def clean_aspect(aspect):
        """Remove leading determiners from aspects."""
        words = aspect.split()
        if words and words[0].lower() in ['the', 'a', 'an', 'this', 'that', 'these', 'those']:
            return ' '.join(words[1:])
        return aspect

    # Extract noun phrases and named entities
    for chunk in doc.noun_chunks:
        aspect_text = clean_aspect(chunk.text.strip())
        if len(aspect_text.split()) <= 3 and len(aspect_text) > 2:
            aspect_candidates.append(aspect_text)

    # Add named entities that might be aspects
    for ent in doc.ents:
        if ent.label_ in ['PRODUCT', 'ORG'] and len(ent.text.split()) <= 3:
            aspect_text = clean_aspect(ent.text.strip())
            aspect_candidates.append(aspect_text)

    # Common aspect keywords for product reviews
    aspect_keywords = {
        'battery', 'camera', 'screen', 'display', 'performance', 'speed',
        'quality', 'price', 'value', 'service', 'support', 'delivery',
        'design', 'build', 'sound', 'speaker', 'microphone', 'charging',
        'storage', 'memory', 'processor', 'cpu', 'gpu', 'graphics',
        'connectivity', 'wifi', 'bluetooth', 'ports', 'durability',
        'reliability', 'usability', 'interface', 'ui', 'ux', 'app',
        'software', 'update', 'compatibility', 'warranty', 'packaging'
    }

    # Filter and prioritize aspects
    for candidate in aspect_candidates:
        candidate_lower = candidate.lower()
        # Direct keyword match
        if any(keyword in candidate_lower for keyword in aspect_keywords):
            aspects.append(candidate)
        # Length and position based filtering
        elif len(candidate.split()) == 1 and len(candidate) > 3:
            aspects.append(candidate)
        elif len(candidate.split()) == 2 and not any(word in stop_words for word in candidate_lower.split()):
            aspects.append(candidate)

    # Remove duplicates and return top aspects
    unique_aspects = list(set(aspects))
    return unique_aspects[:10]  # Limit to top 10 aspects

# Analyze sentiment for specific aspects (optimized)
def analyze_aspect_sentiment(text: str, aspects: list, max_aspects: int = 5):
    """Analyze sentiment for each extracted aspect with performance optimizations."""
    logger = logging.getLogger(__name__)
    if not text or not aspects:
        return {}

    # Limit number of aspects to prevent excessive processing
    aspects = aspects[:max_aspects]

    doc = nlp(text)
    aspect_sentiments = {}

    # Get analyzers once to avoid repeated initialization
    sentiment_analyzer = get_sentiment_analyzer()
    bert_analyzer = get_bert_analyzer()
    irony_analyzer = get_irony_analyzer()

    for aspect in aspects:
        # Find sentences containing the aspect
        aspect_sentences = []
        for sent in doc.sents:
            if aspect.lower() in sent.text.lower():
                aspect_sentences.append(sent.text.strip())

        if aspect_sentences:
            # Analyze sentiment of aspect-related sentences
            combined_text = " ".join(aspect_sentences)
            if len(combined_text) > 10:  # Only analyze if substantial text
                try:
                    # Check if analyzers are available
                    if sentiment_analyzer is not None and bert_analyzer is not None and irony_analyzer is not None:
                        # Use shorter text for faster processing
                        analysis_text = combined_text[:256]  # Reduced from 512

                        sent_roberta = sentiment_analyzer(analysis_text)[0]
                        sent_bert = bert_analyzer(analysis_text)[0]
                        irony_result = irony_analyzer(analysis_text)[0]
                        irony_label, irony_score = irony_result["label"], float(irony_result["score"])

                        final_sentiment, combined_confidence = ensemble_sentiment(
                            {'label': sent_roberta['label'], 'score': float(sent_roberta['score'])},
                            {'label': sent_bert['label'], 'score': float(sent_bert['score'])},
                            irony_label, irony_score
                        )
                        aspect_sentiments[aspect] = {
                            'sentiment': final_sentiment,
                            'confidence': combined_confidence,
                            'irony_score': irony_score,
                            'sentences': aspect_sentences[:2]  # Limit sentences to prevent memory issues
                        }
                    else:
                        # Fallback to lexicon-based analysis if models failed to load
                        lexicon_result = lexicon_sentiment_with_negation(combined_text)
                        aspect_sentiments[aspect] = {
                            'sentiment': lexicon_result['sentiment'],
                            'confidence': lexicon_result['confidence'],
                            'irony_score': 0.0,
                            'sentences': aspect_sentences[:2]
                        }
                except Exception as e:
                    logger.error(f"Error analyzing sentiment for aspect '{aspect}': {e}")
                    # Fallback to lexicon-based analysis
                    lexicon_result = lexicon_sentiment_with_negation(combined_text)
                    aspect_sentiments[aspect] = {
                        'sentiment': lexicon_result['sentiment'],
                        'confidence': lexicon_result['confidence'],
                        'irony_score': 0.0,
                        'sentences': aspect_sentences[:2]
                    }
        else:
            aspect_sentiments[aspect] = {
                'sentiment': 'Neutral',
                'confidence': 0.3,
                'irony_score': 0.0,
                'sentences': []
            }

    return aspect_sentiments

# Generate comprehensive analysis summary
def generate_analysis_summary(aspect_sentiments: dict, overall_sentiment: str = None, overall_confidence: float = 0.0):
    """Generate summary statistics for the analysis."""
    if not aspect_sentiments:
        return {
            'total_aspects': 0,
            'positive_count': 0,
            'negative_count': 0,
            'neutral_count': 0,
            'overall_sentiment': overall_sentiment or 'Neutral',
            'overall_confidence': overall_confidence,
            'aspect_distribution': {}
        }

    sentiment_counts = Counter(aspect['sentiment'] for aspect in aspect_sentiments.values())

    return {
        'total_aspects': len(aspect_sentiments),
        'positive_count': sentiment_counts.get('Positive', 0),
        'negative_count': sentiment_counts.get('Negative', 0),
        'neutral_count': sentiment_counts.get('Neutral', 0),
        'overall_sentiment': overall_sentiment or 'Neutral',
        'overall_confidence': overall_confidence,
        'aspect_distribution': dict(sentiment_counts)
    }

# Enhanced keyword highlighting for aspects
def highlight_aspects(text: str, aspect_sentiments: dict):
    """Highlight aspect terms in the original text."""
    if not text or not aspect_sentiments:
        return text

    # Create a copy of the text for highlighting
    highlighted_text = text

    # Sort aspects by length (longest first) to avoid partial replacements
    sorted_aspects = sorted(aspect_sentiments.keys(), key=len, reverse=True)

    for aspect in sorted_aspects:
        sentiment_info = aspect_sentiments[aspect]
        color = {
            'Positive': 'lightgreen',
            'Negative': 'lightcoral',
            'Neutral': 'lightyellow'
        }.get(sentiment_info['sentiment'], 'lightgray')

        # Use word boundaries to avoid partial replacements
        pattern = r'\b' + re.escape(aspect) + r'\b'
        replacement = f'<span style="background-color: {color}; padding: 2px 4px; border-radius: 3px;">{aspect}</span>'
        highlighted_text = re.sub(pattern, replacement, highlighted_text, flags=re.IGNORECASE)

    return highlighted_text

# Keyword highlighting helper - Highlight sentiment-bearing words from opinion lexicon
def highlight_keywords(text, sentiment=None):
    """Highlight sentiment-bearing words from opinion lexicon in the text with appropriate colors."""
    if not text:
        return text

    try:
        # Process text with spaCy for tokenization
        doc = nlp(text)

        # Create a list of (start, end, replacement) tuples for replacements
        replacements = []
        for token in doc:
            if token.is_punct or token.is_space:
                continue

            lemma = token.lemma_.lower()
            # Only highlight if lemma is a whole word in lexicon (avoid partial matches)
            # Fix: Also check if token.text.lower() matches lemma to avoid missing highlights
            if lemma in positive_words and token.text.lower() == lemma:
                color = "lightgreen"
                replacement = f'<span style="background-color: {color}; padding: 2px 4px; border-radius: 3px;">{token.text}</span>'
                replacements.append((token.idx, token.idx + len(token.text), replacement))
            elif lemma in negative_words and token.text.lower() == lemma:
                color = "lightcoral"
                replacement = f'<span style="background-color: {color}; padding: 2px 4px; border-radius: 3px;">{token.text}</span>'
                replacements.append((token.idx, token.idx + len(token.text), replacement))

        # Sort replacements by start position descending to replace from end to beginning
        replacements.sort(key=lambda x: x[0], reverse=True)

        # Apply replacements to the original text
        highlighted_text = text
        for start, end, repl in replacements:
            highlighted_text = highlighted_text[:start] + repl + highlighted_text[end:]

        return highlighted_text
    except Exception as e:
        logging.warning(f"Failed to highlight keywords: {e}")
        return text

# Simple cache for analysis results to prevent repeated processing
_analysis_cache = {}

def clear_analysis_cache():
    """Clear the analysis cache to free memory."""
    global _analysis_cache
    _analysis_cache.clear()

def analyze_review_detailed(review_text: str, overall_sentiment: str = None, overall_confidence: float = 0.0, mysql=None):
    """Perform comprehensive analysis of a review with caching in DB if mysql connection provided."""
    if mysql is None:
        # Fallback to existing in-memory cache if no mysql connection provided
        cache_key = hash(review_text.strip().lower())
        if cache_key in _analysis_cache:
            cached_result = _analysis_cache[cache_key].copy()
            if overall_sentiment:
                cached_result['summary']['overall_sentiment'] = overall_sentiment
                cached_result['summary']['overall_confidence'] = overall_confidence
            return cached_result

        clean_text = preprocess_text(review_text)
        aspects = extract_aspects(review_text)
        aspect_sentiments = analyze_aspect_sentiment(review_text, aspects, max_aspects=10)
        summary = generate_analysis_summary(aspect_sentiments, overall_sentiment, overall_confidence)
        highlighted_text = highlight_aspects(review_text, aspect_sentiments)

        result = {
            'original_text': review_text,
            'clean_text': clean_text,
            'highlighted_text': highlighted_text,
            'aspects': aspects,
            'aspect_sentiments': aspect_sentiments,
            'summary': summary
        }

        if len(_analysis_cache) < 100:
            _analysis_cache[cache_key] = result.copy()

        return result

    # Use persistent cache in DB
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    try:
        # Check if cached result exists
        cursor.execute("SELECT aspect_sentiments FROM review_aspect_sentiments WHERE review_id = (SELECT review_id FROM reviews WHERE review_text = %s LIMIT 1)", (review_text,))
        row = cursor.fetchone()
        if row:
            aspect_sentiments = json.loads(row['aspect_sentiments'])
            summary = generate_analysis_summary(aspect_sentiments, overall_sentiment, overall_confidence)
            highlighted_text = highlight_aspects(review_text, aspect_sentiments)
            result = {
                'original_text': review_text,
                'clean_text': preprocess_text(review_text),
                'highlighted_text': highlighted_text,
                'aspects': list(aspect_sentiments.keys()),
                'aspect_sentiments': aspect_sentiments,
                'summary': summary
            }
            return result
    except Exception as e:
        logging.error(f"Error fetching cached aspect sentiments: {e}")

    # If no cache, perform analysis
    clean_text = preprocess_text(review_text)
    aspects = extract_aspects(review_text)
    aspect_sentiments = analyze_aspect_sentiment(review_text, aspects, max_aspects=10)
    summary = generate_analysis_summary(aspect_sentiments, overall_sentiment, overall_confidence)
    highlighted_text = highlight_aspects(review_text, aspect_sentiments)

    result = {
        'original_text': review_text,
        'clean_text': clean_text,
        'highlighted_text': highlighted_text,
        'aspects': aspects,
        'aspect_sentiments': aspect_sentiments,
        'summary': summary
    }

    # Save to DB cache
    try:
        cursor.execute("SELECT review_id FROM reviews WHERE review_text = %s LIMIT 1", (review_text,))
        review_row = cursor.fetchone()
        if review_row:
            review_id = review_row['review_id']
            aspect_sentiments_json = json.dumps(aspect_sentiments)
            cursor.execute("""
                INSERT INTO review_aspect_sentiments (review_id, aspect_sentiments)
                VALUES (%s, %s)
                ON DUPLICATE KEY UPDATE aspect_sentiments = VALUES(aspect_sentiments), cached_at = CURRENT_TIMESTAMP
            """, (review_id, aspect_sentiments_json))
            mysql.connection.commit()
    except Exception as e:
        logging.error(f"Error saving aspect sentiments cache: {e}")
    finally:
        cursor.close()

    return result