# ======================================================================
# ██████  ███████ ██████  ██████  ███████  ██████  █████  ████████ ███████ ██████  
# ██   ██ ██      ██   ██ ██   ██ ██      ██      ██   ██    ██    ██      ██   ██ 
# ██   ██ █████   ██████  ██████  █████   ██      ███████    ██    █████   ██   ██ 
# ██   ██ ██      ██      ██   ██ ██      ██      ██   ██    ██    ██      ██   ██ 
# ██████  ███████ ██      ██   ██ ███████  ██████ ██   ██    ██    ███████ ██████  
#
# THIS MODULE IS DISABLED -- DO NOT IMPORT
#
# This module was part of the "Query Intelligence" feature (commit f4d8c36)
# which has been reverted. All code below is commented out and inactive.
#
# If you need to re-enable this feature, remove the triple-quote wrapping
# below and restore the imports in chatbot.py, retrieval.py, and prompts.py.
# ======================================================================

'''
"""Query Intelligence Module - Unified query understanding for smart retrieval and response generation"""
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


class QueryType(Enum):
    """Types of queries the chatbot can handle"""
    FACTUAL_LOOKUP = "factual_lookup"  # Simple facts (lecturer email, office hours, tuition amount)
    EXPLORATORY = "exploratory"  # Broad questions needing comprehensive answers
    COMPARATIVE = "comparative"  # Comparing options (program A vs B)
    PROCEDURAL = "procedural"  # How-to questions
    ELIGIBILITY = "eligibility"  # Can I / Am I eligible questions
    TEMPORAL = "temporal"  # Deadline and date-related questions


class UserProfile(Enum):
    """Inferred user profiles based on query signals"""
    LOCAL_DSE = "local_dse"  # Hong Kong DSE student applying via JUPAS
    LOCAL_NON_JUPAS = "local_non_jupas"  # Local student applying via non-JUPAS
    INTERNATIONAL = "international"  # International student
    TRANSFER = "transfer"  # Transfer student from sub-degree/HD
    CURRENT_STUDENT = "current_student"  # Already enrolled student
    PROSPECTIVE = "prospective"  # General prospective student
    UNKNOWN = "unknown"


@dataclass
class QueryClassification:
    """Complete classification of a user query"""
    query_type: QueryType
    user_profile: UserProfile
    intents: List[str]
    implicit_needs: List[str]
    confidence: float
    expanded_queries: List[str] = field(default_factory=list)
    context_boost_terms: List[str] = field(default_factory=list)
    response_guidelines: Dict[str, Any] = field(default_factory=dict)


class QueryClassifier:
    """Classifies queries into types for appropriate handling"""
    
    def __init__(self):
        # Patterns for each query type
        self.type_patterns = {
            QueryType.FACTUAL_LOOKUP: [
                r'\\b(email|phone|contact|office|room|location)\\b',
                r'\\b(what is|what\'s)\\s+(the|a)?\\s*(email|phone|address|name|title)',
                r'\\bwho\\s+is\\b',
                r'\\b(cost|price|fee|tuition)\\s+(of|for)\\b',
                r'\\bhow\\s+much\\s+(is|does|are)\\b',
            ],
            QueryType.EXPLORATORY: [
                r'\\b(tell me about|what about|overview|explain|describe)\\b',
                r'\\b(acceptance rate|admission rate|competitiveness)\\b',
                r'\\bhow\\s+(does|is|are)\\s+.*\\b(work|like|good|popular)\\b',
                r'\\bwhat\\s+(are|is)\\s+(the\\s+)?(benefits|advantages|features|characteristics)\\b',
                r'\\b(career|job|employment)\\s+(prospects|opportunities|outlook)\\b',
            ],
            QueryType.COMPARATIVE: [
                r'\\b(compare|comparison|vs|versus|difference|between|or)\\b',
                r'\\b(better|worse|prefer|recommend)\\b',
                r'\\bwhich\\s+(one|program|course|option)\\b',
                r'\\b(a|b|option\\s*1|option\\s*2)\\s+(vs|or|versus)\\b',
            ],
            QueryType.PROCEDURAL: [
                r'\\bhow\\s+(to|do|can|should)\\s+\\w+\\b',
                r'\\bsteps?\\s+(to|for)\\b',
                r'\\bprocess\\s+(of|for|to)\\b',
                r'\\b(apply|application|register|enroll)\\b',
                r'\\bwhat\\s+(do|should)\\s+i\\s+(do|need)\\b',
            ],
            QueryType.ELIGIBILITY: [
                r'\\b(can|could|am)\\s+i\\b',
                r'\\b(eligible|eligibility|qualify|qualification)\\b',
                r'\\b(requirement|requirements|needed|need)\\b',
                r'\\b(minimum|required)\\s+(grade|gpa|score|marks)\\b',
                r'\\bdo\\s+i\\s+(need|have|qualify)\\b',
            ],
            QueryType.TEMPORAL: [
                r'\\b(deadline|due\\s+date|by\\s+when|when\\s+is)\\b',
                r'\\b(start|end|open|close|begins?|ends?)\\s+(date|time)\\b',
                r'\\bwhen\\s+(does|do|is|are|can)\\b',
                r'\\b(january|february|march|april|may|june|july|august|september|october|november|december)\\b',
                r'\\b(semester|term|academic\\s+year)\\b.*\\b(start|end|begin)\\b',
            ],
        }
        
        # Keywords that indicate specific query types (weighted)
        self.type_keywords = {
            QueryType.FACTUAL_LOOKUP: {
                'high': ['email', 'phone', 'contact', 'address', 'room', 'office', 'name'],
                'medium': ['who', 'where', 'what is', 'cost', 'fee', 'tuition'],
            },
            QueryType.EXPLORATORY: {
                'high': ['overview', 'explain', 'describe', 'acceptance rate', 'career prospects'],
                'medium': ['about', 'like', 'popular', 'good', 'program information'],
            },
            QueryType.COMPARATIVE: {
                'high': ['compare', 'vs', 'versus', 'difference', 'better'],
                'medium': ['or', 'which', 'recommend', 'prefer'],
            },
            QueryType.PROCEDURAL: {
                'high': ['how to', 'steps', 'process', 'guide', 'tutorial'],
                'medium': ['apply', 'register', 'enroll', 'submit'],
            },
            QueryType.ELIGIBILITY: {
                'high': ['eligible', 'qualify', 'requirement', 'can i', 'minimum'],
                'medium': ['need', 'required', 'criteria', 'prerequisite'],
            },
            QueryType.TEMPORAL: {
                'high': ['deadline', 'due date', 'when', 'date', 'by when'],
                'medium': ['start', 'end', 'open', 'close', 'semester'],
            },
        }

    def classify(self, query: str) -> QueryType:
        """Classify the query into a type"""
        query_lower = query.lower()
        scores = {qt: 0.0 for qt in QueryType}
        
        # Score based on pattern matches
        for query_type, patterns in self.type_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    scores[query_type] += 2.0
        
        # Score based on keyword matches
        for query_type, keywords in self.type_keywords.items():
            for keyword in keywords.get('high', []):
                if keyword in query_lower:
                    scores[query_type] += 3.0
            for keyword in keywords.get('medium', []):
                if keyword in query_lower:
                    scores[query_type] += 1.5
        
        # Get highest scoring type
        max_score = max(scores.values())
        if max_score > 0:
            return max(scores.keys(), key=lambda k: scores[k])
        
        # Default to exploratory for general questions
        return QueryType.EXPLORATORY

    def get_confidence(self, query: str, query_type: QueryType) -> float:
        """Calculate confidence score for the classification"""
        query_lower = query.lower()
        score = 0.0
        max_possible = 0.0
        
        # Check patterns
        patterns = self.type_patterns.get(query_type, [])
        for pattern in patterns:
            max_possible += 2.0
            if re.search(pattern, query_lower):
                score += 2.0
        
        # Check keywords
        keywords = self.type_keywords.get(query_type, {})
        for keyword in keywords.get('high', []):
            max_possible += 3.0
            if keyword in query_lower:
                score += 3.0
        for keyword in keywords.get('medium', []):
            max_possible += 1.5
            if keyword in query_lower:
                score += 1.5
        
        if max_possible == 0:
            return 0.5
        
        return min(1.0, score / max_possible + 0.3)  # Base confidence of 0.3


class UserContextInferencer:
    """Infers user profile from query signals"""
    
    def __init__(self):
        # Geographic/exam system signals
        self.local_signals = {
            'high': ['dse', 'jupas', 'hkdse', 'hong kong', 'hk', 'local'],
            'medium': ['band a', 'band b', 'band c', 'band d', 'band e', '332', '333', '3322'],
        }
        
        self.international_signals = {
            'high': ['international', 'overseas', 'foreign', 'non-local'],
            'medium': ['ib', 'a-level', 'a level', 'sat', 'act', 'gcse', 'ielts', 'toefl', 'visa', 'student visa'],
        }
        
        self.transfer_signals = {
            'high': ['transfer', 'sub-degree', 'hd', 'higher diploma', 'associate degree', 'asso'],
            'medium': ['articulation', 'credit transfer', 'year 3', 'senior year', 'top-up'],
        }
        
        self.current_student_signals = {
            'high': ['i am a student', 'current student', 'enrolled', 'my course', 'my program'],
            'medium': ['year 1', 'year 2', 'year 3', 'year 4', 'semester', 'my gpa', 'my grade'],
        }
        
        self.non_jupas_signals = {
            'high': ['non-jupas', 'nonjupas', 'non jupas', 'direct application'],
            'medium': ['mature student', 'working adult', 'part-time to full-time'],
        }

    def infer(self, query: str, conversation_history: Optional[List[Dict]] = None) -> UserProfile:
        """Infer user profile from query and conversation history"""
        query_lower = query.lower()
        
        # Build combined text from query and history
        combined_text = query_lower
        if conversation_history:
            for exchange in conversation_history[-5:]:  # Look at last 5 exchanges
                combined_text += " " + exchange.get('query', '').lower()
        
        # Score each profile
        scores = {
            UserProfile.LOCAL_DSE: 0,
            UserProfile.LOCAL_NON_JUPAS: 0,
            UserProfile.INTERNATIONAL: 0,
            UserProfile.TRANSFER: 0,
            UserProfile.CURRENT_STUDENT: 0,
        }
        
        # Check local DSE signals
        for signal in self.local_signals['high']:
            if signal in combined_text:
                scores[UserProfile.LOCAL_DSE] += 3
        for signal in self.local_signals['medium']:
            if signal in combined_text:
                scores[UserProfile.LOCAL_DSE] += 1.5
        
        # Check non-JUPAS signals
        for signal in self.non_jupas_signals['high']:
            if signal in combined_text:
                scores[UserProfile.LOCAL_NON_JUPAS] += 3
        for signal in self.non_jupas_signals['medium']:
            if signal in combined_text:
                scores[UserProfile.LOCAL_NON_JUPAS] += 1.5
        
        # Check international signals
        for signal in self.international_signals['high']:
            if signal in combined_text:
                scores[UserProfile.INTERNATIONAL] += 3
        for signal in self.international_signals['medium']:
            if signal in combined_text:
                scores[UserProfile.INTERNATIONAL] += 1.5
        
        # Check transfer signals
        for signal in self.transfer_signals['high']:
            if signal in combined_text:
                scores[UserProfile.TRANSFER] += 3
        for signal in self.transfer_signals['medium']:
            if signal in combined_text:
                scores[UserProfile.TRANSFER] += 1.5
        
        # Check current student signals
        for signal in self.current_student_signals['high']:
            if signal in combined_text:
                scores[UserProfile.CURRENT_STUDENT] += 3
        for signal in self.current_student_signals['medium']:
            if signal in combined_text:
                scores[UserProfile.CURRENT_STUDENT] += 1.5
        
        # Get highest scoring profile
        max_score = max(scores.values())
        if max_score >= 2:
            return max(scores.keys(), key=lambda k: scores[k])
        
        # Default to prospective
        return UserProfile.PROSPECTIVE

    def get_context_boost_terms(self, user_profile: UserProfile) -> List[str]:
        """Get terms to boost in retrieval based on user profile"""
        boost_terms = {
            UserProfile.LOCAL_DSE: ['jupas', 'dse', 'hkdse', 'band', 'local student', 'hong kong'],
            UserProfile.LOCAL_NON_JUPAS: ['non-jupas', 'direct application', 'local'],
            UserProfile.INTERNATIONAL: ['international', 'non-local', 'overseas', 'visa', 'ielts', 'english requirement'],
            UserProfile.TRANSFER: ['transfer', 'credit transfer', 'articulation', 'advanced standing', 'sub-degree', 'senior year'],
            UserProfile.CURRENT_STUDENT: ['current student', 'enrolled', 'registration', 'course selection'],
            UserProfile.PROSPECTIVE: ['admission', 'application', 'requirement'],
            UserProfile.UNKNOWN: [],
        }
        return boost_terms.get(user_profile, [])


class IntentDetector:
    """Detects user intents and implicit information needs"""
    
    def __init__(self):
        # Maps vague terms to concrete needs
        self.intent_mappings = {
            'acceptance rate': ['quota', 'places', 'applicants', 'competitiveness', 'admission statistics', 'band distribution'],
            'program overview': ['curriculum', 'courses', 'modules', 'career prospects', 'learning outcomes'],
            'tuition': ['fee', 'sssdp', 'subsidy', 'scholarship', 'payment', 'installment'],
            'apply': ['application', 'deadline', 'requirements', 'documents', 'process'],
            'scholarship': ['eligibility', 'deadline', 'amount', 'application process', 'criteria'],
            'requirements': ['minimum grade', 'prerequisite', 'english requirement', 'subject requirement'],
            'career': ['employment', 'job prospects', 'industry', 'salary', 'alumni'],
            'accommodation': ['hall', 'hostel', 'residence', 'housing', 'dormitory'],
        }
        
        # Implicit needs based on query type
        self.implicit_needs_by_type = {
            QueryType.FACTUAL_LOOKUP: [],
            QueryType.EXPLORATORY: ['related programs', 'comparison with alternatives', 'key highlights'],
            QueryType.COMPARATIVE: ['pros and cons', 'key differences', 'recommendation'],
            QueryType.PROCEDURAL: ['timeline', 'required documents', 'common mistakes', 'tips'],
            QueryType.ELIGIBILITY: ['alternative pathways', 'exceptions', 'appeal process'],
            QueryType.TEMPORAL: ['related deadlines', 'consequences of missing', 'extension policy'],
        }
        
        # Topic-specific implicit needs
        self.topic_implicit_needs = {
            'tuition': ['subsidy', 'scholarship', 'payment plan', 'financial aid'],
            'program': ['admission requirements', 'career prospects', 'curriculum'],
            'deadline': ['related deadlines', 'late application', 'consequences'],
            'requirements': ['alternative pathways', 'exceptions', 'credit exemption'],
            'application': ['required documents', 'tips', 'timeline', 'interview'],
        }

    def detect_intents(self, query: str, query_type: QueryType) -> List[str]:
        """Detect explicit and implicit intents from query"""
        query_lower = query.lower()
        intents = []
        
        # Check intent mappings
        for trigger, related_intents in self.intent_mappings.items():
            if trigger in query_lower:
                intents.extend(related_intents)
        
        # Add implicit needs based on query type
        intents.extend(self.implicit_needs_by_type.get(query_type, []))
        
        # Add topic-specific implicit needs
        for topic, needs in self.topic_implicit_needs.items():
            if topic in query_lower:
                intents.extend(needs)
        
        return list(set(intents))  # Remove duplicates

    def get_expanded_queries(self, query: str, intents: List[str]) -> List[str]:
        """Generate expanded queries based on intents"""
        expanded = [query]
        query_lower = query.lower()
        
        # For each detected intent, create a related query
        for intent in intents[:5]:  # Limit to top 5 intents
            # Create contextual query combining original topic with intent
            # Extract key topic from query
            topic_words = [w for w in query_lower.split() if len(w) > 3 and w not in 
                         ['what', 'when', 'where', 'which', 'about', 'tell', 'does', 'have', 'with']]
            if topic_words:
                topic = ' '.join(topic_words[:2])
                expanded.append(f"{topic} {intent}")
            else:
                expanded.append(intent)
        
        return list(set(expanded))

    def get_proactive_info_rules(self, query_type: QueryType, intents: List[str]) -> List[str]:
        """Get rules for what proactive information to include"""
        rules = []
        
        if query_type == QueryType.TEMPORAL:
            rules.extend([
                "Include all related deadlines",
                "Mention what happens if deadline is missed",
                "State current date relative to deadline"
            ])
        elif query_type == QueryType.ELIGIBILITY:
            rules.extend([
                "Include alternative pathways if requirements not met",
                "Mention any exceptions or special cases",
                "Provide next steps for eligible candidates"
            ])
        elif query_type == QueryType.EXPLORATORY:
            rules.extend([
                "Provide comprehensive overview with key sections",
                "Include related topics the user might want to explore",
                "Offer to provide more details on specific aspects"
            ])
        
        # Add intent-specific rules
        if 'scholarship' in intents or 'financial aid' in intents:
            rules.append("Include available financial assistance options")
        if 'subsidy' in intents or 'sssdp' in intents:
            rules.append("Explain SSSDP subsidy details if applicable")
        if 'career prospects' in intents:
            rules.append("Include employment and career information")
        
        return rules


class QueryIntelligence:
    """Main class that orchestrates all query intelligence components"""
    
    def __init__(self):
        self.classifier = QueryClassifier()
        self.context_inferencer = UserContextInferencer()
        self.intent_detector = IntentDetector()

    def analyze(self, query: str, conversation_history: Optional[List[Dict]] = None) -> QueryClassification:
        """Perform complete query analysis"""
        # 1. Classify query type
        query_type = self.classifier.classify(query)
        confidence = self.classifier.get_confidence(query, query_type)
        
        # 2. Infer user context
        user_profile = self.context_inferencer.infer(query, conversation_history)
        context_boost_terms = self.context_inferencer.get_context_boost_terms(user_profile)
        
        # 3. Detect intents
        intents = self.intent_detector.detect_intents(query, query_type)
        implicit_needs = self.intent_detector.get_proactive_info_rules(query_type, intents)
        
        # 4. Generate expanded queries
        expanded_queries = self.intent_detector.get_expanded_queries(query, intents)
        
        # 5. Build response guidelines
        response_guidelines = self._build_response_guidelines(query_type, user_profile, intents)
        
        return QueryClassification(
            query_type=query_type,
            user_profile=user_profile,
            intents=intents,
            implicit_needs=implicit_needs,
            confidence=confidence,
            expanded_queries=expanded_queries,
            context_boost_terms=context_boost_terms,
            response_guidelines=response_guidelines
        )

    def _build_response_guidelines(self, query_type: QueryType, user_profile: UserProfile, 
                                   intents: List[str]) -> Dict[str, Any]:
        """Build response generation guidelines based on analysis"""
        guidelines = {
            'structure': self._get_response_structure(query_type),
            'tone': self._get_response_tone(user_profile),
            'depth': self._get_response_depth(query_type),
            'include_follow_up': query_type in [QueryType.EXPLORATORY, QueryType.ELIGIBILITY],
            'proactive_topics': intents[:3] if intents else [],
        }
        return guidelines

    def _get_response_structure(self, query_type: QueryType) -> str:
        """Get recommended response structure for query type"""
        structures = {
            QueryType.FACTUAL_LOOKUP: "direct_answer",
            QueryType.EXPLORATORY: "sectioned",
            QueryType.COMPARATIVE: "comparison_table",
            QueryType.PROCEDURAL: "step_by_step",
            QueryType.ELIGIBILITY: "requirements_checklist",
            QueryType.TEMPORAL: "timeline",
        }
        return structures.get(query_type, "default")

    def _get_response_tone(self, user_profile: UserProfile) -> str:
        """Get recommended response tone based on user profile"""
        # All users get professional but friendly tone
        # Slight adjustments based on profile
        if user_profile == UserProfile.CURRENT_STUDENT:
            return "informative_concise"
        elif user_profile in [UserProfile.LOCAL_DSE, UserProfile.PROSPECTIVE]:
            return "encouraging_helpful"
        elif user_profile == UserProfile.INTERNATIONAL:
            return "clear_detailed"  # May need more explanation
        return "professional_friendly"

    def _get_response_depth(self, query_type: QueryType) -> str:
        """Get recommended response depth"""
        if query_type == QueryType.FACTUAL_LOOKUP:
            return "concise"
        elif query_type in [QueryType.EXPLORATORY, QueryType.COMPARATIVE]:
            return "comprehensive"
        elif query_type == QueryType.PROCEDURAL:
            return "detailed"
        return "moderate"

    def to_dict(self, classification: QueryClassification) -> Dict[str, Any]:
        """Convert classification to dictionary for JSON serialization"""
        return {
            'query_type': classification.query_type.value,
            'user_profile': classification.user_profile.value,
            'intents': classification.intents,
            'implicit_needs': classification.implicit_needs,
            'confidence': classification.confidence,
            'expanded_queries': classification.expanded_queries,
            'context_boost_terms': classification.context_boost_terms,
            'response_guidelines': classification.response_guidelines
        }
'''