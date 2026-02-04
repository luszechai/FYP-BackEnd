"""Query enhancement module for better retrieval"""
import re
from typing import List, Dict


class QueryEnhancer:
    """Enhances queries for better retrieval"""

    def __init__(self):
        self.titles = [
            'professor', 'prof', 'dr', 'doctor', 'lecturer', 'senior lecturer',
            'assistant professor', 'associate professor', 'instructor', 'teacher',
            'faculty', 'staff', 'dean', 'head'
        ]

        # Role-based titles for detecting queries about people by their position
        self.role_titles = [
            'programme leader', 'program leader', 'head of', 'director',
            'dean', 'coordinator', 'chair', 'head', 'leader', 'manager',
            'associate head', 'deputy head', 'acting head'
        ]

        self.department_aliases = {
            'cis': ['computing and information sciences', 'computer science', 'cs', 'computing'],
            'it': ['information technology'],
            'ai': ['artificial intelligence'],
        }

        # Define Program/Course Acronyms
        self.program_aliases = {
            'bsc': 'Bachelor of Science',
            'msc': 'Master of Science',
            'ba': 'Bachelor of Arts',
            'hd': 'Higher Diploma',
            'asso': 'Associate Degree',
            'bba': 'Bachelor of Business Administration',
            'cs': ['computer science', 'computing science', 'computer studies'],
            'it': ['information technology', 'info tech'],
            'ai': ['artificial intelligence', 'machine learning', 'data science'],
            'business': ['business administration', 'management', 'commerce', 'mba'],
            'engineering': ['eng', 'technology'],
            'bs': ['bachelor of science', 'undergraduate science'],
            'bsc': ['bachelor of science', 'b.sc'],
            'ma': ['master of arts', 'masters arts'],
            'ms': ['master of science', 'masters science', 'm.sc'],
            'msc': ['master of science', 'm.sc'],
            'phd': ['doctor of philosophy', 'doctoral', 'doctorate'],
            'mba': ['master of business administration', 'business masters'],
        }

    def is_person_query(self, query: str) -> bool:
        """Detect if query is asking about a person"""
        query_lower = query.lower()

        person_patterns = [
            r'\b(who is|where is|find|locate|contact)\b.*\b[A-Z][a-z]+',
            r'\b(professor|prof|dr|doctor|lecturer|mr|ms|miss)\b',
            r'\b(tell me about|info about|information about)\b.*\b[A-Z]',
        ]

        for pattern in person_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return True

        # Check for capitalized words, but exclude the first word (often capitalized by grammar)
        # and require at least 2 capitalized words OR a capitalized word not at the start
        words = query.split()
        if len(words) == 0:
            return False
        
        # Get all capitalized words (excluding the first word)
        capitalized = [w for w in words[1:] if w and len(w) > 1 and w[0].isupper()]
        
        # Return True if there are multiple capitalized words (likely a name)
        # OR if there's at least one capitalized word in the middle/end of the sentence
        return len(capitalized) >= 1

    def is_role_query(self, query: str) -> bool:
        """Detect if query is asking about a person by their role/position"""
        query_lower = query.lower()
        
        # Patterns for role-based person queries
        role_patterns = [
            r'\bwho\s+is\s+(the\s+)?(programme|program)\s+leader\b',
            r'\bwho\s+is\s+(the\s+)?head\s+of\b',
            r'\bwho\s+is\s+(the\s+)?director\s+of\b',
            r'\bwho\s+is\s+(the\s+)?dean\s+of\b',
            r'\bwho\s+is\s+(the\s+)?coordinator\s+(of|for)\b',
            r'\bwho\s+(leads?|runs?|manages?|heads?)\b',
            r'\b(programme|program)\s+leader\s+of\b',
            r'\bhead\s+of\s+(the\s+)?(department|school|programme|program)\b',
        ]
        
        return any(re.search(p, query_lower) for p in role_patterns)

    def is_scholarship_query(self, query: str) -> bool:
        """Detect if query is about scholarships"""
        query_lower = query.lower()
        scholarship_patterns = [
            r'\bscholarship[s]?\b',
            r'\bfinancial\s+(aid|assistance|support)\b',
            r'\bbursary|bursaries\b',
            r'\bgrant[s]?\b.*\b(student|academic)\b',
            r'\b(academic|merit|need[- ]based)\s+award[s]?\b',
        ]
        return any(re.search(p, query_lower) for p in scholarship_patterns)

    def _extract_scholarship_keywords(self, query: str) -> List[str]:
        """Extract scholarship-related keywords from query"""
        keywords = []
        query_lower = query.lower()
        
        # Core scholarship terms
        scholarship_terms = [
            'scholarship', 'scholarships', 'bursary', 'bursaries', 
            'financial aid', 'grant', 'grants', 'award', 'awards',
            'academic achievement', 'merit', 'need-based'
        ]
        
        for term in scholarship_terms:
            if term in query_lower:
                keywords.append(term)
        
        # Add data format keywords that match the dataset
        keywords.extend(['scholarship_name', 'entrance_scholarships', 'admission_scholarships'])
        
        # Check for listing intent
        list_patterns = ['list', 'what', 'which', 'available', 'types of', 
                        'show', 'tell me', 'give me', 'all']
        if any(p in query_lower for p in list_patterns):
            keywords.extend(['scholarships available', 'available scholarships'])
        
        # Check for eligibility/criteria questions
        if any(term in query_lower for term in ['eligible', 'eligibility', 'qualify', 'criteria', 'requirement']):
            keywords.extend(['eligibility', 'criteria', 'requirements'])
        
        return list(set(keywords))

    def expand_scholarship_query(self, query: str) -> List[str]:
        """Generate query variations for scholarship searches"""
        queries = [query]
        query_lower = query.lower()
        
        # Check if this is a listing query (asking for available scholarships)
        list_patterns = ['list', 'what', 'which', 'available', 'types of', 
                        'show', 'tell me', 'give me', 'all', 'any']
        is_list_query = any(p in query_lower for p in list_patterns)
        
        if is_list_query:
            # Add queries that match the data format in the dataset
            queries.extend([
                "scholarship_name",
                "scholarships available",
                "entrance scholarships",
                "admission scholarships",
                "Academic Achievement Scholarships",
                "available scholarships for students",
                "scholarship eligibility criteria",
                "types of scholarships",
                "scholarship opportunities"
            ])
        
        # Add general scholarship search terms
        queries.extend([
            "scholarship",
            "scholarships offered",
            "financial aid scholarships",
            "scholarship program"
        ])
        
        # Check for specific scholarship types mentioned
        if 'academic' in query_lower or 'achievement' in query_lower:
            queries.append("Academic Achievement Scholarships AAS")
        if 'admission' in query_lower or 'entrance' in query_lower:
            queries.extend(["Admission Scholarships", "entrance scholarships"])
        if 'merit' in query_lower:
            queries.append("merit-based scholarships")
        if 'need' in query_lower or 'financial' in query_lower:
            queries.extend(["need-based scholarships", "financial aid"])
        
        # Check for deadline/date questions
        if any(term in query_lower for term in ['deadline', 'due', 'when', 'date']):
            queries.extend([
                "scholarship deadline",
                "scholarship application deadline",
                "scholarship due date"
            ])
        
        # Check for eligibility questions
        if any(term in query_lower for term in ['eligible', 'eligibility', 'qualify', 'criteria', 'requirement', 'how to']):
            queries.extend([
                "scholarship eligibility",
                "scholarship requirements",
                "scholarship criteria",
                "how to apply for scholarship"
            ])
        
        return list(set(queries))

    def extract_name_components(self, query: str) -> Dict[str, List[str]]:
        """Extract potential name components from query"""
        stop_words = {
            'who', 'is', 'where', 'can', 'i', 'find', 'the', 'from', 'about',
            'tell', 'me', 'how', 'what', 'a', 'an', 'in', 'at', 'on', 'of'
        }

        query_clean = query.lower()
        for title in self.titles:
            query_clean = re.sub(r'\b' + title + r'\b', '', query_clean, flags=re.IGNORECASE)

        words = [w.strip() for w in query_clean.split() if w.strip()]
        words = [w for w in words if w.lower() not in stop_words]

        original_words = query.split()
        capitalized_words = [w for w in original_words if w and len(w) > 1 and w[0].isupper()]

        return {
            'all_terms': words,
            'capitalized': capitalized_words,
            'potential_names': [w for w in words if len(w) > 2]
        }

    def is_program_query(self, query: str) -> bool:
        """Detect if query is about a course or program"""
        query_lower = query.lower()

        # Pattern for course codes (e.g., CS101, COMP 300, ENGL-201)
        # Looks for 2-4 letters, optional space/dash, 3-4 digits
        course_code_pattern = r'\b[a-z]{2,4}[\s-]?\d{3,4}[a-z]?\b'

        # Keywords for degrees
        program_keywords = [
            'bachelor', 'master', 'diploma', 'degree', 'major', 'minor',
            'syllabus', 'prerequisite', 'curriculum', 'core course', 'elective'
        ]

        if re.search(course_code_pattern, query_lower):
            return True

        if any(kw in query_lower for kw in program_keywords):
            return True

        # Check for degree acronyms (BSc, MSc) specifically as whole words
        words = set(query_lower.split())
        if any(alias in words for alias in self.program_aliases):
            return True

        return False

    def expand_person_query(self, query: str) -> List[str]:
        """Generate multiple query variations for person search"""
        if not self.is_person_query(query):
            return [query]

        name_components = self.extract_name_components(query)
        queries = [query]

        for name in name_components['potential_names']:
            queries.append(f"Dr {name}")
            queries.append(f"Professor {name}")
            queries.append(f"Lecturer {name}")
            queries.append(f"Mr {name}")

        if name_components['capitalized']:
            capitalized_query = " ".join(name_components['capitalized'])
            queries.append(capitalized_query)
            queries.append(f"Dr {capitalized_query}")
            queries.append(f"Professor {capitalized_query}")

        for name in name_components['potential_names']:
            queries.append(f"{name}@sfu.edu.hk")

        return list(set(queries))

    def expand_role_query(self, query: str) -> List[str]:
        """Generate query variations for role-based person searches"""
        queries = [query]
        query_lower = query.lower()
        
        # Expand program/department abbreviations (ai -> artificial intelligence)
        for abbrev, expansions in self.department_aliases.items():
            # Check if abbreviation is a standalone word in query
            if re.search(r'\b' + abbrev + r'\b', query_lower):
                for exp in expansions:
                    expanded_query = re.sub(r'\b' + abbrev + r'\b', exp, query_lower)
                    queries.append(expanded_query)
        
        # Role synonyms mapping
        role_synonyms = {
            'programme leader': ['program leader', 'program director', 'programme director', 'programme head'],
            'program leader': ['programme leader', 'program director', 'programme director', 'program head'],
            'head of': ['director of', 'chair of', 'head'],
            'director of': ['head of', 'chair of', 'director'],
            'coordinator': ['coordinator of', 'program coordinator', 'programme coordinator'],
        }
        
        for role, synonyms in role_synonyms.items():
            if role in query_lower:
                for syn in synonyms:
                    queries.append(query_lower.replace(role, syn))
        
        # Extract the subject/program from the query and create direct search queries
        # e.g., "who is the programme leader of ai" -> extract "ai" or "artificial intelligence"
        subject_match = re.search(r'(?:of|for)\s+(?:the\s+)?(.+?)(?:\s*\?|$)', query_lower)
        if subject_match:
            subject = subject_match.group(1).strip()
            # Expand abbreviations in subject
            expanded_subject = subject
            for abbrev, expansions in self.department_aliases.items():
                if abbrev == subject or re.search(r'\b' + abbrev + r'\b', subject):
                    expanded_subject = expansions[0] if expansions else subject
                    break
            
            # Direct role search queries that match the data format
            queries.append(f"Programme Leader {expanded_subject}")
            queries.append(f"Programme Leader of {expanded_subject}")
            queries.append(f"Programme Leader of Bachelor of Science {expanded_subject}")
            queries.append(f"role Programme Leader {expanded_subject}")
        
        return list(set(queries))

    def _extract_role_keywords(self, query: str) -> List[str]:
        """Extract role-related keywords from query"""
        keywords = []
        query_lower = query.lower()
        
        # Add role titles found in query
        for role in self.role_titles:
            if role in query_lower:
                keywords.append(role)
        
        # Expand and add department/program terms
        for abbrev, expansions in self.department_aliases.items():
            if re.search(r'\b' + abbrev + r'\b', query_lower):
                keywords.append(abbrev)
                keywords.extend(expansions)
        
        # Add common role-related terms
        role_terms = ['programme leader', 'program leader', 'director', 'head', 'role']
        keywords.extend([t for t in role_terms if t in query_lower])
        
        return list(set(keywords))

    def expand_program_query(self, query: str) -> List[str]:
        """Generate variations for course/program queries"""
        queries = [query]
        query_lower = query.lower()

        # A. Handle Course Codes (CS 101 <-> CS101)
        # Find codes like "CS 101" or "CS101"
        code_matches = re.findall(r'\b([a-z]{2,4})[\s-]?(\d{3,4}[a-z]?)\b', query_lower)

        for subject, number in code_matches:
            # Add the compact version (CS101)
            queries.append(f"{subject.upper()}{number}")
            # Add the spaced version (CS 101)
            queries.append(f"{subject.upper()} {number}")
            # Add context specific queries
            queries.append(f"{subject.upper()}{number} prerequisite")
            queries.append(f"{subject.upper()}{number} syllabus")

        # B. Expand Degree Acronyms (BSc -> Bachelor of Science)
        for word in query_lower.split():
            if word in self.program_aliases:
                alias_value = self.program_aliases[word]
                # Handle both string and list values
                if isinstance(alias_value, str):
                    new_q = query_lower.replace(word, alias_value)
                    queries.append(new_q)
                elif isinstance(alias_value, list):
                    for expansion in alias_value:
                        new_q = query_lower.replace(word, expansion)
                        queries.append(new_q)

        # C. Expand Departments (using existing department_aliases)
        # If query is "CIS courses", add "Computer Science courses"
        for abbrev, full_names in self.department_aliases.items():
            if abbrev in query_lower.split():  # match exact word
                for full_name in full_names:
                    queries.append(query_lower.replace(abbrev, full_name))

        return list(set(queries))

    def expand_department_query(self, query: str) -> str:
        """Expand department abbreviations"""
        query_lower = query.lower()

        for abbrev, full_names in self.department_aliases.items():
            if abbrev in query_lower:
                for full_name in full_names:
                    if full_name not in query_lower:
                        query_lower += f" {full_name}"

        return query_lower

    def enhance_query(self, query: str) -> Dict[str, any]:
        """Main query enhancement function"""
        is_person = self.is_person_query(query)
        is_program = self.is_program_query(query)
        is_anaphora = self.is_anaphora_query(query)
        is_role = self.is_role_query(query)
        is_scholarship = self.is_scholarship_query(query)

        enhanced = {
            'original': query,
            'is_person_query': is_person,
            'is_program_query': is_program,
            'is_anaphora_query': is_anaphora,
            'is_role_query': is_role,
            'is_scholarship_query': is_scholarship,
            'expanded_queries': [],
            'keywords': []
        }

        # Prioritize role queries (asking about person by position) over generic person queries
        # Then scholarship queries, program queries, then person queries
        if is_role:
            # Role-based queries: "who is the programme leader of AI"
            enhanced['expanded_queries'] = self.expand_role_query(query)
            enhanced['keywords'] = self._extract_role_keywords(query)
        elif is_scholarship:
            # Scholarship queries: "what scholarships are available", "list of scholarships"
            enhanced['expanded_queries'] = self.expand_scholarship_query(query)
            enhanced['keywords'] = self._extract_scholarship_keywords(query)
        elif is_program:
            enhanced['expanded_queries'] = self.expand_program_query(query)
            # Use course codes as high-value keywords
            enhanced['keywords'] = re.findall(r'\b[a-z]{2,4}\d{3,4}\b', query.lower().replace(" ", ""))
        elif is_person:
            enhanced['expanded_queries'] = self.expand_person_query(query)
            enhanced['keywords'] = self.extract_name_components(query)['potential_names']
        else:
            enhanced['expanded_queries'] = [query]
            enhanced['keywords'] = query.split()

        enhanced['department_expanded'] = self.expand_department_query(query)

        return enhanced

    def is_anaphora_query(self, query: str) -> bool:
        """Detect if query contains anaphora or references (pronouns, ordinals, demonstratives)"""
        query_lower = query.lower()
        
        # Pronouns that typically refer to previous context
        pronouns = [
            r'\b(it|its|they|them|their|this|that|these|those)\b',
            r'\b(he|she|him|her|his|hers)\b'
        ]
        
        # Ordinal references ("the first one", "the second", "the last", etc.)
        ordinal_patterns = [
            r'\b(the\s+)?(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+(one|option|item|program|scholarship|deadline|requirement)\b',
            r'\b(the\s+)?(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b',
            r'\b(the\s+)?(last|previous|earlier|above|mentioned)\s+(one|option|item|program|scholarship|deadline|requirement)\b',
            r'\b(the\s+)?(last|previous|earlier|above|mentioned)\b'
        ]
        
        # Demonstrative references
        demonstrative_patterns = [
            r'\b(this|that|these|those)\s+(one|option|item|program|scholarship|deadline|requirement)\b',
            r'\b(this|that|these|those)\b'
        ]
        
        # Check for pronouns (but exclude common false positives)
        for pattern in pronouns:
            if re.search(pattern, query_lower):
                # Exclude common phrases that aren't anaphora
                if not re.search(r'\b(it\s+is|it\s+was|it\s+can|it\s+will|it\s+has|it\s+does|this\s+is|that\s+is|these\s+are|those\s+are)\b', query_lower):
                    return True
        
        # Check for ordinal references
        for pattern in ordinal_patterns:
            if re.search(pattern, query_lower):
                return True
        
        # Check for demonstrative references (more strict)
        for pattern in demonstrative_patterns:
            if re.search(pattern, query_lower):
                return True
        
        # Very short queries (1-3 words) are often references
        words = query_lower.split()
        if len(words) <= 3 and any(word in ['it', 'this', 'that', 'first', 'second', 'last', 'one'] for word in words):
            return True
        
        return False

    def categorize_query(self, query: str) -> str:
        """Categorize query into predefined categories"""
        query_lower = query.lower()

        # Define category keywords
        categories = {
            'Admission': ['admission', 'apply', 'application', 'requirement', 'eligibility', 'deadline'],
            'Faculty': ['professor', 'faculty', 'staff', 'lecturer', 'dr', 'teacher', 'instructor'],
            'Fees': ['fee', 'tuition', 'cost', 'payment', 'scholarship', 'financial'],
            'Programs': ['program', 'course', 'major', 'degree', 'bachelor', 'master', 'phd'],
            'Financial Aid': ['scholarship', 'grant', 'loan', 'aid', 'bursary', 'funding'],
            'Contact': ['contact', 'email', 'phone', 'office', 'location', 'address'],
            'Location': ['where', 'location', 'campus', 'building', 'room'],
            'Date/Time': ['when', 'date', 'time', 'deadline', 'due date', 'today', 'tomorrow', 
                          'next week', 'next month', 'when is', 'what time', 'what date', 
                          'schedule', 'calendar', 'opening', 'closing', 'latest']
        }

        for category, keywords in categories.items():
            if any(keyword in query_lower for keyword in keywords):
                return category

        return 'General'

