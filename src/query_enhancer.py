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

        words = query.split()
        capitalized = [w for w in words if w and len(w) > 1 and w[0].isupper()]
        return len(capitalized) >= 1

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

        enhanced = {
            'original': query,
            'is_person_query': is_person,
            'is_program_query': is_program,
            'expanded_queries': [],
            'keywords': []
        }

        if is_person:
            enhanced['expanded_queries'] = self.expand_person_query(query)
            enhanced['keywords'] = self.extract_name_components(query)['potential_names']
        elif is_program:
            enhanced['expanded_queries'] = self.expand_program_query(query)
            # Use course codes as high-value keywords
            enhanced['keywords'] = re.findall(r'\b[a-z]{2,4}\d{3,4}\b', query.lower().replace(" ", ""))
        else:
            enhanced['expanded_queries'] = [query]
            enhanced['keywords'] = query.split()

        enhanced['department_expanded'] = self.expand_department_query(query)

        return enhanced

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

