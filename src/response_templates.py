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
"""Response Templates Module - Structured response templates for each query type"""
from typing import Dict, List, Any, Optional
from src.query_intelligence import QueryType, UserProfile


class ResponseTemplates:
    """Provides structured response templates based on query classification"""
    
    # Template definitions for each query type
    TEMPLATES = {
        QueryType.FACTUAL_LOOKUP: {
            'structure': 'direct_answer',
            'format': """**{answer_title}**
{direct_answer}

{additional_context}""",
            'guidelines': [
                "Provide the specific fact directly",
                "Include relevant contact information if applicable",
                "Keep response concise and focused",
                "Add brief context only if necessary for clarity"
            ],
            'max_length': 'short',
            'include_sources': True,
            'follow_up_prompt': None
        },
        
        QueryType.EXPLORATORY: {
            'structure': 'sectioned',
            'format': """**{main_topic}**

{overview}

**Key Highlights**
{highlights}

**Details**
{details}

{proactive_info}

{follow_up}""",
            'guidelines': [
                "Start with a concise overview",
                "Organize information into clear sections",
                "Include key statistics and facts",
                "Add related information the user might find useful",
                "End with a helpful follow-up question or offer"
            ],
            'max_length': 'comprehensive',
            'include_sources': True,
            'follow_up_prompt': "Would you like more details about any specific aspect?"
        },
        
        QueryType.COMPARATIVE: {
            'structure': 'comparison',
            'format': """**Comparison: {option_a} vs {option_b}**

**Overview**
{overview}

**Key Differences**
| Aspect | {option_a} | {option_b} |
|--------|{col_a_separator}|{col_b_separator}|
{comparison_rows}

**Summary**
{recommendation}

{follow_up}""",
            'guidelines': [
                "Present options side by side",
                "Highlight key differences clearly",
                "Use tables or structured lists for easy comparison",
                "Provide an objective summary",
                "Offer personalized recommendation if context allows"
            ],
            'max_length': 'moderate',
            'include_sources': True,
            'follow_up_prompt': "Which option would you like to know more about?"
        },
        
        QueryType.PROCEDURAL: {
            'structure': 'step_by_step',
            'format': """**How to {process_name}**

**Overview**
{overview}

**Steps**
{steps}

**Important Tips**
{tips}

**Common Mistakes to Avoid**
{mistakes}

{deadline_info}""",
            'guidelines': [
                "Break down the process into clear, numbered steps",
                "Include required documents or prerequisites",
                "Mention important deadlines",
                "Add helpful tips for success",
                "Warn about common mistakes"
            ],
            'max_length': 'detailed',
            'include_sources': True,
            'follow_up_prompt': "Do you need help with any specific step?"
        },
        
        QueryType.ELIGIBILITY: {
            'structure': 'requirements_checklist',
            'format': """**Eligibility for {subject}**

**Requirements**
{requirements}

**Your Situation**
{situation_assessment}

**Alternative Pathways**
{alternatives}

**Next Steps**
{next_steps}

{follow_up}""",
            'guidelines': [
                "List all requirements clearly",
                "Indicate which are mandatory vs preferred",
                "Provide alternative pathways if requirements not met",
                "Include exceptions or special cases",
                "Give clear next steps for eligible candidates"
            ],
            'max_length': 'moderate',
            'include_sources': True,
            'follow_up_prompt': "Would you like to know about alternative pathways or specific requirements?"
        },
        
        QueryType.TEMPORAL: {
            'structure': 'timeline',
            'format': """**{deadline_subject}**

**Key Date**
{main_deadline}

**Status**
{status_relative_to_today}

**Related Deadlines**
{related_deadlines}

**What to Do**
{action_items}

**If You Miss the Deadline**
{missed_deadline_info}""",
            'guidelines': [
                "State the main deadline clearly and prominently",
                "Indicate if the deadline has passed or is upcoming",
                "Include all related deadlines",
                "Explain consequences of missing the deadline",
                "Provide alternative options if deadline is passed"
            ],
            'max_length': 'moderate',
            'include_sources': True,
            'follow_up_prompt': "Would you like to know about other important dates?"
        }
    }
    
    # User profile specific adjustments
    PROFILE_ADJUSTMENTS = {
        UserProfile.LOCAL_DSE: {
            'boost_terms': ['JUPAS', 'DSE', 'Band A/B/C', 'local student'],
            'additional_context': [
                "Information relevant to JUPAS applicants",
                "DSE grade requirements and equivalents"
            ],
            'tone': 'encouraging'
        },
        UserProfile.INTERNATIONAL: {
            'boost_terms': ['international', 'non-local', 'visa', 'IELTS'],
            'additional_context': [
                "English language requirements",
                "Visa and immigration information",
                "International student support services"
            ],
            'tone': 'detailed_and_clear'
        },
        UserProfile.TRANSFER: {
            'boost_terms': ['transfer', 'credit', 'articulation', 'sub-degree'],
            'additional_context': [
                "Credit transfer policies",
                "Articulation arrangements",
                "Senior year admission requirements"
            ],
            'tone': 'informative'
        },
        UserProfile.CURRENT_STUDENT: {
            'boost_terms': ['current student', 'enrolled', 'registration'],
            'additional_context': [
                "Student portal information",
                "Academic advisor contact"
            ],
            'tone': 'concise'
        },
        UserProfile.PROSPECTIVE: {
            'boost_terms': ['admission', 'application', 'requirement'],
            'additional_context': [
                "Application process overview",
                "Campus visit information",
                "Contact for admission inquiries"
            ],
            'tone': 'welcoming'
        }
    }
    
    # Proactive information rules
    PROACTIVE_INFO = {
        'tuition': {
            'related_topics': ['SSSDP subsidy', 'scholarship', 'payment plan', 'financial aid'],
            'template': """
**Financial Information**
{financial_details}"""
        },
        'program': {
            'related_topics': ['admission requirements', 'career prospects', 'curriculum highlights'],
            'template': """
**Related Information**
{related_details}"""
        },
        'deadline': {
            'related_topics': ['other deadlines', 'late application policy', 'extension possibility'],
            'template': """
**Other Important Dates**
{deadline_details}"""
        },
        'scholarship': {
            'related_topics': ['eligibility', 'application process', 'other financial aid'],
            'template': """
**Scholarship Information**
{scholarship_details}"""
        },
        'admission': {
            'related_topics': ['required documents', 'interview', 'timeline'],
            'template': """
**Admission Tips**
{admission_details}"""
        }
    }

    @classmethod
    def get_template(cls, query_type: QueryType) -> Dict[str, Any]:
        """Get the template for a specific query type"""
        return cls.TEMPLATES.get(query_type, cls.TEMPLATES[QueryType.EXPLORATORY])

    @classmethod
    def get_profile_adjustments(cls, user_profile: UserProfile) -> Dict[str, Any]:
        """Get adjustments for a specific user profile"""
        return cls.PROFILE_ADJUSTMENTS.get(user_profile, cls.PROFILE_ADJUSTMENTS[UserProfile.PROSPECTIVE])

    @classmethod
    def get_proactive_info_template(cls, topic: str) -> Optional[Dict[str, Any]]:
        """Get proactive information template for a topic"""
        for key, value in cls.PROACTIVE_INFO.items():
            if key in topic.lower():
                return value
        return None

    @classmethod
    def get_response_structure_prompt(cls, query_type: QueryType, user_profile: UserProfile) -> str:
        """Generate a prompt instruction for response structure"""
        template = cls.get_template(query_type)
        profile_adj = cls.get_profile_adjustments(user_profile)
        
        structure_prompts = {
            'direct_answer': """Structure your response as a direct answer:
- State the specific information requested immediately
- Add brief relevant context if needed
- Keep response concise and focused""",
            
            'sectioned': """Structure your response in clear sections:
- Start with a brief overview (2-3 sentences)
- Use **bold headers** for each section
- Include Key Highlights, Details, and Related Information
- End with an offer to provide more details""",
            
            'comparison': """Structure your response as a comparison:
- Provide a brief overview of both options
- Use a comparison format highlighting key differences
- Include a summary with objective analysis
- If possible, offer a personalized recommendation""",
            
            'step_by_step': """Structure your response as a step-by-step guide:
- Number each step clearly (1, 2, 3...)
- Include any prerequisites at the beginning
- Add helpful tips after the steps
- Mention important deadlines if applicable""",
            
            'requirements_checklist': """Structure your response as a requirements checklist:
- List all requirements clearly (use bullet points)
- Indicate which are mandatory vs preferred
- Include alternative pathways if available
- Provide clear next steps""",
            
            'timeline': """Structure your response around the timeline:
- State the key date prominently
- Indicate if it's upcoming or has passed (relative to current date)
- List related deadlines
- Explain what happens if the deadline is missed"""
        }
        
        base_prompt = structure_prompts.get(template['structure'], structure_prompts['sectioned'])
        
        # Add profile-specific guidance
        tone_guidance = {
            'encouraging': "\\n- Use an encouraging and supportive tone",
            'detailed_and_clear': "\\n- Provide extra clarity for complex terms\\n- Avoid local jargon that international students may not know",
            'informative': "\\n- Focus on practical, actionable information",
            'concise': "\\n- Keep response brief and to the point",
            'welcoming': "\\n- Use a welcoming tone to prospective students"
        }
        
        profile_tone = profile_adj.get('tone', 'professional')
        base_prompt += tone_guidance.get(profile_tone, "")
        
        # Add max length guidance
        length_guidance = {
            'short': "\\n- Keep response under 150 words",
            'moderate': "\\n- Aim for 150-300 words",
            'detailed': "\\n- Provide detailed response (300-500 words)",
            'comprehensive': "\\n- Provide comprehensive coverage (400-600 words)"
        }
        base_prompt += length_guidance.get(template['max_length'], length_guidance['moderate'])
        
        return base_prompt

    @classmethod
    def get_proactive_info_prompt(cls, intents: List[str], query_type: QueryType) -> str:
        """Generate prompt for including proactive information"""
        if not intents:
            return ""
        
        prompt_parts = ["\\nProactive Information to Include:"]
        
        for intent in intents[:3]:  # Limit to top 3
            template = cls.get_proactive_info_template(intent)
            if template:
                prompt_parts.append(f"- {intent}: {', '.join(template['related_topics'][:3])}")
        
        if query_type == QueryType.TEMPORAL:
            prompt_parts.append("- Always mention related deadlines and consequences of missing them")
        elif query_type == QueryType.ELIGIBILITY:
            prompt_parts.append("- Always include alternative pathways if requirements are not met")
        elif query_type == QueryType.EXPLORATORY:
            prompt_parts.append("- Offer to provide more details on specific aspects")
        
        if len(prompt_parts) > 1:
            return "\\n".join(prompt_parts)
        return ""

    @classmethod
    def format_follow_up_question(cls, query_type: QueryType, user_profile: UserProfile) -> Optional[str]:
        """Get appropriate follow-up question based on context"""
        template = cls.get_template(query_type)
        
        if not template.get('follow_up_prompt'):
            return None
        
        # Customize follow-up based on user profile
        profile_specific_followups = {
            UserProfile.LOCAL_DSE: {
                QueryType.ELIGIBILITY: "Would you like to know about the JUPAS application process?",
                QueryType.EXPLORATORY: "Would you like to know about DSE grade requirements or JUPAS codes?",
            },
            UserProfile.INTERNATIONAL: {
                QueryType.ELIGIBILITY: "Would you like information about visa requirements or English language tests?",
                QueryType.EXPLORATORY: "Would you like details about international student support services?",
            },
            UserProfile.TRANSFER: {
                QueryType.ELIGIBILITY: "Would you like to know about credit transfer policies?",
                QueryType.EXPLORATORY: "Would you like details about articulation arrangements?",
            }
        }
        
        profile_followups = profile_specific_followups.get(user_profile, {})
        return profile_followups.get(query_type, template['follow_up_prompt'])


class ResponseFormatter:
    """Formats responses according to templates"""
    
    @staticmethod
    def format_bullet_list(items: List[str], prefix: str = "- ") -> str:
        """Format a list of items as bullet points"""
        return "\\n".join(f"{prefix}{item}" for item in items)
    
    @staticmethod
    def format_numbered_list(items: List[str]) -> str:
        """Format a list of items as numbered list"""
        return "\\n".join(f"{i+1}. {item}" for i, item in enumerate(items))
    
    @staticmethod
    def format_comparison_table(headers: List[str], rows: List[List[str]]) -> str:
        """Format data as a markdown comparison table"""
        if not headers or not rows:
            return ""
        
        # Calculate column widths
        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                if i < len(col_widths):
                    col_widths[i] = max(col_widths[i], len(str(cell)))
        
        # Build table
        lines = []
        # Header row
        header_row = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
        lines.append(f"| {header_row} |")
        # Separator
        separator = " | ".join("-" * w for w in col_widths)
        lines.append(f"| {separator} |")
        # Data rows
        for row in rows:
            data_row = " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row))
            lines.append(f"| {data_row} |")
        
        return "\\n".join(lines)
    
    @staticmethod
    def format_deadline_status(deadline_date: str, current_date: str, is_passed: bool) -> str:
        """Format deadline with status indicator"""
        if is_passed:
            return f"~~{deadline_date}~~ (This deadline has passed as of {current_date})"
        else:
            return f"**{deadline_date}** (Upcoming)"
    
    @staticmethod
    def wrap_section(title: str, content: str) -> str:
        """Wrap content in a titled section"""
        if not content.strip():
            return ""
        return f"**{title}**\\n{content}\\n"
'''
