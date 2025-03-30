import os
import spacy
from typing import Dict, List, Optional
from transformers import pipeline
from spacy.matcher import Matcher
from spacy.tokens import Doc

class ResumeParser:
    def __init__(self):
        # Load spaCy model
        self.nlp = spacy.load('en_core_web_sm')
        # Initialize the NER pipeline for custom entity recognition
        self.ner = pipeline('ner', model='dbmdz/bert-large-cased-finetuned-conll03-english')
        # Initialize matcher for custom patterns
        self.matcher = Matcher(self.nlp.vocab)
    
    def parse_resume_text(self, text: str) -> Dict:
        """Parse resume text using spaCy and transformers"""
        try:
            if not text or not text.strip():
                print('Error: Empty resume text provided')
                return self._get_empty_result()

            # Process text with spaCy
            doc = self.nlp(text.strip())
            
            # Extract structured data
            data = {
                'skills': self._extract_skills_from_text(doc),
                'education': self._extract_education_from_text(doc),
                'work_experience': self._extract_experience_from_text(doc)
            }
            
            # Process the extracted data
            return self._process_parsed_data(data)
            
        except Exception as e:
            print(f'Unexpected error parsing resume: {str(e)}')
            return self._get_empty_result()
    
    def _process_parsed_data(self, data: Dict) -> Dict:
        """Process the parsed resume data into structured format"""
        result = {
            'skills': self._extract_skills(data.get('skills', [])),
            'education': self._extract_education(data.get('education', [])),
            'experience': self._extract_experience(data.get('work_experience', [])),
            'analysis': self._analyze_resume(data)
        }
        return result
    
    def _extract_skills_from_text(self, doc: Doc) -> List[Dict]:
        """Extract skills from text using spaCy patterns"""
        skills_data = []
        
        # Define skill patterns
        skill_patterns = [
            # Programming Languages
            [{'LOWER': {'IN': ['python', 'java', 'javascript', 'c++', 'ruby', 'php']}}],
            # Web Technologies
            [{'LOWER': {'IN': ['html', 'css', 'react', 'angular', 'vue', 'node.js']}}],
            # Databases
            [{'LOWER': {'IN': ['sql', 'mongodb', 'postgresql', 'mysql', 'oracle']}}],
            # Cloud & DevOps
            [{'LOWER': {'IN': ['aws', 'azure', 'gcp', 'docker', 'kubernetes']}}],
            # Tools & Frameworks
            [{'LOWER': {'IN': ['git', 'jenkins', 'jira', 'confluence', 'maven']}}]
        ]
        
        # Add patterns to matcher
        for pattern in skill_patterns:
            self.matcher.add('SKILL', [pattern])
        
        # Find matches
        matches = self.matcher(doc)
        for match_id, start, end in matches:
            skill_name = doc[start:end].text
            if skill_name.lower() not in [s.get('name', '').lower() for s in skills_data]:
                skills_data.append({'name': skill_name})
        
        return skills_data
        
    def _extract_skills(self, skills_data: List) -> Dict[str, List[str]]:
        """Categorize extracted skills"""
        skills = {
            'programming': [],
            'web': [],
            'database': [],
            'cloud': [],
            'tools': [],
            'methodologies': [],
            'analytics': [],
            'soft_skills': [],
            'certifications': []
        }
        
        for skill in skills_data:
            skill_name = skill.get('name', '').lower()
            # Add skill to appropriate category based on keywords
            if any(lang in skill_name for lang in ['python', 'java', 'javascript', 'c++', 'ruby']):
                skills['programming'].append(skill_name)
            elif any(tech in skill_name for tech in ['html', 'css', 'react', 'angular', 'vue']):
                skills['web'].append(skill_name)
            elif any(db in skill_name for db in ['sql', 'mongodb', 'postgresql', 'mysql']):
                skills['database'].append(skill_name)
            elif any(cloud in skill_name for cloud in ['aws', 'azure', 'gcp', 'docker']):
                skills['cloud'].append(skill_name)
            elif any(tool in skill_name for tool in ['git', 'jenkins', 'jira', 'confluence']):
                skills['tools'].append(skill_name)
            elif any(method in skill_name for method in ['agile', 'scrum', 'kanban', 'waterfall']):
                skills['methodologies'].append(skill_name)
            elif any(analytics in skill_name for analytics in ['machine learning', 'data analysis', 'statistics']):
                skills['analytics'].append(skill_name)
            elif any(soft in skill_name for soft in ['leadership', 'communication', 'teamwork']):
                skills['soft_skills'].append(skill_name)
            elif any(cert in skill_name for cert in ['certified', 'certification', 'license']):
                skills['certifications'].append(skill_name)
        
        return skills
    
    def _extract_education_from_text(self, doc: Doc) -> List[Dict]:
        """Extract education information using NER and pattern matching"""
        education_entries = []
        
        # Define education patterns
        edu_patterns = [
            # Degree patterns
            [{'LOWER': {'IN': ['bachelor', 'master', 'phd', 'bs', 'ms', 'ba', 'ma']}},
             {'LOWER': {'IN': ['of', 'in']}},
             {'POS': 'PROPN', 'OP': '+'}],
            # School patterns
            [{'LOWER': 'university'},
             {'POS': 'PROPN', 'OP': '+'}]
        ]
        
        # Add patterns to matcher
        for pattern in edu_patterns:
            self.matcher.add('EDUCATION', [pattern])
        
        # Find matches
        matches = self.matcher(doc)
        current_entry = {}
        
        for match_id, start, end in matches:
            span = doc[start:end]
            text = span.text
            
            if any(degree in text.lower() for degree in ['bachelor', 'master', 'phd']):
                current_entry['degree'] = text
            elif 'university' in text.lower():
                current_entry['institution'] = text
            
            if current_entry and len(current_entry) >= 2:
                education_entries.append({
                    'degree': current_entry.get('degree', ''),
                    'field': '',  # Will be extracted from context
                    'institution': current_entry.get('institution', ''),
                    'date': '',  # Will be extracted from context
                    'gpa': ''  # Will be extracted from context
                })
                current_entry = {}
        
        return education_entries
    
    def _extract_experience_from_text(self, doc: Doc) -> List[Dict]:
        """Extract work experience using NER and pattern matching"""
        experience_entries = []
        
        # Define work experience patterns
        exp_patterns = [
            # Job title patterns
            [{'POS': {'IN': ['PROPN', 'NOUN']}, 'OP': '+'}, 
             {'LOWER': 'at'},
             {'POS': 'PROPN', 'OP': '+'}],
            # Company patterns
            [{'ENT_TYPE': 'ORG'}]
        ]
        
        # Add patterns to matcher
        for pattern in exp_patterns:
            self.matcher.add('EXPERIENCE', [pattern])
        
        # Find matches
        matches = self.matcher(doc)
        current_entry = {}
        
        for match_id, start, end in matches:
            span = doc[start:end]
            text = span.text
            
            if 'at' in text:
                parts = text.split('at')
                current_entry['title'] = parts[0].strip()
                current_entry['organization'] = parts[1].strip()
            elif span.ent_type_ == 'ORG':
                current_entry['organization'] = text
            
            if current_entry and len(current_entry) >= 2:
                # Extract responsibilities from surrounding context
                responsibilities = [sent.text for sent in span.sent.doc.sents 
                                  if any(word.like_num for word in sent)]
                
                experience_entries.append({
                    'title': current_entry.get('title', ''),
                    'organization': current_entry.get('organization', ''),
                    'date_range': '',  # Will be extracted from context
                    'responsibilities': responsibilities if responsibilities else []
                })
                current_entry = {}
        
        return experience_entries
    
    def _analyze_resume(self, data: Dict) -> Dict:
        """Analyze resume quality and provide feedback"""
        analysis = {
            'structure': {
                'strengths': [],
                'improvements': [],
                'suggestions': []
            },
            'skills': {
                'strengths': [],
                'improvements': [],
                'suggestions': []
            },
            'education': {
                'strengths': [],
                'improvements': [],
                'suggestions': []
            },
            'experience': {
                'strengths': [],
                'improvements': [],
                'suggestions': []
            }
        }
        
        # Analyze structure
        sections_present = []
        if data.get('skills'):
            sections_present.append('Skills')
        if data.get('education'):
            sections_present.append('Education')
        if data.get('work_experience'):
            sections_present.append('Work Experience')
            
        if len(sections_present) >= 3:
            analysis['structure']['strengths'].append(f"Well-structured resume with {', '.join(sections_present)} sections")
        else:
            missing_sections = set(['Skills', 'Education', 'Work Experience']) - set(sections_present)
            if missing_sections:
                analysis['structure']['improvements'].append(f"Missing key sections: {', '.join(missing_sections)}")
                analysis['structure']['suggestions'].append(f"Add {', '.join(missing_sections).lower()} sections to improve completeness")
        
        # Analyze skills
        skills = data.get('skills', [])
        if skills:
            unique_skills = set(skill.get('name', '').lower() for skill in skills if skill.get('name'))
            if len(unique_skills) >= 8:
                analysis['skills']['strengths'].append(f"Strong skill set with {len(unique_skills)} unique skills")
            else:
                analysis['skills']['improvements'].append("Limited variety of skills")
                analysis['skills']['suggestions'].append("Add more diverse technical and soft skills relevant to your target role")
        
        # Analyze education
        education = data.get('education', [])
        if education:
            recent_education = max((edu for edu in education if edu.get('end_date')), 
                                 key=lambda x: x.get('end_date', ''), 
                                 default=None)
            if recent_education:
                analysis['education']['strengths'].append(f"Recent education: {recent_education.get('degree', '')} from {recent_education.get('school', '')}")
            if any(edu.get('gpa') for edu in education):
                analysis['education']['strengths'].append("Academic performance metrics included")
        else:
            analysis['education']['improvements'].append("Education section needs enhancement")
            analysis['education']['suggestions'].append("Add your educational background with degrees, institutions, and graduation dates")
        
        # Analyze experience
        experience = data.get('work_experience', [])
        if experience:
            recent_jobs = sorted((exp for exp in experience if exp.get('end_date')), 
                               key=lambda x: x.get('end_date', ''),
                               reverse=True)[:2]
            
            if recent_jobs:
                analysis['experience']['strengths'].append(f"Recent experience as {recent_jobs[0].get('title', '')}")
                if len(recent_jobs) > 1:
                    analysis['experience']['strengths'].append(f"Diverse experience with multiple roles including {recent_jobs[1].get('title', '')}")
                
            detailed_descriptions = sum(1 for exp in experience 
                                      if exp.get('description') and 
                                      len(exp.get('description', '').split('\n')) >= 3)
            
            if detailed_descriptions >= len(experience):
                analysis['experience']['strengths'].append("Comprehensive job descriptions with achievements")
            else:
                analysis['experience']['improvements'].append("Some job descriptions lack detail")
                analysis['experience']['suggestions'].append("Expand job descriptions with specific achievements and responsibilities")
        else:
            analysis['experience']['improvements'].append("Work experience section is missing or empty")
            analysis['experience']['suggestions'].append("Add your work history with detailed responsibilities and achievements")
        
        # Add ATS compatibility analysis
        ats_analysis = self._analyze_ats_compatibility(data)
        analysis['ats_compatibility'] = ats_analysis
        
        return analysis
    
    def _analyze_ats_compatibility(self, data: Dict) -> Dict:
        """Analyze resume for ATS compatibility"""
        ats_analysis = {
            'strengths': [],
            'improvements': [],
            'suggestions': []
        }
        
        # Check for basic ATS requirements
        if data.get('skills') and data.get('education') and data.get('work_experience'):
            ats_analysis['strengths'].append("Resume contains all essential sections for ATS scanning")
        
        # Check for detailed work experience
        experience = data.get('work_experience', [])
        if experience:
            has_dates = all(exp.get('start_date') and exp.get('end_date') for exp in experience)
            has_titles = all(exp.get('title') for exp in experience)
            
            if has_dates and has_titles:
                ats_analysis['strengths'].append("Work experience is well-structured with clear dates and titles")
            else:
                ats_analysis['improvements'].append("Some work experience entries missing dates or titles")
                ats_analysis['suggestions'].append("Ensure all work experience entries have clear dates and job titles")
        
        # Check for skills formatting
        skills = data.get('skills', [])
        if skills:
            skill_names = [skill.get('name', '').lower() for skill in skills if skill.get('name')]
            if len(skill_names) >= 8:
                ats_analysis['strengths'].append("Good variety of skills that can be detected by ATS")
            else:
                ats_analysis['improvements'].append("Limited number of clearly defined skills")
                ats_analysis['suggestions'].append("Add more relevant skills using standard industry terms")
        
        return ats_analysis
    
    def _get_empty_result(self) -> Dict:
        """Return empty result structure when parsing fails"""
        return {
            'skills': {
                'programming': [],
                'web': [],
                'database': [],
                'cloud': [],
                'tools': [],
                'methodologies': [],
                'analytics': [],
                'soft_skills': [],
                'certifications': []
            },
            'education': [],
            'experience': [],
            'analysis': {
                'structure': {
                    'strengths': [],
                    'improvements': ['Unable to parse resume properly'],
                    'suggestions': ['Please ensure the resume is in a standard format and try again']
                },
                'skills': {
                    'strengths': [],
                    'improvements': ['No skills were identified in the resume'],
                    'suggestions': ['Add relevant technical and soft skills to your resume']
                },
                'education': {
                    'strengths': [],
                    'improvements': ['Education section is missing or could not be parsed'],
                    'suggestions': ['Include your educational background with degrees and institutions']
                },
                'experience': {
                    'strengths': [],
                    'improvements': ['Work experience section is missing or could not be parsed'],
                    'suggestions': ['Add your work history with detailed responsibilities and achievements']
                }
            }
        }