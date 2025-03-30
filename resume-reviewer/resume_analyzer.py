import re
from collections import Counter
import spacy
from keybert import KeyBERT
from sklearn.feature_extraction.text import CountVectorizer
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from sentence_transformers import SentenceTransformer

# Initialize spaCy and KeyBERT models
try:
    nlp = spacy.load('en_core_web_sm')
    keybert_model = KeyBERT()
    print("Successfully loaded spaCy and KeyBERT models")
except Exception as e:
    print(f"Warning: Could not load NLP models: {str(e)}")
    raise

def extract_skills(text):
    # Process text with spaCy
    doc = nlp(text)
    
    # Initialize skill categories
    found_skills = {}
    
    # Define skill categories with expanded technical and soft skills
    skill_categories = {
        'programming': [
            'python', 'java', 'javascript', 'c++', 'ruby', 'php', 'c#', 'swift', 'kotlin', 'go',
            'rust', 'typescript', 'scala', 'perl', 'r', 'matlab', 'bash', 'shell', 'powershell',
            'vba', 'dart', 'groovy', 'lua', 'objective-c', 'assembly', 'cobol', 'fortran', 'haskell',
            'erlang', 'clojure', 'f#', 'julia', 'prolog', 'scheme', 'lisp', 'ada', 'pascal',
            'visual basic', 'delphi', 'abap', 'pl/sql', 't-sql', 'apex', 'solidity'
        ],
        'web': [
            'html', 'css', 'react', 'angular', 'vue', 'node.js', 'tailwind', 'reactjs', 'vite',
            'next.js', 'express', 'django', 'flask', 'bootstrap', 'jquery', 'sass', 'less', 'webpack',
            'gatsby', 'svelte', 'nuxt.js', 'ember', 'backbone', 'graphql', 'rest', 'soap', 'pwa',
            'web components', 'webgl', 'three.js', 'web3', 'dapp', 'meteor', 'laravel', 'spring',
            'asp.net', 'ruby on rails', 'symfony', 'fastapi', 'nestjs', 'strapi', 'remix',
            'web assembly', 'web sockets', 'web workers', 'service workers', 'progressive web apps'
        ],
        'database': [
            'sql', 'mongodb', 'postgresql', 'mysql', 'redis', 'nosql', 'key-value store',
            'firebase', 'dynamodb', 'cassandra', 'oracle', 'sqlite', 'mariadb', 'elasticsearch',
            'neo4j', 'couchdb', 'influxdb', 'hbase', 'cockroachdb', 'rethinkdb', 'realm',
            'supabase', 'planetscale', 'timescaledb', 'clickhouse', 'snowflake', 'bigquery',
            'redshift', 'data warehousing', 'etl', 'data modeling', 'database design',
            'database administration', 'data migration', 'data integration'
        ],
        'cloud': [
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'api', 'rest api', 'authentication',
            'serverless', 'microservices', 'ci/cd', 'heroku', 'digitalocean', 'openshift',
            'cloud foundry', 'terraform', 'ansible', 'puppet', 'chef', 'jenkins', 'gitlab ci',
            'github actions', 'circleci', 'travis ci', 'aws lambda', 'azure functions',
            'google cloud functions', 'cloud run', 'eks', 'aks', 'gke', 'cloud native',
            'infrastructure as code', 'platform as a service', 'software as a service'
        ],
        'tools': [
            'git', 'jenkins', 'jira', 'confluence', 'github', 'gitlab', 'bitbucket', 'terraform',
            'ansible', 'puppet', 'chef', 'docker', 'kubernetes', 'vagrant', 'prometheus', 'grafana',
            'elk stack', 'splunk', 'newrelic', 'datadog', 'sonarqube', 'selenium', 'postman',
            'swagger', 'vscode', 'intellij', 'eclipse', 'android studio', 'xcode', 'vim',
            'emacs', 'sublime text', 'atom', 'visual studio', 'jupyter', 'rstudio',
            'tableau desktop', 'power bi desktop', 'adobe creative suite'
        ],
        'methodologies': [
            'agile', 'scrum', 'kanban', 'waterfall', 'problem solving', 'analytical thinking',
            'tdd', 'bdd', 'devops', 'lean', 'six sigma', 'prince2', 'itil', 'togaf', 'safe',
            'extreme programming', 'crystal', 'fdd', 'rad', 'spiral model', 'design thinking',
            'systems thinking', 'object oriented design', 'functional programming',
            'domain driven design', 'behavior driven development', 'test driven development',
            'continuous integration', 'continuous delivery', 'continuous deployment'
        ],
        'analytics': [
            'machine learning', 'data analysis', 'statistics', 'tableau', 'numpy', 'pandas',
            'scikit-learn', 'matplotlib', 'logistic regression', 'random forest', 'mlflow',
            'tensorflow', 'pytorch', 'keras', 'deep learning', 'nlp', 'computer vision',
            'reinforcement learning', 'time series analysis', 'a/b testing', 'data mining',
            'big data', 'hadoop', 'spark', 'power bi', 'looker', 'data visualization',
            'predictive analytics', 'prescriptive analytics', 'descriptive analytics',
            'statistical analysis', 'business intelligence', 'data warehousing'
        ],
        'soft_skills': [
            'leadership', 'communication', 'teamwork', 'problem solving', 'critical thinking',
            'time management', 'project management', 'presentation', 'negotiation',
            'conflict resolution', 'adaptability', 'creativity', 'attention to detail',
            'organization', 'mentoring', 'customer service', 'decision making',
            'emotional intelligence', 'strategic thinking', 'analytical thinking',
            'interpersonal skills', 'collaboration', 'cultural awareness', 'empathy',
            'active listening', 'written communication', 'verbal communication',
            'public speaking', 'team building', 'coaching', 'mentoring'
        ],
        'certifications': [
            'aws certified', 'azure certified', 'google certified', 'cisco certified', 'comptia',
            'pmp', 'scrum', 'itil', 'cka', 'ckad', 'ceh', 'cissp', 'security+', 'network+', 'a+',
            'rhce', 'mcse', 'oracle certified', 'salesforce certified', 'sap certified',
            'vmware certified', 'istqb', 'prince2', 'six sigma', 'cism', 'cisa', 'ccna', 'ccnp',
            'aws solutions architect', 'aws developer', 'aws sysops', 'azure administrator',
            'azure developer', 'google cloud architect', 'google cloud developer'
        ]
    }
    
    # Process text with spaCy
    doc = nlp(text)
    
    # Extract skills using enhanced NER and pattern matching with improved accuracy
    found_skills = {category: [] for category in skill_categories}
    
    # Use KeyBERT for keyword extraction with optimized parameters
    keywords = keybert_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 3),  # Increased to catch longer skill phrases
        stop_words='english',
        use_maxsum=True,
        nr_candidates=50,  # Increased for better coverage
        top_n=25,  # Increased to capture more potential skills
        diversity=0.7  # Added to ensure diverse skill extraction
    )
    
    # Extract skills from keywords using traditional approach
    for keyword, _ in keywords:
        keyword_lower = keyword.lower()
        for category, skills in skill_categories.items():
            if any(skill in keyword_lower for skill in skills):
                found_skills[category].append(keyword)
    
    # Use Hugging Face transformers for enhanced skill extraction if available
    if sentence_model is not None:
        try:
            # Create embeddings for all skills and the resume text chunks
            all_skills = [skill for category_skills in skill_categories.values() for skill in category_skills]
            
            # Split text into chunks to process with transformer
            chunks = [sent.text for sent in doc.sents]
            
            # Get embeddings
            skill_embeddings = sentence_model.encode(all_skills)
            chunk_embeddings = sentence_model.encode(chunks)
            
            # Find semantic matches between skills and text chunks
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np
            
            # For each chunk, find the most similar skills
            for chunk_idx, chunk_embedding in enumerate(chunk_embeddings):
                similarities = cosine_similarity(
                    chunk_embedding.reshape(1, -1),
                    skill_embeddings
                )[0]
                
                # Get top matches
                top_indices = np.argsort(similarities)[-5:]  # Top 5 matches
                
                # Add matched skills to found_skills
                for idx in top_indices:
                    if similarities[idx] > 0.5:  # Similarity threshold
                        matched_skill = all_skills[idx]
                        for category, skills in skill_categories.items():
                            if matched_skill in skills:
                                if matched_skill not in found_skills[category]:
                                    found_skills[category].append(matched_skill)
        except Exception as e:
            print(f"Error in transformer-based skill extraction: {str(e)}")
    
    # Use Hugging Face NER for additional entity extraction if available
    if ner_pipeline is not None:
        try:
            # Extract entities using the NER pipeline
            entities = ner_pipeline(text)
            
            # Look for skill-related entities
            for entity in entities:
                entity_text = entity['word'].lower()
                # Check if entity matches any skill
                for category, skills in skill_categories.items():
                    if any(skill in entity_text for skill in skills) and entity_text not in found_skills[category]:
                        found_skills[category].append(entity_text)
        except Exception as e:
            print(f"Error in NER-based skill extraction: {str(e)}")
    
    # Extract additional skills using spaCy's pattern matching (traditional approach)
    for token in doc:
        token_text = token.text.lower()
        for category, skills in skill_categories.items():
            if token_text in skills and token_text not in found_skills[category]:
                found_skills[category].append(token_text)
    
    return found_skills

def analyze_education(text):
    # Process text with spaCy
    doc = nlp(text)
    
    education_info = []
    current_entry = {}
    
    # Enhanced education-related patterns with more variations
    education_patterns = {
        'degree': [
            # Higher Education
            'bachelor', 'master', 'phd', 'doctorate', 'post graduate', 'undergraduate', 'graduate',
            # Engineering
            'b\.tech', 'btech', 'm\.tech', 'mtech', 'be', 'b\.e', 'me', 'm\.e',
            # Science and Commerce
            'b\.sc', 'bsc', 'm\.sc', 'msc', 'b\.com', 'bcom', 'm\.com', 'mcom',
            # Business and Management
            'bba', 'mba', 'pgdm', 'executive',
            # School Education
            'intermediate', '10th', 'tenth', '12th', 'twelfth', 'high school',
            'secondary', 'higher secondary', 'diploma', 'ssc', 'hsc', 'cbse', 'icse'
        ],
        'field': [
            # Engineering Fields
            'computer science', 'engineering', 'electronics', 'communication',
            'electronics and communication', 'ece', 'cse', 'it', 'information technology',
            'mechanical', 'civil', 'electrical', 'chemical', 'biotechnology', 'aerospace',
            # Science Fields
            'science', 'physics', 'chemistry', 'mathematics', 'biology',
            # Business Fields
            'business', 'management', 'finance', 'marketing', 'human resources',
            'operations', 'supply chain', 'economics', 'accounting',
            # Modern Fields
            'data science', 'artificial intelligence', 'machine learning', 'robotics',
            'cyber security', 'cloud computing', 'blockchain', 'iot',
            # Other Fields
            'arts', 'commerce', 'humanities', 'social sciences', 'design',
            'architecture', 'media', 'journalism'
        ],
        'institution': [
            'university', 'college', 'institute', 'school', 'academy',
            'board of education', 'state board', 'international school',
            'polytechnic', 'deemed university', 'autonomous', 'iit', 'nit',
            'bits', 'vit', 'mit', 'central board', 'state board'
        ]
    }
    
    # Enhanced education information extraction
    for sent in doc.sents:
        sent_text = sent.text.lower()
        
        # Check for education patterns with improved pattern matching
        for category, patterns in education_patterns.items():
            for pattern in patterns:
                pattern_regex = f'\\b{pattern}\\b'
                if re.search(pattern_regex, sent_text, re.IGNORECASE):
                    # Extract the full phrase containing the education information
                    for chunk in sent.noun_chunks:
                        chunk_text = chunk.text.lower()
                        if re.search(pattern_regex, chunk_text, re.IGNORECASE):
                            if category == 'degree' and not current_entry.get('degree'):
                                # Look for degree with specialization
                                degree_context = re.search(
                                    f'([^.]*{pattern}[^.]*(?:in|of|with specialization in)?[^.]*)',
                                    sent_text,
                                    re.IGNORECASE
                                )
                                if degree_context:
                                    current_entry['degree'] = degree_context.group(1).strip()
                                else:
                                    current_entry['degree'] = chunk.text.strip()
                            
                            elif category == 'field' and not current_entry.get('field'):
                                # Look for field with broader context
                                field_context = re.search(
                                    f'([^.]*{pattern}[^.]*)',
                                    sent_text,
                                    re.IGNORECASE
                                )
                                if field_context:
                                    current_entry['field'] = field_context.group(1).strip()
                                else:
                                    current_entry['field'] = chunk.text.strip()
                            
                            elif category == 'institution' and not current_entry.get('institution'):
                                current_entry['institution'] = chunk.text.strip()
        
        # Extract dates with improved pattern matching
        date_patterns = [
            r'(\d{4})\s*-\s*(\d{4}|present|ongoing)',  # Year range
            r'(\d{4})\s*to\s*(\d{4}|present|ongoing)',   # Year with 'to'
            r'(\d{4})\s*–\s*(\d{4}|present|ongoing)',   # Year with en dash
            r'(\d{2}/\d{2})\s*-\s*(\d{2}/\d{2})',      # MM/YY format
            r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s*\d{4}'  # Month Year
        ]
        
        for pattern in date_patterns:
            date_match = re.search(pattern, sent_text, re.IGNORECASE)
            if date_match and not current_entry.get('date'):
                current_entry['date'] = date_match.group(0)
                break
        
        # Extract academic performance metrics with enhanced patterns
        performance_patterns = {
            'cgpa': [
                r'\bcgpa\s*[:-]?\s*(\d+\.\d+)\b',
                r'\bgpa\s*[:-]?\s*(\d+\.\d+)\b',
                r'\bcgpa\s*of\s*(\d+\.\d+)\b'
            ],
            'percentage': [
                r'\b(\d{2,3})\s*%',
                r'\bpercentage\s*[:-]?\s*(\d{2,3}(?:\.\d+)?)\b',
                r'\bpercentile\s*[:-]?\s*(\d{2,3}(?:\.\d+)?)\b',
                r'\bmarks\s*[:-]?\s*(\d{2,3}(?:\.\d+)?)\b'
            ],
            'grade': [
                r'\bgrade\s*[:-]?\s*([A-Z][+-]?)\b',
                r'\bgrade\s*point\s*[:-]?\s*(\d+\.\d+)\b'
            ]
        }
        
        for metric, patterns in performance_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, sent_text, re.IGNORECASE)
                if match and not current_entry.get(metric):
                    current_entry[metric] = match.group(1)
        
        # Extract achievements and honors
        achievement_patterns = [
            r'\b(gold\s*medal|silver\s*medal|bronze\s*medal)\b',
            r'\b(first|second|third)\s*rank\b',
            r'\b(distinction|merit|honors|honours)\b',
            r'\b(scholarship|award|fellowship)\b'
        ]
        
        for pattern in achievement_patterns:
            achievement_match = re.search(pattern, sent_text, re.IGNORECASE)
            if achievement_match:
                if 'achievements' not in current_entry:
                    current_entry['achievements'] = []
                current_entry['achievements'].append(achievement_match.group(0))
        
        # If we have found a complete entry, add it to the list
        if len(current_entry) >= 2:  # At least 2 fields should be present
            # Sort achievements if present
            if 'achievements' in current_entry:
                current_entry['achievements'] = sorted(set(current_entry['achievements']))
            
            education_info.append(dict(current_entry))
            current_entry = {}
    
    return education_info

def analyze_experience(text):
    # Process text with spaCy
    doc = nlp(text)
    
    experience_entries = []
    current_entry = {}
    
    # Define patterns for experience extraction
    job_title_patterns = [
        # Technical roles
        'engineer', 'developer', 'architect', 'programmer', 'coder', 'technician',
        'administrator', 'analyst', 'scientist', 'researcher', 'specialist', 'consultant',
        # Management roles
        'manager', 'lead', 'head', 'director', 'supervisor', 'coordinator',
        'chief', 'officer', 'executive', 'president', 'vp', 'vice president',
        # C-level roles
        'ceo', 'cto', 'cio', 'cfo', 'coo', 'cdo', 'cmo',
        # Other roles
        'designer', 'strategist', 'planner', 'associate', 'assistant', 'advisor',
        'intern', 'trainee', 'graduate', 'junior', 'senior', 'principal', 'staff',
        'founder', 'co-founder', 'owner', 'partner', 'contractor', 'freelancer'
    ]
    
    # Role prefixes and suffixes for better title detection
    role_prefixes = ['senior', 'junior', 'lead', 'principal', 'staff', 'chief', 'head of', 'director of', 'vp of']
    role_suffixes = ['engineer', 'developer', 'architect', 'manager', 'analyst', 'consultant', 'specialist']
    
    # Improved date pattern for experience entries
    date_pattern = r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\s*(?:-|to|–)\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}|\d{4}\s*(?:-|to|–)\s*(?:\d{4}|present|current|ongoing))'
    
    for sent in doc.sents:
        sent_text = sent.text
        sent_lower = sent_text.lower()
        
        # Look for date ranges with enhanced pattern matching
        date_match = re.search(date_pattern, sent_lower)
        
        # Extract job titles using role patterns
        for title_pattern in job_title_patterns:
            if title_pattern in sent_lower:
                # Check for role prefixes and suffixes
                for prefix in role_prefixes:
                    prefix_match = re.search(f'{prefix}\s+\w+\s+{title_pattern}', sent_lower)
                    if prefix_match and not current_entry.get('title'):
                        current_entry['title'] = sent_text[prefix_match.start():prefix_match.end()].strip()
                        break
                
                # If no prefix match found, look for basic title
                if not current_entry.get('title'):
                    title_match = re.search(f'\w+\s+{title_pattern}|{title_pattern}\s+\w+', sent_lower)
                    if title_match:
                        current_entry['title'] = sent_text[title_match.start():title_match.end()].strip()
        
        # Extract organization names using NER and pattern matching
        org_patterns = ['inc', 'corp', 'corporation', 'ltd', 'limited', 'llc', 'company', 'technologies', 'solutions', 'systems', 'group', 'labs']
        for ent in sent.ents:
            if ent.label_ == 'ORG' and not current_entry.get('organization'):
                current_entry['organization'] = ent.text.strip()
            elif any(pattern in ent.text.lower() for pattern in org_patterns) and not current_entry.get('organization'):
                current_entry['organization'] = ent.text.strip()
        
        if date_match:
            if current_entry and 'title' in current_entry:
                experience_entries.append(dict(current_entry))
                current_entry = {}
            current_entry['date_range'] = date_match.group(0)
        
        # Extract responsibilities
        if not current_entry.get('responsibilities'):
            current_entry['responsibilities'] = []
        
        # Look for bullet points or numbered lists
        responsibility_match = re.search(r'[•\-\*]\s*(.+)', sent_text)
        if responsibility_match:
            current_entry['responsibilities'].append(responsibility_match.group(1).strip())
        elif any(verb.dep_ == 'ROOT' for verb in sent):
            current_entry['responsibilities'].append(sent_text.strip())
    
    # Add the last entry if it exists
    if current_entry and ('title' in current_entry or 'organization' in current_entry):
        experience_entries.append(dict(current_entry))
    
    return experience_entries

    # Look for project titles and technical roles
    project_keywords = ['app', 'application', 'system', 'platform', 'builder', 'clone', 'analyzer', 'reviewer', 'dashboard']
    if not current_entry.get('title') and any(keyword in sent_text.lower() for keyword in project_keywords):
        # Extract potential project title
        for chunk in sent.noun_chunks:
            if any(keyword in chunk.text.lower() for keyword in project_keywords):
                current_entry['title'] = chunk.text.strip()
                break
        
    # Extract responsibilities and achievements
    if current_entry and 'date_range' in current_entry:
        if not current_entry.get('responsibilities'):
            current_entry['responsibilities'] = []
            
            # Expanded action verbs for comprehensive responsibility detection
            action_verbs = ['developed', 'managed', 'led', 'created', 'implemented', 'designed', 'architected', 'built', 'deployed', 'optimized', 'improved', 'enhanced', 'coordinated', 'supervised', 'mentored', 'trained', 'analyzed', 'researched', 'solved', 'troubleshot', 'debugged', 'maintained', 'automated', 'streamlined', 'integrated', 'collaborated', 'initiated', 'spearheaded', 'orchestrated', 'achieved', 'reduced', 'increased', 'generated', 'delivered', 'established']
            if any(verb in sent_text.lower() for verb in action_verbs):
                current_entry['responsibilities'].append(sent_text.strip())
    
    # Add the last entry if it exists
    if current_entry and ('title' in current_entry or 'organization' in current_entry):
        # Clean up and format the entry
        if 'responsibilities' in current_entry:
            # Remove duplicates while preserving order
            current_entry['responsibilities'] = list(dict.fromkeys(current_entry['responsibilities']))
            
            # Identify achievements within responsibilities
            achievements = []
            responsibilities = []
            
            for resp in current_entry['responsibilities']:
                # Check for metrics and impact statements
                if any(pattern in resp.lower() for pattern in ['increased', 'decreased', 'reduced', 'improved', 'achieved', 'led', 'managed', 'developed']) and \
                   any(char.isdigit() for char in resp):
                    achievements.append(resp)
                else:
                    responsibilities.append(resp)
            
            current_entry['responsibilities'] = responsibilities
            if achievements:
                current_entry['achievements'] = achievements
        
        experience_entries.append(dict(current_entry))
    
    # Sort entries by date (most recent first) if possible
    try:
        experience_entries.sort(key=lambda x: '9999' if 'present' in x.get('date_range', '').lower() \
                               else re.findall(r'\d{4}', x.get('date_range', ''))[-1], reverse=True)
    except:
        pass
    
    return experience_entries

def analyze_resume_quality(text, skills_found, education_entries, experience_entries):
    # Process text with spaCy for additional insights
    doc = nlp(text)
    analysis = {}
    
    # Structure and Formatting Analysis with enhanced evaluation criteria
    structure_analysis = {
        'strengths': [],
        'improvements': [],
        'suggestions': [],
        'metrics': {
            'content_score': 0,
            'format_score': 0,
            'completeness_score': 0,
            'overall_score': 0
        }
    }
    
    # Use Hugging Face sentiment analysis if available
    if sentiment_analyzer is not None:
        try:
            # Split text into chunks to process with transformer (due to token limits)
            chunks = [sent.text for sent in doc.sents]
            
            # Process chunks in batches to avoid token limits
            batch_size = 5
            sentiment_scores = []
            
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i+batch_size]
                # Join batch with spaces to create a coherent text
                batch_text = ' '.join(batch)
                if batch_text.strip():
                    result = sentiment_analyzer(batch_text)[0]
                    sentiment_scores.append(result)
            
            # Calculate overall sentiment
            positive_count = sum(1 for score in sentiment_scores if score['label'] == 'POSITIVE')
            total_count = len(sentiment_scores) if sentiment_scores else 1
            positive_ratio = positive_count / total_count
            
            # Add sentiment analysis to structure analysis
            if positive_ratio >= 0.7:
                structure_analysis['strengths'].append('Strong positive tone throughout the resume')
            elif positive_ratio >= 0.5:
                structure_analysis['strengths'].append('Balanced professional tone in the resume')
            else:
                structure_analysis['improvements'].append('Resume tone could be more positive and achievement-oriented')
                structure_analysis['suggestions'].append('Use more positive action verbs and highlight achievements')
        except Exception as e:
            print(f"Error in sentiment analysis: {str(e)}")
    
    # Check for contact information using NER
    contact_entities = [ent.text for ent in doc.ents if ent.label_ in ['PERSON', 'EMAIL', 'PHONE']]
    if contact_entities:
        structure_analysis['strengths'].append('Contact information is present and properly formatted')
        if not any(re.match(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}', ent) for ent in contact_entities):
            structure_analysis['improvements'].append('Email address format could be improved')
            structure_analysis['suggestions'].append('Ensure email address follows standard format (example@domain.com)')
    else:
        structure_analysis['improvements'].append('Missing or improperly formatted contact information')
        structure_analysis['suggestions'].append('Add complete contact information including email, phone, and location')
    
    # Analyze document structure using NLP
    sentences = list(doc.sents)
    first_sentence = next(doc.sents).text.lower()
    
    if any(keyword in first_sentence for keyword in ['summary', 'objective', 'profile']):
        structure_analysis['strengths'].append('Begins with a clear professional summary/objective')
    else:
        structure_analysis['improvements'].append('Missing professional summary or objective')
        structure_analysis['suggestions'].append('Start with a concise professional summary highlighting key qualifications')
    
    # Analyze section organization
    section_headers = [sent.text.strip() for sent in sentences if sent.text.isupper() or 
                      (sent.text.strip().istitle() and len(sent.text.split()) <= 3)]
    
    if len(section_headers) >= 3:
        structure_analysis['strengths'].append('Clear section organization with proper headers')
        if len(set(section_headers)) == len(section_headers):
            structure_analysis['strengths'].append('Logical section flow and organization')
    else:
        structure_analysis['improvements'].append('Limited or unclear section organization')
        structure_analysis['suggestions'].append('Organize content into clear sections with descriptive headers')
    
    analysis['structure'] = structure_analysis
    
    # Enhanced Skills Analysis
    skills_analysis = {
        'identified_skills': skills_found,
        'strengths': [],
        'improvements': [],
        'suggestions': []
    }
    
    total_skills = sum(len(skills) for skills in skills_found.values())
    unique_categories = sum(1 for skills in skills_found.values() if skills)
    
    if total_skills >= 8:
        skills_analysis['strengths'].append('Comprehensive range of technical skills')
        if unique_categories >= 3:
            skills_analysis['strengths'].append('Well-rounded skill set across multiple domains')
    else:
        skills_analysis['improvements'].append('Limited range of skills presented')
        skills_analysis['suggestions'].append('Add more industry-specific technical and soft skills')
    
    # Analyze skill relevance and grouping
    for category, skills in skills_found.items():
        if len(skills) > 0:
            skills_analysis['strengths'].append(f'Strong {category} skills demonstrated')
        elif category in ['programming', 'web', 'cloud']:
            skills_analysis['improvements'].append(f'Consider adding more {category} skills')
            skills_analysis['suggestions'].append(f'Include popular {category} technologies and frameworks')
    
    analysis['skills'] = skills_analysis
    
    # Enhanced Education Analysis
    education_analysis = {
        'entries': education_entries,
        'strengths': [],
        'improvements': [],
        'suggestions': []
    }
    
    if education_entries:
        education_analysis['strengths'].append('Education history is present')
        
        # Analyze education entries quality
        degree_count = sum(1 for entry in education_entries if entry.get('degree'))
        field_count = sum(1 for entry in education_entries if entry.get('field'))
        institution_count = sum(1 for entry in education_entries if entry.get('institution'))
        date_count = sum(1 for entry in education_entries if entry.get('date'))
        
        if len(education_entries) >= 2:
            education_analysis['strengths'].append('Multiple educational qualifications demonstrated')
        
        if degree_count > 0:
            education_analysis['strengths'].append('Degree information clearly stated')
            if field_count == degree_count:
                education_analysis['strengths'].append('Field of study specified for all degrees')
        else:
            education_analysis['improvements'].append('Degree information could be more specific')
            education_analysis['suggestions'].append('Clearly state your degree type (e.g., Bachelor of Science, Master of Arts)')
        
        if institution_count == len(education_entries):
            education_analysis['strengths'].append('Educational institutions clearly listed')
        else:
            education_analysis['improvements'].append('Some educational institutions not specified')
            education_analysis['suggestions'].append('Include the name of each educational institution')
        
        if date_count == len(education_entries):
            education_analysis['strengths'].append('Education timeline well documented')
        else:
            education_analysis['improvements'].append('Missing dates for some education entries')
            education_analysis['suggestions'].append('Add dates for all educational qualifications')
    else:
        education_analysis['improvements'].append('Missing or incomplete education information')
        education_analysis['suggestions'].append('Add your educational background with dates and institutions')
    
    analysis['education'] = education_analysis
    
    # Enhanced Experience Analysis
    experience_analysis = {
        'entries': experience_entries,
        'strengths': [],
        'improvements': [],
        'suggestions': []
    }
    
    if experience_entries:
        experience_analysis['strengths'].append('Work experience is documented')
        
        # Analyze experience entries quality
        entries_with_title = sum(1 for entry in experience_entries if entry.get('title'))
        entries_with_org = sum(1 for entry in experience_entries if entry.get('organization'))
        entries_with_dates = sum(1 for entry in experience_entries if entry.get('date_range'))
        entries_with_resp = sum(1 for entry in experience_entries if entry.get('responsibilities') and len(entry['responsibilities']) >= 2)
        
        if len(experience_entries) >= 2:
            experience_analysis['strengths'].append('Multiple relevant positions demonstrated')
        
        if entries_with_title == len(experience_entries):
            experience_analysis['strengths'].append('Job titles clearly stated')
        else:
            experience_analysis['improvements'].append('Some job titles missing or unclear')
            experience_analysis['suggestions'].append('Ensure each position has a clear job title')
        
        if entries_with_org == len(experience_entries):
            experience_analysis['strengths'].append('Organizations clearly identified')
        else:
            experience_analysis['improvements'].append('Some organization names missing')
            experience_analysis['suggestions'].append('Include organization names for all positions')
        
        if entries_with_resp > 0:
            if all(len(entry.get('responsibilities', [])) >= 3 for entry in experience_entries):
                experience_analysis['strengths'].append('Comprehensive description of responsibilities')
            else:
                experience_analysis['improvements'].append('Some positions lack detailed responsibilities')
                experience_analysis['suggestions'].append('Add 3-5 key responsibilities for each position')
        
        # Analyze action verbs and achievements
        action_verbs_used = set()
        for entry in experience_entries:
            for resp in entry.get('responsibilities', []):
                words = resp.lower().split()
                action_verbs_used.update(word for word in words if word in ['developed', 'managed', 'led', 'created', 'implemented', 'designed', 'achieved', 'improved', 'increased', 'reduced'])
        
        if len(action_verbs_used) >= 5:
            experience_analysis['strengths'].append('Strong use of action verbs to describe achievements')
        else:
            experience_analysis['improvements'].append('Limited use of action verbs')
            experience_analysis['suggestions'].append('Use more action verbs to describe your achievements')
    else:
        experience_analysis['improvements'].append('Limited or missing work experience details')
        experience_analysis['suggestions'].append('Add detailed work experience with achievements and responsibilities')
    
    analysis['experience'] = experience_analysis
    
    # Enhanced ATS Compatibility Analysis
    ats_analysis = {
        'strengths': [],
        'improvements': [],
        'suggestions': []
    }
    
    # Analyze keyword density and distribution
    total_skills = sum(len(skills) for skills in skills_found.values())
    skill_density = total_skills / len(doc)
    
    if total_skills >= 8:
        ats_analysis['strengths'].append('Good keyword optimization for ATS systems')
        if skill_density < 0.05:  # Not too dense with keywords
            ats_analysis['strengths'].append('Natural integration of keywords')
        else:
            ats_analysis['improvements'].append('Keyword density might be too high')
            ats_analysis['suggestions'].append('Ensure keywords flow naturally in the content')
    else:
        ats_analysis['improvements'].append('Limited use of industry-specific keywords')
        ats_analysis['suggestions'].append('Incorporate more relevant keywords from job descriptions')
    
    # Check formatting consistency
    if re.search(r'\d{4}[-–](?:\d{4}|present)', text.lower()):
        ats_analysis['strengths'].append('Consistent date formatting')
    else:
        ats_analysis['improvements'].append('Inconsistent or missing date formats')
        ats_analysis['suggestions'].append('Use consistent date format throughout (e.g., YYYY-YYYY)')
    
    # Check for common ATS-friendly formatting
    if not re.search(r'[\t\f\v]', text):  # No complex formatting
        ats_analysis['strengths'].append('Clean, ATS-friendly formatting')
    else:
        ats_analysis['improvements'].append('Complex formatting may cause ATS issues')
        ats_analysis['suggestions'].append('Use simple, consistent formatting without tables or columns')
    
    # Check section headers
    common_section_headers = ['experience', 'education', 'skills', 'summary', 'objective']
    found_headers = sum(1 for header in common_section_headers if re.search(rf'\b{header}\b', text.lower()))
    
    if found_headers >= 3:
        ats_analysis['strengths'].append('Standard section headers present')
    else:
        ats_analysis['improvements'].append('Non-standard or missing section headers')
        ats_analysis['suggestions'].append('Use standard section headers (e.g., Experience, Education, Skills)')
    
    analysis['ats'] = ats_analysis
    
    return analysis

def analyze_resume_with_ai(resume_text):
    try:
        print("Starting resume analysis with AI...")
        
        # Initialize analysis sections
        structure_analysis = {
            'strengths': [],
            'improvements': [],
            'suggestions': []
        }
        
        # Extract skills with enhanced transformer-based extraction
        print("Extracting skills...")
        skills = extract_skills(resume_text)
        
        # Analyze education
        print("Analyzing education...")
        education = analyze_education(resume_text)
        
        # Analyze experience
        print("Analyzing experience...")
        experience = analyze_experience(resume_text)
        
        # Perform qualitative analysis with transformer-enhanced sentiment analysis
        print("Performing qualitative analysis...")
        analysis = analyze_resume_quality(resume_text, skills, education, experience)
        
        # Enhanced structure analysis
        if len(resume_text.strip()) > 0:
            structure_analysis['strengths'].append('Resume content is present')
            
            # Check for sections with improved pattern matching
            sections = ['education', 'experience', 'skills', 'projects', 'achievements', 'certifications']
            found_sections = [section for section in sections if re.search(rf'\b{section}\b', resume_text.lower())]
            
            if found_sections:
                structure_analysis['strengths'].append(f'Clear section headers present: {", ".join(found_sections)}')
            else:
                structure_analysis['improvements'].append('Missing clear section headers')
                structure_analysis['suggestions'].append('Add clear section headers (Education, Experience, Skills, etc.)')
            
            # Enhanced formatting analysis
            lines = resume_text.split('\n')
            non_empty_lines = [line.strip() for line in lines if line.strip()]
            
            if len(non_empty_lines) > 10:
                structure_analysis['strengths'].append('Content is properly structured with sufficient detail')
                
                # Check for consistent formatting
                bullet_points = sum(1 for line in non_empty_lines if line.strip().startswith(('•', '-', '*')))
                if bullet_points > 5:
                    structure_analysis['strengths'].append('Good use of bullet points for readability')
            else:
                structure_analysis['improvements'].append('Limited content or poor structure')
                structure_analysis['suggestions'].append('Expand content with more details and better organization')
        else:
            structure_analysis['improvements'].append('Resume appears to be empty')
            structure_analysis['suggestions'].append('Ensure the resume file contains content')
        
        # Enhanced skills analysis
        skills_analysis = {
            'identified_skills': [],
            'strengths': [],
            'improvements': [],
            'suggestions': []
        }
        
        # Process identified skills with improved categorization
        technical_categories = ['programming', 'web', 'database', 'cloud', 'tools']
        soft_categories = ['methodologies', 'soft_skills']
        
        technical_skills_count = sum(len(skills[cat]) for cat in technical_categories if cat in skills)
        soft_skills_count = sum(len(skills[cat]) for cat in soft_categories if cat in skills)
        
        for category, skill_list in skills.items():
            if skill_list:
                skills_analysis['identified_skills'].extend(skill_list)
                if len(skill_list) >= 3:
                    skills_analysis['strengths'].append(f'Strong {category} skillset with {len(skill_list)} skills')
                else:
                    skills_analysis['improvements'].append(f'Limited {category} skills')
                    skills_analysis['suggestions'].append(f'Consider adding more relevant {category} skills')
        
        if technical_skills_count >= 8:
            skills_analysis['strengths'].append('Comprehensive technical skill profile')
        if soft_skills_count >= 3:
            skills_analysis['strengths'].append('Good balance of soft skills')
        
        # Enhanced education analysis
        education_analysis = {
            'entries': education if education else [],
            'strengths': [],
            'improvements': [],
            'suggestions': []
        }
        
        if education:
            education_analysis['strengths'].append(f'Education history present with {len(education)} qualifications')
            
            # Check for education details completeness
            complete_entries = sum(1 for entry in education if all(key in entry for key in ['degree', 'field', 'institution', 'date']))
            if complete_entries == len(education):
                education_analysis['strengths'].append('Comprehensive education details provided')
            else:
                education_analysis['improvements'].append('Some education entries missing details')
                education_analysis['suggestions'].append('Include complete details for all educational qualifications')
        else:
            education_analysis['improvements'].append('No education history found')
            education_analysis['suggestions'].append('Add your educational background with complete details')
        
        # Enhanced experience analysis
        experience_analysis = {
            'entries': experience if experience else [],
            'strengths': [],
            'improvements': [],
            'suggestions': []
        }
        
        if experience:
            experience_analysis['strengths'].append(f'Work experience documented with {len(experience)} positions')
            
            # Analyze experience quality
            for entry in experience:
                if entry.get('responsibilities') and len(entry['responsibilities']) >= 3:
                    experience_analysis['strengths'].append(f'Detailed responsibilities for {entry.get("title", "position")}')
                else:
                    experience_analysis['improvements'].append('Some positions lack detailed responsibilities')
                    experience_analysis['suggestions'].append('Add 3-5 specific responsibilities for each position')
        else:
            experience_analysis['improvements'].append('No work experience found')
            experience_analysis['suggestions'].append('Add your work experience with detailed responsibilities')
        
        # Enhanced ATS analysis
        ats_analysis = {
            'strengths': [],
            'improvements': [],
            'suggestions': []
        }
        
        # Comprehensive ATS checks
        keyword_count = len(skills_analysis['identified_skills'])
        if keyword_count >= 15:
            ats_analysis['strengths'].append('Strong keyword optimization')
        elif keyword_count >= 8:
            ats_analysis['strengths'].append('Good keyword presence')
        else:
            ats_analysis['improvements'].append('Limited keyword optimization')
            ats_analysis['suggestions'].append('Add more industry-specific keywords and skills')
        
        # Check for action verbs
        action_verbs = ['developed', 'managed', 'led', 'created', 'implemented', 'designed', 'achieved']
        action_verb_count = sum(1 for verb in action_verbs if verb in resume_text.lower())
        if action_verb_count >= 5:
            ats_analysis['strengths'].append('Good use of action verbs')
        else:
            ats_analysis['improvements'].append('Limited use of action verbs')
            ats_analysis['suggestions'].append('Use more action verbs to describe achievements and responsibilities')
        
        # Return enhanced analysis
        result = {
            'structure': structure_analysis,
            'skills': skills_analysis,
            'education': education_analysis,
            'experience': experience_analysis,
            'ats': ats_analysis
        }
        
        print("Resume analysis completed successfully.")
        return result
    except Exception as e:
        print(f"Resume Analysis Error: {str(e)}")
        return f"Error: Unable to analyze resume. {str(e)}"