import os
import pickle
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import webbrowser
import base64
import pandas as pd
import re
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from googleapiclient.errors import HttpError
from googletrans import Translator
from flask import Flask, render_template, request, redirect, url_for,session
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Load T5 model and tokenizer
model_name = "t5-small" #Or any other t5 model you want to use
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device) # Move model to GPU if available

def get_gmail_service():
    """Authenticates with Gmail API and returns the Gmail service object."""
    creds = None

    # Load token if it exists
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)

    # If no token is present, get new authentication
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
               'credentials.json', SCOPES)

            auth_url, _ = flow.authorization_url(prompt='consent')
            # Store authentication in session
            session['auth_url'] = auth_url
            return None  # Redirect to authentication page


        # Save the credentials for future runs
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    service = build('gmail', 'v1', credentials=creds)
    return service


def get_message_content(service, message_id, user_id='me'):
    message = service.users().messages().get(userId=user_id, id=message_id, format='full').execute()
    payload = message['payload']
    content = "no text content"

    if 'parts' in payload:
        parts = payload['parts']
        for part in parts:
            if part['mimeType'] == 'text/plain':
                if 'data' in part['body']:
                    data = part['body']['data']
                    try:
                        decoded_data = base64.urlsafe_b64decode(data).decode('utf-8', 'ignore')
                        content = decoded_data
                        break
                    except Exception as e:
                        print(f"Error decoding text/plain: {e}")

            elif part['mimeType'] == 'text/html':
                if 'data' in part['body']:
                    data = part['body']['data']
                    try:
                        decoded_data = base64.urlsafe_b64decode(data).decode('utf-8', 'ignore')
                        soup = BeautifulSoup(decoded_data, 'html.parser')
                        content = soup.get_text(separator=' ', strip=True)
                        break
                    except Exception as e:
                        print(f"Error decoding text/html: {e}")

    elif 'body' in payload and 'data' in payload['body']:
        data = payload['body']['data']
        try:
            decoded_data = base64.urlsafe_b64decode(data).decode('utf-8', 'ignore')
            content = decoded_data
        except Exception as e:
            print(f"Error decoding data: {e}")

    headers = {}
    if 'headers' in payload:
        for header in payload['headers']:
           headers[header['name']] = header['value']

    subject = headers.get('Subject',"")
    sender = headers.get('From',"")

    return  {'message_id': message_id, 'subject': subject, 'sender':sender, 'content': content}



def fetch_filtered_emails(service, user_id='me', query='', max_results=5):
    """Fetches emails from the inbox that match the given query."""

    email_list = []
    try:
        results = service.users().messages().list(userId=user_id, q=query, maxResults=max_results).execute()
        messages = results.get('messages', [])

        for message in messages:
            email_data = get_message_content(service, message['id'], user_id)
            email_list.append(email_data)
    except HttpError as error:
        print(f"An error occurred: {error}")
    return email_list

job_names = [
    "Software Engineer",
    "Data Scientist",
    "Product Manager",
    "UI/UX Designer",
    "Cloud Architect",
    "AIML Engineer",
    "Full-Stack Developer",
    "DevOps Engineer",
    "Cybersecurity Analyst",
    "Business Analyst",
    "Digital Marketing Specialist",
    "Sales Executive",
    "Human Resources Manager",
    "Mechanical Engineer",
    "Electrical Engineer",
    "Civil Engineer",
    "Content Writer",
    "Graphic Designer",
    "Project Manager",
    "Data Analyst",
    "Customer Support Representative",
    "Network Administrator",
    "Quality Assurance Engineer",
    "Financial Analyst",
    "Research Scientist",
    "Operations Manager",
    "Game Developer",
    "Database Administrator",
    "SEO Specialist",
    "Marketing Manager"
]


def extract_insights(text):
    job_history_keywords = [
        "previous role", "past experience", "employment history", "worked at", "responsibilities included",
        "managed", "led", "achieved", "past job", "former position", "roles at", "job position", "employment", "experience",
        "during the time", "in the role of", "before this", "after that", "prior to that", "during my tenure at","fresher"
    ]
    skills_keywords = [
        "proficient in", "experienced with", "knowledge of", "familiar with", "expertise in", "ability to", "skills in", "skill set includes",
        "strong in", "competent in", "versed in", "trained in", "adept at", "mastery of", "specialized in",
        "technical skill", "management skill", "interpersonal skill" , "soft skill", "hard skill"
    ]
    training_keywords = [
        "training program", "certification", "coursework", "workshop", "seminar", "online course", "training in", "training on",
        "certified in", "completed", "attended", "bootcamp", "learning program", "educational program", "studied", "degree in", "certificate in"
    ]
    feedback_keywords = [
        "feedback on", "performance review", "client satisfaction", "positive review", "negative review", "constructive criticism",
        "improvement area", "areas for development", "comments on", "evaluation of", "rated as", "scored", "recommendation", "testimonial",
        "praise for", "critique of", "input on", "suggested that" , "response to feedback"
    ]
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    tokens = word_tokenize(text.lower())
    filtered_tokens = [lemmatizer.lemmatize(w) for w in tokens if not w in stop_words and w.isalpha()]
    text = " ".join(filtered_tokens)

    job_history_pattern = r"(?:(?:previous|past|former)\s+role:|employment(?:ed)?\s+history:|worked\s+at|responsibilities\s+included|managed|led|achieved|during\s+my\s+tenure\s+at|in\s+the\s+role\s+of|before\s+this|after\s+that|prior\s+to\s+that)(.?\.)(?=\s(?:(?:previous|past|former)\s+role:|employment(?:ed)?\s+history:|worked\s+at|responsibilities\s+included|managed|led|achieved|during\s+my\s+tenure\s+at|in\s+the\s+role\s+of|before\s+this|after\s+that|prior\s+to\s+that)|$)"

    skills_pattern = r"(?:(?:proficient|experienced|knowledge|familiar|expertise)\s+in|ability\s+to|skills\s+in|strong\s+in|competent\s+in|versed\s+in|trained\s+in|adept\s+at|mastery\s+of|specialized\s+in)(.?)(?=\s(?:(?:proficient|experienced|knowledge|familiar|expertise)\s+in|ability\s+to|skills\s+in|strong\s+in|competent\s+in|versed\s+in|trained\s+in|adept\s+at|mastery\s+of|specialized\s+in)|$)"

    training_pattern = r"(?:(?:training|certification|coursework|workshop|seminar|online course)\s+in|certified\s+in|completed|attended|bootcamp|learning\s+program|educational\s+program|studied|degree\s+in|certificate\s+in)(.?)(?=\s(?:(?:training|certification|coursework|workshop|seminar|online course)\s+in|certified\s+in|completed|attended|bootcamp|learning\s+program|educational\s+program|studied|degree\s+in|certificate\s+in)|$)"

    feedback_pattern = r"(?:(?:feedback|review|client satisfaction|positive review|negative review|constructive criticism)\s+on|improvement\s+area|areas\s+for\s+development|comments\s+on|evaluation\s+of|rated\s+as|scored|recommendation|testimonial|praise\s+for|critique\s+of|input\s+on|suggested\s+that|response\s+to\s+feedback)(.?)(?=\s(?:(?:feedback|review|client satisfaction|positive review|negative review|constructive criticism)\s+on|improvement\s+area|areas\s+for\s+development|comments\s+on|evaluation\s+of|rated\s+as|scored|recommendation|testimonial|praise\s+for|critique\s+of|input\s+on|suggested\s+that|response\s+to\s+feedback)|$)"

    extracted_jobs = []
    for job in job_names:
        if re.search(r'\b' + re.escape(job) + r'\b',text, re.IGNORECASE):
          extracted_jobs.append(job)

    job_history = re.findall(job_history_pattern, text, re.IGNORECASE)
    skills = re.findall(skills_pattern, text, re.IGNORECASE)
    training = re.findall(training_pattern, text, re.IGNORECASE)
    feedback = re.findall(feedback_pattern, text, re.IGNORECASE)

    return {
        "jobs": extracted_jobs,
        "job_history": job_history,
        "skills": skills,
        "training": training,
        "feedback": feedback,
    }


# Simple database of resources (can be improved for real application)

online_resources = {
    # Technical Domains
    "data analyst": {
        "courses": [
            {"name": "IBM Data Analyst", "link": "https://www.coursera.org/professional-certificates/ibm-data-analyst", "platform": "Coursera"},
            {"name": "Excel to MySQL: Analytic Techniques for Business Specialization", "link": "https://www.coursera.org/specializations/excel-mysql", "platform": "Coursera"},
            {"name": "Data Analysis with Python", "link": "https://www.udemy.com/course/data-analysis-with-pandas/", "platform": "Udemy"},
            {"name": "Data Analysis and Visualization Fundamentals", "link": "https://www.edx.org/certificates/professional-certificate/ibm-data-analysis-and-visualization-fundamentals", "platform": "edX"}
        ],
        "videos": [
            {"name": "Data Analyst Full Course", "link": "https://www.youtube.com/playlist?list=PLOWRNl6YgsT79ezWdEhOjvK4D-cQfr7ys", "platform": "YouTube"},
            {"name": "Introduction to Data Analysis", "link": "https://www.youtube.com/playlist?list=PLRueFtKLr0QN7MmQ8pdpQerOe_s8vGJG4", "platform": "YouTube"}
        ]
    },
    "software engineer": {
        "courses": [
            {"name": "Software Engineering Master Track", "link": "https://www.coursera.org/professional-certificates/software-engineering-master-track", "platform": "Coursera"},
            {"name": "CS50's Introduction to Computer Science", "link": "https://www.edx.org/course/introduction-computer-science-harvardx-cs50x", "platform": "edX"},
            {"name": "Clean Code: Writing Code for Humans", "link": "https://www.udemy.com/course/writing-clean-code/", "platform": "Udemy"},
            {"name": "Full Stack Web Development", "link": "https://www.coursera.org/specializations/full-stack", "platform": "Coursera"}
        ],
        "videos": [
            {"name": "Software Engineering Playlist", "link": "https://www.youtube.com/playlist?list=PLWKjhJtqVnxf9N2EwU76v72Vw01d3h11J", "platform": "YouTube"},
            {"name": "Clean Code Explained", "link": "https://www.youtube.com/watch?v=ZI3q-_vjSZE", "platform": "YouTube"}
        ]
    },
    "data scientist": {
        "courses": [
            {"name": "Data Science Specialization", "link": "https://www.coursera.org/specializations/jhu-data-science", "platform": "Coursera"},
            {"name": "Python for Data Science", "link": "https://www.edx.org/professional-certificate/uc-berkeleyx-python-for-data-science", "platform": "edX"},
            {"name": "Advanced Data Analysis", "link": "https://www.udemy.com/course/advanced-data-analysis-python/", "platform": "Udemy"}
        ],
        "videos": [
            {"name": "Data Science with Python", "link": "https://www.youtube.com/watch?v=5qS-j0Yw49k", "platform": "YouTube"},
            {"name": "Statistics for Data Science", "link": "https://www.youtube.com/watch?v=xxpc-HPKN28", "platform": "YouTube"}
        ]
    },
     "AIML engineer": {
        "courses": [
            {"name": "Machine Learning Specialization", "link": "https://www.coursera.org/specializations/machine-learning", "platform": "Coursera"},
            {"name": "Deep Learning Specialization", "link": "https://www.coursera.org/specializations/deep-learning", "platform": "Coursera"},
            {"name": "AI for Everyone", "link": "https://www.coursera.org/learn/ai-for-everyone", "platform": "Coursera"}
        ],
        "videos": [
            {"name": "Machine Learning Fundamentals", "link": "https://www.youtube.com/watch?v=GIsg-Zp4mXU", "platform": "YouTube"},
            {"name": "Deep Learning Basics", "link": "https://www.youtube.com/watch?v=aircAruvnKk", "platform": "YouTube"}
        ]
    },
    "UI/UX designer": {
        "courses": [
            {"name": "Google UX Design Professional Certificate", "link": "https://www.coursera.org/professional-certificates/google-ux-design", "platform": "Coursera"},
            {"name": "UI/UX Design Specialization", "link": "https://www.coursera.org/specializations/ui-ux-design", "platform": "Coursera"},
            {"name": "Adobe XD Masterclass", "link": "https://www.udemy.com/course/adobe-xd-masterclass-ui-ux-design/", "platform": "Udemy"}
        ],
        "videos": [
            {"name": "UI/UX Design Fundamentals", "link": "https://www.youtube.com/watch?v=cT1-kK1_h58", "platform": "YouTube"},
            {"name": "Design Thinking Process", "link": "https://www.youtube.com/watch?v=_r0VX-aU_T8", "platform": "YouTube"}
        ]

    },
    # Non-Tech Domains
    "digital marketer": {
        "courses": [
            {"name": "Digital Marketing Specialization", "link": "https://www.coursera.org/specializations/digital-marketing", "platform": "Coursera"},
            {"name": "Google Digital Marketing Certification", "link": "https://learndigital.withgoogle.com/digitalgarage/course/digital-marketing", "platform": "Google"},
            {"name": "SEO for Beginners", "link": "https://www.udemy.com/course/seo-for-beginners/", "platform": "Udemy"}
        ],
        "videos": [
            {"name": "Introduction to Digital Marketing", "link": "https://www.youtube.com/watch?v=DvwS7cV9GmQ", "platform": "YouTube"},
            {"name": "Content Marketing Strategy", "link": "https://www.youtube.com/watch?v=hJhR8yLoDfE", "platform": "YouTube"}
        ]
    },
    "content writer": {
        "courses": [
            {"name": "Creative Writing Specialization", "link": "https://www.coursera.org/specializations/creative-writing", "platform": "Coursera"},
            {"name": "Writing for the Web", "link": "https://www.udemy.com/course/writing-for-the-web/", "platform": "Udemy"},
            {"name": "Copywriting for Beginners", "link": "https://www.udemy.com/course/copywriting-secrets/", "platform": "Udemy"}
        ],
        "videos": [
            {"name": "Content Writing Tips", "link": "https://www.youtube.com/watch?v=NYL2Vtmbh38", "platform": "YouTube"},
            {"name": "How to Write Blog Posts", "link": "https://www.youtube.com/watch?v=9H6GCe7hmmg", "platform": "YouTube"}
        ]
    },
    "graphic designer": {
        "courses": [
            {"name": "Graphic Design Specialization", "link": "https://www.coursera.org/specializations/graphic-design", "platform": "Coursera"},
            {"name": "Adobe Photoshop for Beginners", "link": "https://www.udemy.com/course/adobe-photoshop-course/", "platform": "Udemy"},
            {"name": "Canva Design Basics", "link": "https://www.udemy.com/course/canva-design-course/", "platform": "Udemy"}
        ],
        "videos": [
            {"name": "Graphic Design Basics", "link": "https://www.youtube.com/watch?v=V-NWV3B1gLE", "platform": "YouTube"},
            {"name": "Color Theory for Designers", "link": "https://www.youtube.com/watch?v=3Q0LnSOrAm4", "platform": "YouTube"}
        ]
    },
     "financial analyst": {
        "courses": [
            {"name": "Financial Analyst Certification Program", "link": "https://corporatefinanceinstitute.com/certifications/financial-analyst-training-program-fmva/", "platform": "CFI"},
            {"name": "Investment Management Specialization", "link": "https://www.coursera.org/specializations/investment-management", "platform": "Coursera"},
            {"name": "Excel for Finance", "link": "https://www.udemy.com/course/excel-for-finance-and-accounting/", "platform": "Udemy"}
        ],
        "videos": [
            {"name": "Introduction to Financial Analysis", "link": "https://www.youtube.com/watch?v=nb91cXJ9z0k", "platform": "YouTube"},
            {"name": "Understanding Financial Statements", "link": "https://www.youtube.com/watch?v=RhdxUs7XfP4", "platform": "YouTube"}
        ]
    },
    "cybersecurity specialist": {
        "courses": [
            {"name": "Cybersecurity Specialization", "link": "https://www.coursera.org/specializations/cyber-security", "platform": "Coursera"},
            {"name": "Certified Ethical Hacker (CEH)", "link": "https://www.eccouncil.org/trainings/certified-ethical-hacker/", "platform": "EC-Council"}
        ],
        "videos": [
            {"name": "Introduction to Cybersecurity", "link": "https://www.youtube.com/watch?v=fNzpcB7ODxQ", "platform": "YouTube"}
        ]
    },
    "cloud engineer": {
        "courses": [
            {"name": "Google Cloud Professional Cloud Architect", "link": "https://www.coursera.org/professional-certificates/google-cloud-architect", "platform": "Coursera"},
            {"name": "AWS Certified Solutions Architect", "link": "https://aws.amazon.com/certification/certified-solutions-architect-associate/", "platform": "AWS"}
        ],
        "videos": [
            {"name": "AWS Basics", "link": "https://www.youtube.com/watch?v=ulprqHHWlng", "platform": "YouTube"}
        ]
    },
    "blockchain developer": {
        "courses": [
            {"name": "Blockchain Specialization", "link": "https://www.coursera.org/specializations/blockchain", "platform": "Coursera"},
            {"name": "Ethereum and Solidity", "link": "https://www.udemy.com/course/ethereum-and-solidity-the-complete-developers-guide/", "platform": "Udemy"}
        ],
        "videos": [
            {"name": "Blockchain Basics", "link": "https://www.youtube.com/watch?v=SSo_EIwHSd4", "platform": "YouTube"}
        ]
    }
  }


job_openings = {
    "data analyst": [
        {"title": "Data Analyst", "link": "https://www.foundit.in/job/data-analyst-lynk-delhi-31205430?utm_campaign=google_jobs_apply", "company": "Energy Aspects"},
        {"title": "Senior Data Analyst", "link": "https://in.trabajo.org/job-3150-44553ee01fde049f939d8de7f0c0a902?utm_campaign=google_jobs_apply", "company": "Wipro"}
    ],
    "software engineer": [
        {"title": "Software Engineer", "link": "https://www.example.com/software-engineer1", "company": "Acme Corp"},
        {"title": "Senior Software Engineer", "link": "https://www.example.com/software-engineer2", "company": "Beta Co"}
    ],
    "data scientist": [
        {"title": "Data Scientist", "link": "https://www.example.com/data-scientist1", "company": "Gamma Inc"},
        {"title": "Senior Data Scientist", "link": "https://www.example.com/data-scientist2", "company": "Delta Co"}
    ],
   "digital marketer": [
        {"title": "Digital Marketing Specialist", "link": "https://www.example.com/digital-marketer1", "company": "MarketingPro"},
        {"title": "SEO Analyst", "link": "https://www.example.com/seo-analyst", "company": "SearchTech"}
    ],
    "content writer": [
        {"title": "Content Writer", "link": "https://www.example.com/content-writer", "company": "WriteWell"},
        {"title": "Copywriter", "link": "https://www.example.com/copywriter", "company": "AdText"}
    ],
    "graphic designer": [
        {"title": "Graphic Designer", "link": "https://www.example.com/graphic-designer", "company": "DesignHub"},
        {"title": "Creative Visualizer", "link": "https://www.example.com/creative-visualizer", "company": "VisualWorks"}
    ],
      "financial analyst": [
        {"title": "Financial Analyst", "link": "https://www.example.com/financial-analyst", "company": "FinCorp"},
        {"title": "Investment Analyst", "link": "https://www.example.com/investment-analyst", "company": "InvestWell"}
    ],
    "cybersecurity specialist": [
        {"title": "Cybersecurity Analyst", "link": "https://www.example.com/cybersecurity-analyst", "company": "SecureTech"},
        {"title": "Penetration Tester", "link": "https://www.example.com/penetration-tester", "company": "CyberProtect"}
    ],
    "cloud engineer": [
        {"title": "Cloud Architect", "link": "https://www.example.com/cloud-architect", "company": "Cloudify"},
        {"title": "DevOps Engineer", "link": "https://www.example.com/devops-engineer", "company": "BuildTech"}
    ],
     "blockchain developer": [
        {"title": "Blockchain Engineer", "link": "https://www.example.com/blockchain-engineer", "company": "LedgerTech"},
        {"title": "Smart Contract Developer", "link": "https://www.example.com/smart-contract-developer", "company": "CryptoHub"}
    ]
}

def translate_text(text, target_language):
     translator = Translator()
     try:
        translated = translator.translate(text, dest=target_language)
        return translated.text
     except Exception as e:
        print (f"Error during translation: {e}")
        return text #Return original text in case of failure

def generate_roadmap(courses, skill_gaps, career_ambitions, language):
    """Generates a study roadmap using T5 model."""
    input_text = f"Courses: {', '.join(courses)}. Skill Gaps: {skill_gaps}. Career Ambitions: {career_ambitions}. Generate a learning roadmap."
    input_text = translate_text(input_text, 'en')
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    output_ids = model.generate(input_ids, max_length=512, num_beams=5)  # Adjust parameters
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    generated_text= translate_text(generated_text, language)
    return generated_text


def recommend_resources(user_profile, language):
    recommendations = {"courses": [], "videos": [], "jobs":[]}

    combined_skills = user_profile["skills"] + user_profile["jobs"]

    if user_profile.get('current_role'):
       combined_skills.append(user_profile["current_role"])

    if user_profile.get('skill_gaps'):
       combined_skills.append(user_profile["skill_gaps"])

    if user_profile.get('career_ambitions'):
       combined_skills.append(user_profile["career_ambitions"])
    skill_set = set(combined_skills)

    for skill in skill_set:
        if skill in online_resources:
          recommendations["courses"].extend(online_resources[skill].get("courses",[]))
          recommendations["videos"].extend(online_resources[skill].get("videos",[]))

    for job_title in user_profile["jobs"]:
      if job_title in job_openings:
          recommendations["jobs"].extend(job_openings[job_title])
    if user_profile.get("career_ambitions") and  user_profile.get("career_ambitions") in job_openings:
         recommendations["jobs"].extend(job_openings[user_profile.get("career_ambitions")])

    unique_courses = { (course["name"],course['link']) :course for course in recommendations['courses'] }.values()
    unique_videos = { (video["name"],video['link']) : video for video in recommendations['videos'] }.values()
    unique_jobs = { (job["title"],job['link']) : job for job in recommendations['jobs'] }.values()


    if unique_courses:
      for course in unique_courses:
         course['name'] = translate_text(course['name'],language)
         course['platform'] = translate_text(course['platform'],language)

    if unique_videos:
      for video in unique_videos:
        video['name']= translate_text(video['name'],language)
        video['platform']= translate_text(video['platform'],language)
    if unique_jobs:
      for job in unique_jobs:
          job['title'] = translate_text(job['title'],language)
          job['company'] = translate_text(job['company'],language)
    return {"courses": list(unique_courses), "videos":list(unique_videos), "jobs": list(unique_jobs)}


@app.route('/')
def index():
     service = get_gmail_service()
     if service is None:
            return redirect(url_for('authenticate'))
     return render_template('index.html')


@app.route('/authenticate')
def authenticate():
    auth_url = session.get('auth_url', None)
    if not auth_url:
      return redirect(url_for('index'))
    return render_template('index1.html', auth_url=auth_url)

@app.route('/oauth2callback')
def oauth2callback():
    flow = InstalledAppFlow.from_client_secrets_file(
       'credentials.json', SCOPES)
    flow.fetch_token(code=request.args.get('code'))
    creds = flow.credentials
    # Save the credentials for future runs
    with open('token.pickle', 'wb') as token:
        pickle.dump(creds, token)
    return redirect(url_for('index'))

@app.route('/process', methods=['POST'])
def process():
    service = get_gmail_service()

    if service is None:
      return redirect(url_for('authenticate'))

    language = request.form['language']
    current_role = request.form['current_role'].lower()
    skill_gaps = request.form['skill_gaps'].lower()
    career_ambitions = request.form['career_ambitions'].lower()

    query = f"{current_role} OR {skill_gaps} OR {career_ambitions}"
    filtered_emails = fetch_filtered_emails(service, query=query)
    df = pd.DataFrame(filtered_emails)
    if df.empty:
        return render_template('results.html', error="No emails found matching your search terms!")
    df[['jobs','job_history','skills','training','feedback']] = df['content'].apply(lambda content: pd.Series(extract_insights(content)))
    user_profile = {
        "jobs": list(df['jobs'].explode().unique()),
        "job_history": list(df['job_history'].explode().unique()),
        "skills": list(df['skills'].explode().unique()),
        "training": list(df['training'].explode().unique()),
        "feedback": list(df['feedback'].explode().unique()),
        "current_role": current_role,
        "skill_gaps": skill_gaps,
        "career_ambitions":career_ambitions
    }
    recommendations= recommend_resources(user_profile,language)
    roadmap = generate_roadmap(user_profile['jobs'],user_profile['skill_gaps'],user_profile['career_ambitions'],language)
    return render_template('results.html', emails=df.to_dict(orient='records'), user_profile=user_profile, language=language,recommendations=recommendations, roadmap=roadmap)

if __name__ == '__main__':
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('stopwords')
    app.run(debug=True)