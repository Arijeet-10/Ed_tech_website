from flask import Flask, render_template, request, jsonify
import pickle
import re
import logging

app = Flask(__name__)

# Set up logging (optional but recommended)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Data embedded directly in the script
resources_db = {
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
            {"name": "Software Engineering Master Track", "link": "https://www.coursera.org/professional-certificates/devops-and-software-engineering", "platform": "Coursera"},
            {"name": "CS50's Introduction to Computer Science", "link": "https://www.edx.org/course/introduction-computer-science-harvardx-cs50x", "platform": "edX"},
            {"name": "Clean Code: Writing Code for Humans", "link": "https://www.udemy.com/course/writing-clean-code/", "platform": "Udemy"},
            {"name": "Full Stack Web Development", "link": "https://www.guvi.in/blog/best-full-stack-development-online-courses/", "platform": "Coursera"}
        ],
        "videos": [
            {"name": "Software Engineering Playlist", "link": "https://youtu.be/BDsCCtFl8WE", "platform": "YouTube"},
            {"name": "Clean Code Explained", "link": "https://youtu.be/7EmboKQH8lM", "platform": "YouTube"}
        ]
    },
    "data scientist": {
        "courses": [
            {"name": "Data Science Specialization", "link": "https://www.coursera.org/specializations/jhu-data-science", "platform": "Coursera"},
            {"name": "Python for Data Science", "link": "https://cdss.berkeley.edu/dsus/academics/class-enrollment-info", "platform": "edX"},
            {"name": "Advanced Data Analysis", "link": "https://www.reddit.com/r/dataanalysis/comments/12lyue5/review_of_google_advanced_data_analysis/", "platform": "Udemy"}
        ],
        "videos": [
            {"name": "Data Science with Python", "link": "https://youtu.be/mkv5mxYu0Wk?si=OlXRgf-yO8mQwKOm", "platform": "YouTube"},
            {"name": "Statistics for Data Science", "link": "https://www.youtube.com/watch?v=xxpc-HPKN28", "platform": "YouTube"}
        ]
    },
     "aiml engineer": {
        "courses": [
            {"name": "Machine Learning Specialization", "link": "https://www.coursera.org/specializations/machine-learning", "platform": "Coursera"},
            {"name": "Deep Learning Specialization", "link": "https://www.coursera.org/specializations/deep-learning", "platform": "Coursera"},
            {"name": "AI for Everyone", "link": "https://www.coursera.org/learn/ai-for-everyone", "platform": "Coursera"}
        ],
        "videos": [
            {"name": "Machine Learning Fundamentals", "link": "https://www.uopeople.edu/blog/machine-learning-courses/", "platform": "Uopeople"},
            {"name": "Deep Learning Basics", "link": "https://www.youtube.com/watch?v=aircAruvnKk", "platform": "YouTube"}
        ]
    },
    "ui/ux designer": {
        "courses": [
            {"name": "Google UX Design Professional Certificate", "link": "https://www.coursera.org/professional-certificates/google-ux-design", "platform": "Coursera"},
            {"name": "UI/UX Design Specialization", "link": "https://www.coursera.org/specializations/ui-ux-design", "platform": "Coursera"},
            {"name": "Adobe XD Masterclass", "link": "https://www.simplilearn.com/free-adobe-xd-course-skillup", "platform": "Simplilearn"}
        ],
        "videos": [
            {"name": "UI/UX Design Fundamentals", "link": "https://designlab.com/blog/best-ux-design-course-online", "platform": "DesignLab"},
            {"name": "Design Thinking Process", "link": "https://www.youtube.com/watch?v=_r0VX-aU_T8", "platform": "YouTube"}
        ]

    },
    # Non-Tech Domains
    "digital marketer": {
        "courses": [
            {"name": "Digital Marketing Specialization", "link": "https://www.coursera.org/specializations/digital-marketing", "platform": "Coursera"},
            {"name": "Google Digital Marketing Certification", "link": "https://learndigital.withgoogle.com/digitalgarage/course/digital-marketing", "platform": "Google"},
            {"name": "SEO for Beginners", "link": "https://www.coursera.org/learn/seo-fundamentals", "platform": "Coursesera"}
        ],
        "videos": [
            {"name": "Introduction to Digital Marketing", "link": "https://www.youtube.com/watch?v=DvwS7cV9GmQ", "platform": "YouTube"},
            {"name": "Content Marketing Strategy", "link": "https://academy.hubspot.com/courses/content-marketing", "platform": "HubSpot Academy"}
        ]
    },
    "content writer": {
        "courses": [
            {"name": "Creative Writing Specialization", "link": "https://www.coursera.org/specializations/creative-writing", "platform": "Coursera"},
            {"name": "Writing for the Web", "link": "https://www.linkedin.com/learning/writing-for-the-web/welcome?u=229219690", "platform": "LinkedIN Learning"},
            {"name": "Copywriting for Beginners", "link": "https://www.udemy.com/course/copywriting-secrets/", "platform": "Udemy"}
        ],
        "videos": [
            {"name": "Content Writing Tips", "link": "https://youtu.be/8BdZ0dUu7VQ?si=iEpjVm92wnLDRnRT", "platform": "YouTube"},
            {"name": "How to Write Blog Posts", "link": "https://youtu.be/Q8rN3JKqUc8?si=xUJTLqwgB7gyb6LV", "platform": "YouTube"}
        ]
    },
    "graphic designer": {
        "courses": [
            {"name": "Graphic Design Specialization", "link": "https://www.coursera.org/specializations/graphic-design", "platform": "Coursera"},
            {"name": "Adobe Photoshop for Beginners", "link": "https://www.udemy.com/course/adobe-photoshop-course/", "platform": "Udemy"},
            {"name": "Canva Design Basics", "link": "https://youtu.be/J0jE0OsF1zo?si=oWns6lUKPse1bS3-", "platform": "YouTube"}
        ],
        "videos": [
            {"name": "Graphic Design Basics", "link": "https://youtu.be/GQS7wPujL2k?si=sC9Ccf3TFpPDN7nD", "platform": "YouTube"},
            {"name": "Color Theory for Designers", "link": "https://youtu.be/7iY4QFqTlpE?si=upcYDDRGXgz4zyVb", "platform": "YouTube"}
        ]
    },
     "financial analyst": {
        "courses": [
            {"name": "Financial Analyst Certification Program", "link": "https://www.khanacademy.org/economics-finance-domain/core-finance", "platform": "Khan Academy"},
            {"name": "Investment Management Specialization", "link": "https://www.coursera.org/specializations/investment-management", "platform": "Coursera"},
            {"name": "Excel for Finance", "link": "https://www.udemy.com/course/excel-for-finance-and-accounting/", "platform": "Udemy"}
        ],
        "videos": [
            {"name": "Introduction to Financial Analysis", "link": "https://youtu.be/Fi1wkUczuyk?si=d2Vt0N0dqxPG3W6z", "platform": "YouTube"},
            {"name": "Understanding Financial Statements", "link": "https://youtu.be/mnJDA3YXL9g?si=q0uuPW7fyyqEvmwM", "platform": "YouTube"}
        ]
    },
    "cybersecurity specialist": {
        "courses": [
            {"name": "Cybersecurity Specialization", "link": "https://www.coursera.org/professional-certificates/ibm-cybersecurity-analyst", "platform": "Coursera"},
            {"name": "Certified Ethical Hacker (CEH)", "link": "https://www.eccouncil.org/trainings/certified-ethical-hacker/", "platform": "EC-Council"}
        ],
        "videos": [
            {"name": "Introduction to Cybersecurity", "link": "https://youtu.be/z5nc9MDbvkw?si=H0XgtqET9NC-lgc9", "platform": "YouTube"}
        ]
    },
    "cybersecurity": {
        "courses": [
            {"name": "Cybersecurity Specialization", "link": "https://www.coursera.org/professional-certificates/ibm-cybersecurity-analyst", "platform": "Coursera"},
            {"name": "Certified Ethical Hacker (CEH)", "link": "https://www.eccouncil.org/trainings/certified-ethical-hacker/", "platform": "EC-Council"}
        ],
        "videos": [
            {"name": "Introduction to Cybersecurity", "link": "https://youtu.be/z5nc9MDbvkw?si=H0XgtqET9NC-lgc9", "platform": "YouTube"}
        ]
    },
    "cloud engineer": {
        "courses": [
            {"name": "Google Cloud Professional Cloud Architect", "link": "https://www.pluralsight.com/cloud-guru", "platform": "PLURALSIGHT"},
            {"name": "AWS Certified Solutions Architect", "link": "https://aws.amazon.com/certification/certified-solutions-architect-associate/", "platform": "AWS"}
        ],
        "videos": [
            {"name": "AWS Basics", "link": "https://youtu.be/k1RI5locZE4?si=HcRDvpe7rfDWwLMZ", "platform": "YouTube"}
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
        {"title": "Software Engineer", "link": "https://www.coursera.org/articles/software-engineer?form=MG0AV3", "company": "Acme Corp"},
        {"title": "Senior Software Engineer", "link": "https://www.theforage.com/blog/careers/software-engineering-resources?form=MG0AV3", "company": "Beta Co"}
    ],
    "data scientist": [
        {"title": "Data Scientist", "link": "https://www.coursera.org/specializations/jhu-data-science?form=MG0AV3", "company": "Gamma Inc"},
        {"title": "Senior Data Scientist", "link": "https://www.edx.org/certificates/professional-certificate/harvardx-data-science?form=MG0AV3", "company": "Delta Co"}
    ],
   "digital marketer": [
        {"title": "Digital Marketing Specialist", "link": "https://academy.hubspot.com/courses/digital-marketing?form=MG0AV3", "company": "MarketingPro"},
        {"title": "SEO Analyst", "link": "https://moz.com/beginners-guide-to-seo?form=MG0AV3", "company": "SearchTech"}
    ],
    "content writer": [
        {"title": "Content Writer", "link": "https://contentmarketinginstitute.com/articles/writing-examples-tools-tips/?form=MG0AV3", "company": "WriteWell"},
        {"title": "Copywriter", "link": "https://www.henryharvin.com/blog/content-writing-platforms/", "company": "AdText"}
    ],
    "graphic designer": [
        {"title": "Graphic Designer", "link": "https://www.timechamp.io/blogs/explore-the-top-10-freelance-graphic-design-websites/", "company": "DesignHub"},
        {"title": "Creative Visualizer", "link": "https://365datascience.com/trending/data-visualization-project-ideas/", "company": "VisualWorks"}
    ],
      "financial analyst": [
        {"title": "Financial Analyst", "link": "https://romeromentoring.com/financial-analyst-skills/", "company": "FinCorp"},
        {"title": "Investment Analyst", "link": "https://www.ollusa.edu/blog/financial-analyst-skills.html", "company": "InvestWell"}
    ],
    "cybersecurity specialist": [
        {"title": "Cybersecurity Analyst", "link": "https://www.infosectrain.com/courses/cybersecurity-analyst-training/", "company": "SecureTech"},
        {"title": "Penetration Tester", "link": "https://www.coursera.org/in/articles/how-to-become-a-penetration-tester", "company": "CyberProtect"}
    ],
    "cybersecurity": [
        {"title": "Cybersecurity Analyst", "link": "https://www.infosectrain.com/courses/cybersecurity-analyst-training/", "company": "SecureTech"},
        {"title": "Penetration Tester", "link": "https://www.coursera.org/in/articles/how-to-become-a-penetration-tester", "company": "CyberProtect"}
    ],
    "cloud engineer": [
        {"title": "Cloud Architect", "link": "https://www.coursera.org/articles/how-to-become-a-cloud-architect", "company": "Cloudify"},
        {"title": "DevOps Engineer", "link": "https://www.geeksforgeeks.org/devops-engineer-skills/", "company": "BuildTech"}
    ],
     "blockchain developer": [
        {"title": "Blockchain Engineer", "link": "https://www.gsdcouncil.org/blogs/blockchain-developer-skills", "company": "LedgerTech"},
        {"title": "Smart Contract Developer", "link": "https://www.techtarget.com/searchcio/tip/How-to-become-a-smart-contract-developer", "company": "CryptoHub"}
    ]
}

job_titles = set(job_openings.keys())

# Load the trained ML model if available (optional)
def load_model():
    try:
        with open('emails.pkl', 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        logging.warning("No ML model found. Using basic logic.")
        return None

model = load_model()


@app.route('/')
def index():
    """Render the main input page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Predict resources based on user input."""
    try:
        data = request.get_json()  # Use get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        role = data.get('role', '').strip().lower()
        skills = [skill.strip().lower() for skill in data.get('skills', '').split(',') if skill.strip()] if data.get('skills') else []
        skill_gaps = data.get('skill_gaps', '').strip().lower()
        career_ambitions = data.get('career_ambitions', '').strip().lower()
        language = data.get('language', '').strip()


        # Combine inputs to determine resources
        recommendations = {"courses": [], "videos": [], "jobs": []}
        
        # Combine inputs into one single string to search within resources_db
        combined_inputs_string = ' '.join(skills + [role, skill_gaps, career_ambitions]).strip()

        # Create regex patterns for all resource keys
        resource_patterns = {key: re.compile(r'\b' + re.escape(key) + r'\b') for key in resources_db}

        # Search resources based on input words/phrases
        for resource_key, pattern in resource_patterns.items():
            if pattern.search(combined_inputs_string):
                recommendations["courses"].extend(resources_db[resource_key]["courses"])
                recommendations["videos"].extend(resources_db[resource_key]["videos"])

        # Search job openings based on exact matches of job titles
        job_title_patterns = {key: re.compile(r'\b' + re.escape(key) + r'\b') for key in job_titles}
        for job_title, pattern in job_title_patterns.items():
             if pattern.search(combined_inputs_string):
                 recommendations["jobs"].extend(job_openings[job_title])

        # Remove duplicates
        recommendations["courses"] = list({v["name"]: v for v in recommendations["courses"]}.values())
        recommendations["videos"] = list({v["name"]: v for v in recommendations["videos"]}.values())
        recommendations["jobs"] = list({v["title"]: v for v in recommendations["jobs"]}.values())

        # Example of how to use the language input (e.g., display message)
        if language:
            recommendations['message'] = f"Resources tailored for you in {language}!"
        if model is None:
             recommendations['message'] = f"ML model unavailable, using basic logic!"


        return jsonify(recommendations)

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)