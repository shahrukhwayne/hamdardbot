"""
intents.py — Training data for Jamia Hamdard University Chatbot
Contains 25+ intents with patterns and responses.
"""

INTENTS = [
    {
        "tag": "greeting",
        "patterns": [
            "hello", "hi", "hey", "good morning", "good afternoon",
            "good evening", "howdy", "what's up", "greetings", "salaam"
        ],
        "responses": [
            "Assalamu Alaikum! Welcome to Jamia Hamdard University. How can I assist you today?",
            "Hello! I'm the Jamia Hamdard virtual assistant. What would you like to know?",
            "Hi there! Welcome to Jamia Hamdard. Feel free to ask me anything about the university."
        ],
        "dynamic": False
    },
    {
        "tag": "goodbye",
        "patterns": [
            "bye", "goodbye", "see you", "take care", "quit", "exit",
            "later", "farewell", "have a good day", "thanks bye"
        ],
        "responses": [
            "Thank you for visiting Jamia Hamdard! Have a great day.",
            "Goodbye! Feel free to return if you have more questions.",
            "Take care! Jamia Hamdard is always here for you."
        ],
        "dynamic": False
    },
    {
        "tag": "bot_identity",
        "patterns": [
            "who are you", "what are you", "are you a bot", "are you human",
            "what is your name", "introduce yourself", "tell me about yourself",
            "who made you", "are you ai", "what can you do"
        ],
        "responses": [
            "I am HamdardBot, an AI-powered virtual assistant for Jamia Hamdard University. I can help you with admissions, courses, fees, hostel, placements, and more!",
            "I'm an intelligent chatbot built specifically for Jamia Hamdard University. I combine machine learning with live web data to answer your queries accurately.",
            "I'm HamdardBot — your virtual guide to Jamia Hamdard University. Ask me anything about the university!"
        ],
        "dynamic": False
    },
    {
        "tag": "courses",
        "patterns": [
            "what courses are offered", "list of programs", "available courses",
            "what can i study", "undergraduate programs", "postgraduate programs",
            "btech courses", "mtech programs", "phd programs", "mba courses",
            "departments", "faculties", "which streams", "course list"
        ],
        "responses": [
            "Jamia Hamdard offers programs in Pharmacy, Medicine (MBBS), Nursing, Management, Computer Science, Engineering, Law, and Islamic Studies. Visit jamiahamdard.ac.in for the full list.",
            "The university has diverse faculties including Pharmacy, Medicine, Allied Health Sciences, Management & IT, Law, and more. Would you like details on any specific faculty?"
        ],
        "dynamic": False
    },
    {
        "tag": "admission",
        "patterns": [
            "how to apply", "admission process", "application form", "admission 2024",
            "admission 2025", "eligibility criteria", "when does admission start",
            "last date admission", "admission notice", "apply online", "admission open",
            "how to get admission", "entrance exam", "selection process"
        ],
        "responses": [
            "Fetching latest admission information from the official website...",
        ],
        "dynamic": True
    },
    {
        "tag": "fees",
        "patterns": [
            "fee structure", "how much fees", "course fees", "annual fees",
            "semester fees", "tuition fees", "hostel fees", "total fees",
            "payment", "fee details", "how much does it cost", "expenses"
        ],
        "responses": [
            "Fee structure varies by program. For example, B.Pharm is approx ₹80,000-₹1,20,000/year, MBBS is ₹7-9 Lakhs/year. Contact the admissions office at +91-11-26059688 for exact figures.",
            "Fees differ across programs. Please visit jamiahamdard.ac.in/fees or contact the Finance Office for the most current fee structure."
        ],
        "dynamic": False
    },
    {
        "tag": "hostel",
        "patterns": [
            "hostel", "accommodation", "where to stay", "hostel facilities",
            "hostel fees", "is hostel available", "boys hostel", "girls hostel",
            "hostel application", "on campus housing", "dormitory", "room"
        ],
        "responses": [
            "Jamia Hamdard provides separate hostel facilities for boys and girls on campus. Hostels are equipped with Wi-Fi, mess, sports facilities, and 24/7 security. Apply through the admissions office.",
            "The university has well-maintained hostels for both male and female students. Contact the Hostel Office at accommodation@jamiahamdard.ac.in for availability and fees."
        ],
        "dynamic": False
    },
    {
        "tag": "scholarships",
        "patterns": [
            "scholarship", "financial aid", "merit scholarship", "need based",
            "fee waiver", "bursary", "stipend", "free education", "discount in fees",
            "government scholarship", "minority scholarship"
        ],
        "responses": [
            "Jamia Hamdard offers merit-based and need-based scholarships. There are also special scholarships for minority students and toppers. Contact the Student Affairs office for eligibility details.",
            "Several scholarships are available: Hamdard Merit Scholarship, Minority Scholarships, and government schemes like NSP. Visit the Financial Aid office for more information."
        ],
        "dynamic": False
    },
    {
        "tag": "placements",
        "patterns": [
            "placement", "job", "campus recruitment", "companies visiting",
            "average salary", "highest package", "placement record",
            "career", "internship", "industry tie ups", "placement cell",
            "who recruits", "job offers"
        ],
        "responses": [
            "Jamia Hamdard has an active Training & Placement Cell. Top recruiters include Sun Pharma, Cipla, Ranbaxy, KPMG, Deloitte, and many healthcare firms. Average packages vary by department.",
            "The Placement Cell facilitates campus placements across pharma, healthcare, IT, and management sectors. Many students also pursue higher studies abroad. Contact placement@jamiahamdard.ac.in for details."
        ],
        "dynamic": False
    },
    {
        "tag": "rankings",
        "patterns": [
            "ranking", "rank", "nirf ranking", "qs ranking", "world ranking",
            "india ranking", "how good is jamia hamdard", "accreditation",
            "naac grade", "rating", "position", "best university"
        ],
        "responses": [
            "Jamia Hamdard is consistently ranked among India's top universities. It is NAAC A+ accredited and holds excellent NIRF rankings, especially in Pharmacy where it often ranks in the Top 5 nationally.",
            "The university is recognized for academic excellence with NAAC A+ grade. Jamia Hamdard's Pharmacy faculty is one of the best in Asia."
        ],
        "dynamic": False
    },
    {
        "tag": "contact",
        "patterns": [
            "contact", "phone number", "email", "address", "location",
            "how to reach", "office number", "helpline", "where is jamia hamdard",
            "website", "social media", "call", "reach out"
        ],
        "responses": [
            "Jamia Hamdard, Hamdard Nagar, New Delhi - 110062. Phone: +91-11-26059688. Email: info@jamiahamdard.ac.in. Website: www.jamiahamdard.ac.in",
            "You can reach Jamia Hamdard at: 📞 +91-11-26059688 | 📧 info@jamiahamdard.ac.in | 📍 Hamdard Nagar, New Delhi - 110062"
        ],
        "dynamic": False
    },
    {
        "tag": "news",
        "patterns": [
            "latest news", "recent news", "university news", "updates",
            "what's new", "announcements", "events", "notice board",
            "current news", "happenings", "university events"
        ],
        "responses": [
            "Fetching latest news from Jamia Hamdard...",
        ],
        "dynamic": True
    },
    {
        "tag": "notices",
        "patterns": [
            "notice", "circular", "official notice", "important notice",
            "exam notice", "result notice", "fee notice", "university circular",
            "latest circular", "academic notice"
        ],
        "responses": [
            "Fetching latest official notices from Jamia Hamdard...",
        ],
        "dynamic": True
    },
    {
        "tag": "library",
        "patterns": [
            "library", "books", "digital library", "e-library", "journals",
            "reading room", "library hours", "library facilities",
            "research papers", "thesis", "library access"
        ],
        "responses": [
            "Jamia Hamdard has a state-of-the-art central library with over 1 lakh books, journals, and digital resources. Library hours are 8 AM - 10 PM on weekdays.",
            "The university library provides access to e-journals, INFLIBNET, IEEE, and PubMed databases. Students get free access to thousands of research papers."
        ],
        "dynamic": False
    },
    {
        "tag": "sports",
        "patterns": [
            "sports", "games", "gym", "stadium", "cricket", "football",
            "basketball", "sports facilities", "sports complex", "athletics",
            "playground", "outdoor activities"
        ],
        "responses": [
            "Jamia Hamdard has excellent sports infrastructure including cricket grounds, football fields, basketball courts, a gymnasium, and a swimming pool.",
            "Students can participate in various sports activities. The university regularly organizes inter-departmental tournaments and participates in national-level events."
        ],
        "dynamic": False
    },
    {
        "tag": "faculty",
        "patterns": [
            "faculty", "professors", "teachers", "staff", "who teaches",
            "faculty list", "teacher details", "academic staff",
            "hod", "dean", "vice chancellor"
        ],
        "responses": [
            "Jamia Hamdard has distinguished faculty across all departments, many of whom are internationally acclaimed researchers and practitioners. Visit jamiahamdard.ac.in for the faculty directory.",
            "The university employs highly qualified professors, many with PhDs from top institutions. Each department has a dedicated Head of Department (HOD) and a Dean."
        ],
        "dynamic": False
    },
    {
        "tag": "research",
        "patterns": [
            "research", "phd", "research programs", "research facilities",
            "innovation", "lab", "laboratory", "projects", "funded research",
            "research center", "publications", "patents"
        ],
        "responses": [
            "Jamia Hamdard is a research-intensive university with multiple funded research projects, patents, and international collaborations. It has specialized research centres in Unani medicine, Pharmacy, and Biotech.",
            "The university actively encourages research with state-of-the-art labs and ICMR/DBT funded projects. PhD admissions are open in most departments."
        ],
        "dynamic": False
    },
    {
        "tag": "transport",
        "patterns": [
            "transport", "bus", "how to reach", "metro", "nearest metro",
            "cab", "distance from", "commute", "public transport",
            "nearest station", "bus route", "shuttle"
        ],
        "responses": [
            "Jamia Hamdard is located in Hamdard Nagar, New Delhi. Nearest metro: Tughlaqabad (Violet Line). DTC buses and autos are readily available.",
            "The university is well connected by metro and road. The nearest metro station is Tughlaqabad on the Violet Line. Campus buses are available for students from select locations."
        ],
        "dynamic": False
    },
    {
        "tag": "canteen",
        "patterns": [
            "canteen", "food", "cafeteria", "mess", "eat", "restaurant",
            "dining", "food options", "halal food", "meals", "tiffin"
        ],
        "responses": [
            "Jamia Hamdard has a central canteen and multiple food stalls on campus offering hygienic halal food at subsidized rates. The hostel mess provides three meals a day.",
            "Campus has a well-equipped canteen serving nutritious halal food. A variety of cuisines including North Indian, South Indian, and continental options are available."
        ],
        "dynamic": False
    },
    {
        "tag": "exams",
        "patterns": [
            "exam", "examination", "exam schedule", "exam date",
            "when are exams", "result", "grade", "marks",
            "admit card", "hall ticket", "back paper", "reappear"
        ],
        "responses": [
            "Exam schedules and results are published on the official portal at jamiahamdard.ac.in. Students must check the notice board and official website regularly for updates.",
            "Examinations are conducted at the end of each semester. Results are declared on the university portal within 4-6 weeks of the last exam."
        ],
        "dynamic": False
    },
    {
        "tag": "clubs",
        "patterns": [
            "clubs", "student clubs", "societies", "cultural activities",
            "extra curricular", "nss", "ncc", "student union", "fest",
            "cultural fest", "technical club", "student activities"
        ],
        "responses": [
            "Jamia Hamdard has active student clubs including a Cultural Society, Technical Club, NSS unit, Photography Club, Debate Club, and more. Annual fests are organized for students.",
            "Students can join various clubs and societies to develop their skills and personality. The annual cultural fest is a major highlight of the academic year."
        ],
        "dynamic": False
    },
    {
        "tag": "international",
        "patterns": [
            "foreign students", "international admission", "study in india",
            "visa", "international students", "mou", "foreign collaboration",
            "exchange program", "overseas"
        ],
        "responses": [
            "Jamia Hamdard welcomes international students and has MOUs with universities worldwide. International students can apply directly through the Foreign Students Cell. Contact: foreign@jamiahamdard.ac.in",
            "The university has collaborations with institutions in USA, UK, Malaysia, and several Middle Eastern countries. Exchange programs and joint research opportunities are available."
        ],
        "dynamic": False
    },
    {
        "tag": "pharmacy",
        "patterns": [
            "pharmacy", "b pharm", "m pharm", "pharm d", "pharmacology",
            "pharmacy courses", "drug", "medicine degree", "pharmacy admission"
        ],
        "responses": [
            "The Faculty of Pharmacy at Jamia Hamdard is one of the best in Asia and consistently ranks in the Top 5 in India. Programs offered: B.Pharm, M.Pharm (various specializations), and Pharm.D.",
            "Jamia Hamdard's Pharmacy faculty is internationally renowned. It offers B.Pharm, M.Pharm, Pharm.D., and PhD programs with world-class lab facilities."
        ],
        "dynamic": False
    },
    {
        "tag": "medicine",
        "patterns": [
            "mbbs", "medicine", "medical", "doctor", "md", "ms surgery",
            "clinical", "hamdard hospital", "hospital", "unani", "ayurveda",
            "nursing", "allied health"
        ],
        "responses": [
            "Jamia Hamdard offers MBBS through its affiliated hospital and has a strong Faculty of Medicine with MD, MS, and super-specialty programs. Unani medicine is also a flagship program.",
            "The university has a medical college and Majeedia Hospital for clinical training. MBBS, MD, MS, BDS, and Nursing programs are available."
        ],
        "dynamic": False
    },
    {
        "tag": "management",
        "patterns": [
            "mba", "bba", "management", "business", "finance",
            "marketing", "hr management", "mba admission", "executive mba",
            "management programs"
        ],
        "responses": [
            "The School of Management Studies at Jamia Hamdard offers MBA and BBA programs with specializations in Finance, Marketing, HR, and Healthcare Management.",
            "MBA programs at Jamia Hamdard are AICTE-approved with strong industry linkages. Admission is through CAT/MAT/CMAT scores."
        ],
        "dynamic": False
    },
    {
        "tag": "thanks",
        "patterns": [
            "thank you", "thanks", "thank you so much", "that was helpful",
            "great", "awesome", "perfect", "wonderful", "appreciate it"
        ],
        "responses": [
            "You're welcome! Jamia Hamdard is always here to help. Is there anything else you'd like to know?",
            "Happy to help! Don't hesitate to ask if you have more questions about Jamia Hamdard.",
            "Glad I could assist! Feel free to reach out anytime."
        ],
        "dynamic": False
    },
    {
        "tag": "confused",
        "patterns": [
            "i don't understand", "what do you mean", "can you explain",
            "confused", "unclear", "repeat that", "say again",
            "i didn't get that", "elaborate", "more details"
        ],
        "responses": [
            "I'm sorry if I wasn't clear. Could you rephrase your question? I'm here to help with anything related to Jamia Hamdard University.",
            "Let me try to clarify. Please share more details about what you'd like to know, and I'll do my best to assist!"
        ],
        "dynamic": False
    },
    {
    "tag": "campus_life",
    "patterns": [
        "campus life",
        "student life",
        "campus environment",
        "opportunities",
        "activities in university",
        "life at jamia hamdard",
        "what is campus like",
        "facilities and opportunities",
        "extra curricular activities"
    ],
    "responses": [
        "Jamia Hamdard offers a vibrant campus life with modern facilities, student clubs, cultural events, sports, and strong academic environment. Students benefit from research opportunities, internships, and industry exposure."
    ],
    "dynamic": False
}
]
