import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import ipywidgets as widgets
from IPython.display import display, clear_output, Markdown

# ------------------------------------------------------------
# ✅ Example Data
# ------------------------------------------------------------

employees = [
    {
        "id": 1,
        "name": "Alice Johnson",
        "skills": ["Python", "React", "AWS"],
        "experience_years": 5,
        "projects": ["E-commerce Platform", "Healthcare Dashboard"],
        "availability": "available"
    },
    {
        "id": 2,
        "name": "Bob Smith",
        "skills": ["Java", "Spring", "Docker"],
        "experience_years": 4,
        "projects": ["Banking System", "Inventory Management"],
        "availability": "busy"
    },
    {
        "id": 3,
        "name": "Dr. Sarah Chen",
        "skills": ["Machine Learning", "TensorFlow", "PyTorch"],
        "experience_years": 6,
        "projects": ["Medical Diagnosis Platform", "Healthcare Dashboard"],
        "availability": "available"
    },
    {
        "id": 4,
        "name": "Michael Rodriguez",
        "skills": ["Machine Learning", "scikit-learn", "pandas"],
        "experience_years": 4,
        "projects": ["Patient Risk Prediction System", "EHR Analysis"],
        "availability": "available"
    },
    {
        "id": 5,
        "name": "Emma Brown",
        "skills": ["Python", "AWS", "Docker"],
        "experience_years": 7,
        "projects": ["Cloud Migration", "Healthcare Dashboard"],
        "availability": "available"
    },
    {
        "id": 6,
        "name": "Faisal Ahmed",
        "skills": ["C#", ".NET", "SQL Server"],
        "experience_years": 8,
        "projects": ["Hospital Management System", "ERP Solutions"],
        "availability": "available"
    },
    {
        "id": 7,
        "name": "Grace Thompson",
        "skills": ["JavaScript", "Angular", "Firebase"],
        "experience_years": 2,
        "projects": ["E-commerce Site", "Chat Application"],
        "availability": "available"
    },
    {
        "id": 8,
        "name": "Harish Patel",
        "skills": ["Python", "Machine Learning", "Pandas"],
        "experience_years": 4,
        "projects": ["Sales Prediction", "Healthcare Analytics"],
        "availability": "not available"
    },
    {
        "id": 9,
        "name": "Isabella Scott",
        "skills": ["React", "Node.js", "GraphQL"],
        "experience_years": 3,
        "projects": ["Social Media Platform", "Food Delivery App"],
        "availability": "available"
    },
    {
        "id": 10,
        "name": "John Carter",
        "skills": ["AWS", "Terraform", "Docker"],
        "experience_years": 5,
        "projects": ["Cloud Infrastructure Setup", "Security Auditing"],
        "availability": "available"
    },
    {
        "id": 11,
        "name": "Kavita Joshi",
        "skills": ["Python", "Data Analysis", "NLP"],
        "experience_years": 6,
        "projects": ["Chatbot Development", "Document Classification"],
        "availability": "available"
    },
    {
        "id": 12,
        "name": "Liam Murphy",
        "skills": ["Java", "Spring Boot", "Kubernetes"],
        "experience_years": 4,
        "projects": ["Microservices Architecture", "Banking Systems"],
        "availability": "not available"
    },
    {
        "id": 13,
        "name": "Maya Kapoor",
        "skills": ["React Native", "Redux", "TypeScript"],
        "experience_years": 3,
        "projects": ["Healthcare Mobile App", "E-learning Platform"],
        "availability": "available"
    },
    {
        "id": 14,
        "name": "Nathan Zhang",
        "skills": ["Python", "Computer Vision", "OpenCV"],
        "experience_years": 5,
        "projects": ["Face Recognition", "Video Analytics"],
        "availability": "available"
    },
    {
        "id": 15,
        "name": "Olivia Fernandez",
        "skills": ["SQL", "Data Warehousing", "ETL"],
        "experience_years": 7,
        "projects": ["Financial Reporting", "Supply Chain Analytics"],
        "availability": "available"
    },
]

df = pd.DataFrame(employees)

# Add profile text for embedding
def build_profile_text(row):
    return f"{row['name']}, skills: {', '.join(row['skills'])}, " \
           f"experience: {row['experience_years']} years, " \
           f"projects: {', '.join(row['projects'])}, availability: {row['availability']}"

df["profile_text"] = df.apply(build_profile_text, axis=1)

# ------------------------------------------------------------
# ✅ Embeddings
# ------------------------------------------------------------

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_text(text):
    return model.encode(text)

# Compute embeddings for each profile
print("Generating embeddings...")
df["embedding"] = df["profile_text"].apply(embed_text)

# ------------------------------------------------------------
# ✅ Retrieval Logic
# ------------------------------------------------------------

def retrieve_candidates(user_query, skill_filter, exp_filter, availability_filter, top_k=3):
    query_embedding = embed_text(user_query)
    employee_embeddings = np.vstack(df["embedding"].values)
    similarities = cosine_similarity([query_embedding], employee_embeddings)[0]
    df["similarity"] = similarities

    filtered_df = df.copy()

    if skill_filter != "Any":
        filtered_df = filtered_df[filtered_df.skills.apply(lambda skills: skill_filter in skills)]
    
    if availability_filter != "Any":
        filtered_df = filtered_df[filtered_df.availability == availability_filter.lower()]
    
    filtered_df = filtered_df[filtered_df.experience_years >= exp_filter]

    top_matches = filtered_df.sort_values("similarity", ascending=False).head(top_k)

    return top_matches

def generate_template_response(user_query, candidates_df):
    if candidates_df.empty:
        return f"⚠️ No employees found matching your query: {user_query}"

    response = f"📋 Based on your requirements — *{user_query}*, here are {len(candidates_df)} strong candidate(s):\n\n"

    for _, row in candidates_df.iterrows():
        name = f"**{row['name']}**"
        exp = row['experience_years']
        skills = ", ".join(row['skills'])
        projects = ", ".join([f"'{p}'" for p in row['projects']])
        availability = row['availability']
        
        paragraph = (
            f"{name} has {exp} years of relevant experience and has contributed to projects like {projects}. "
            f"Key skills include {skills}. "
            f"This candidate is currently **{availability}**.\n\n"
        )
        response += paragraph

    response += "Would you like me to check their availability for meetings or provide more details on their project contributions?"
    
    return response

# ------------------------------------------------------------
# ✅ ipywidgets UI
# ------------------------------------------------------------

def run_ui():
    all_skills = sorted(list(set(skill for sublist in df.skills for skill in sublist)))
    all_skills.insert(0, "Any")

    skill_dropdown = widgets.Dropdown(
        options=all_skills,
        description='Skill:',
        layout=widgets.Layout(width='50%')
    )

    exp_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=10,
        step=1,
        description='Min Exp (yrs):',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='50%')
    )

    availability_toggle = widgets.ToggleButtons(
        options=['Any', 'Available', 'Busy'],
        description='Availability:',
        style={'button_width': '80px'},
    )

    text_query = widgets.Text(
        placeholder='Optional keywords, e.g. "healthcare projects"',
        description='Query:',
        layout=widgets.Layout(width='80%')
    )

    ask_button = widgets.Button(
        description="Search",
        button_style='success',
        icon='search'
    )

    output = widgets.Output()

    def on_button_click(b):
        user_query = text_query.value or " "
        skill = skill_dropdown.value
        min_exp = exp_slider.value
        availability = availability_toggle.value

        candidates = retrieve_candidates(user_query, skill, min_exp, availability)
        response = generate_template_response(user_query, candidates)

        with output:
            clear_output()
            display(Markdown(f"----- **HR Chatbot Response** -----\n\n{response}"))
            
            if not candidates.empty:
                display(candidates[["name", "skills", "experience_years", "projects", "availability"]])

    ask_button.on_click(on_button_click)

    ui = widgets.VBox([
        widgets.HTML("<h3>💼 HR Assistant Chatbot</h3>"),
        text_query,
        skill_dropdown,
        exp_slider,
        availability_toggle,
        ask_button,
        output
    ])

    display(ui)

# ------------------------------------------------------------
# ✅ Main Entry Point
# ------------------------------------------------------------

if __name__ == "__main__":
    print("✅ HR Assistant Chatbot app is ready!")
    print("To run it, execute this script in a Jupyter Notebook cell:")
    print("from app import run_ui")
    print("run_ui()")
