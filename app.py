import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr

# -----------------------------------------------
# ✅ Example Data
# -----------------------------------------------

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

# Build profile text
def build_profile_text(row):
    return f"{row['name']}, skills: {', '.join(row['skills'])}, " \
           f"experience: {row['experience_years']} years, " \
           f"projects: {', '.join(row['projects'])}, availability: {row['availability']}"

df["profile_text"] = df.apply(build_profile_text, axis=1)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_text(text):
    return model.encode(text)

df["embedding"] = df["profile_text"].apply(embed_text)

# Search logic
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

def chatbot_interface(user_query, skill, exp, availability):
    candidates = retrieve_candidates(user_query, skill, exp, availability)
    response = generate_template_response(user_query, candidates)
    return response

# Define Gradio UI
skills_list = sorted(list(set(skill for sublist in df.skills for skill in sublist)))
skills_list.insert(0, "Any")

iface = gr.Interface(
    fn=chatbot_interface,
    inputs=[
        gr.Textbox(label="Query", placeholder="e.g. healthcare projects"),
        gr.Dropdown(choices=skills_list, label="Skill", value="Any"),
        gr.Slider(minimum=0, maximum=10, step=1, label="Minimum Experience (Years)", value=0),
        gr.Radio(choices=["Any", "available", "busy"], label="Availability", value="Any"),
    ],
    outputs=gr.Markdown(label="Chatbot Response"),
    title="💼 HR Assistant Chatbot",
    description="Search and discover employees matching your requirements.",
)

if __name__ == "__main__":
    iface.launch()
