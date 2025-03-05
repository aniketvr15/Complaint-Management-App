import streamlit as st
import sqlite3
from transformers import pipeline

# Initialize the summarization and classification pipelines
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
classifier = pipeline("zero-shot-classification")

def init_db():
    conn = sqlite3.connect("complaints.db")
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS complaints (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        complaint TEXT,
                        summary TEXT,
                        category TEXT)''')
    conn.commit()
    conn.close()

def save_complaint(complaint, summary, category):
    conn = sqlite3.connect("complaints.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO complaints (complaint, summary, category) VALUES (?, ?, ?)", (complaint, summary, category))
    conn.commit()
    conn.close()

def get_past_complaints():
    conn = sqlite3.connect("complaints.db")
    cursor = conn.cursor()
    cursor.execute("SELECT complaint, summary, category FROM complaints ORDER BY id DESC")
    data = cursor.fetchall()
    conn.close()
    return data

def summarize_complaint(complaint_text):
    summary = summarizer(complaint_text, max_length=50, min_length=10, do_sample=False)
    return summary[0]['summary_text']

def categorize_complaint(complaint_text):
    labels = ["Billing Issue", "Delivery Delay", "Product Quality", "Customer Support", "Technical Issue"]
    result = classifier(complaint_text, candidate_labels=labels)
    return result["labels"][0]

def main():
    st.set_page_config(page_title="Complaint Management System", layout="wide")
    
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Submit Complaint", "Past Complaints"])
    
    if page == "Submit Complaint":
        st.title("Submit a Complaint")
        complaint_text = st.text_area("Enter your complaint:")
        
        if st.button("Submit"):
            if complaint_text.strip():
                summary = summarize_complaint(complaint_text)
                category = categorize_complaint(complaint_text)
                save_complaint(complaint_text, summary, category)
                
                st.success("Complaint submitted successfully!")
                st.write("### Summary:", summary)
                st.write("### Suggested Category:", category)
            else:
                st.error("Complaint text cannot be empty!")
    
    elif page == "Past Complaints":
        st.title("Past Complaints")
        complaints = get_past_complaints()
        
        if complaints:
            for complaint, summary, category in complaints:
                with st.expander(complaint[:100] + "..."):
                    st.write("**Summary:**", summary)
                    st.write("**Category:**", category)
        else:
            st.info("No past complaints found.")

if __name__ == "__main__":
    init_db()
    main()
