import streamlit as st
import os
import glob
from main import app  

st.set_page_config(page_title="DeepDive-Architect-AI", layout="wide")

with st.sidebar:
    st.header(" Past Research Files")
    

    md_files = [f for f in glob.glob("*.md") if f.lower() != "readme.md"]
    
    if not md_files:
        st.info("No research files generated yet.")
    else:
        for file in md_files:
            col1, col2 = st.columns([3, 1])
            with col1:
                
                st.write(f" {file}")
            with col2:
                
                if st.button("Del", key=f"del_{file}", help=f"Delete {file}"):
                    os.remove(file)
                    st.success("Deleted!")
                    st.rerun() 


st.title("DeepDive-Architect-AI")
st.markdown("Enter a topic below, and the agent will research and generate a comprehensive study guide.")

user_topic = st.text_input("What do you want to learn about?", placeholder="e.g., NLP in Deep Learning")

if st.button("Generate Study Plan", type="primary"):
    if user_topic:
        with st.spinner(f"Agent is researching '{user_topic}'... This might take a minute!"):
            
            
            result = app.invoke({"topic": user_topic, "sections": []})
            
            
            final_markdown_text = result.get("final", "Error: Could not generate content.")
            
        
            safe_filename = user_topic.strip().lower().replace(" ", "_") + ".md"

            st.success("Research Complete!")
      
            st.download_button(
                label=" Download Markdown File",
                data=final_markdown_text,
                file_name=safe_filename,
                mime="text/markdown"
            )
            
            st.markdown("---")
            
           
            st.markdown(final_markdown_text)
            
    else:
        st.warning("Please enter a topic first!")