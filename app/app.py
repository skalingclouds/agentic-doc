import streamlit as st
import pandas as pd
from agentic_doc.parse import parse
from pydantic import create_model, BaseModel, Field
from typing import Dict, Any, List
from PyPDF2 import PdfReader, PdfWriter
from pdf2image import convert_from_bytes
from io import BytesIO
import os
from datetime import datetime

# App title
st.title("Agentic Doc Enterprise OCR Platform")

# Sidebar for API Key (assuming .env is loaded, but for demo)
st.sidebar.header("Configuration")
api_key = st.sidebar.text_input("Vision Agent API Key", type="password")
if api_key:
    os.environ["VISION_AGENT_API_KEY"] = api_key

# State management
if 'processed_docs' not in st.session_state:
    st.session_state.processed_docs = {'Approved': [], 'Rejected': [], 'Flagged': []}
if 'current_doc' not in st.session_state:
    st.session_state.current_doc = None
if 'schema_model' not in st.session_state:
    st.session_state.schema_model = None
if 'review_queue' not in st.session_state:
    st.session_state.review_queue = []

# Tab layout
tabs = st.tabs(["Upload & Process", "Schema Builder", "Review Queue", "Approved", "Rejected", "Flagged"])

with tabs[0]:
    st.header("Upload PDFs")
    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
    if st.button("Process PDFs") and uploaded_files:
        for file in uploaded_files:
            file_bytes = file.read()
            reader = PdfReader(BytesIO(file_bytes))
            num_pages = len(reader.pages)
            if num_pages > 1:
                for page_num in range(num_pages):
                    writer = PdfWriter()
                    writer.add_page(reader.pages[page_num])
                    page_bytes = BytesIO()
                    writer.write(page_bytes)
                    page_bytes.seek(0)
                    # Parse single page
                    results = parse(page_bytes.getvalue())
                    parsed = results[0] if results else None
                    extracted = None
                    if st.session_state.schema_model and parsed:
                        extract_results = parse(page_bytes.getvalue(), extraction_model=st.session_state.schema_model)
                        extracted = extract_results[0].extraction.dict() if extract_results else None
                    st.session_state.review_queue.append({
                        'name': f"{file.name}_page_{page_num+1}",
                        'parsed': parsed,
                        'file_bytes': page_bytes.getvalue(),
                        'extracted': extracted
                    })
                    st.success(f"Processed {file.name} page {page_num+1}")
            else:
                # Single page
                results = parse(file_bytes)
                parsed = results[0] if results else None
                extracted = None
                if st.session_state.schema_model:
                    extract_results = parse(file_bytes, extraction_model=st.session_state.schema_model)
                    extracted = extract_results[0].extraction.dict() if extract_results else None
                st.session_state.review_queue.append({
                    'name': file.name,
                    'parsed': parsed,
                    'file_bytes': file_bytes,
                    'extracted': extracted
                })
                st.success(f"Processed {file.name}")

with tabs[1]:
    st.header("AI Schema Builder")
    st.write("Define fields to extract (e.g., for sample.pdf: focus on handwritten values like names, dates, signatures. Leave blank fields as 'Blank'.")
    st.write("1. Set number of fields.")
    st.write("2. Enter name, type, description for each.")
    st.write("3. Click 'Build Schema' to create.")
    st.write("4. Click 'Apply Schema to Queue' to re-process pending documents.")
    num_fields = st.number_input("Number of fields", min_value=1, max_value=20, value=4)
    fields = {}
    for i in range(num_fields):
        col1, col2, col3 = st.columns(3)
        field_name = col1.text_input(f"Field Name {i+1}", value=f"field_{i+1}")
        field_type = col2.selectbox(f"Type {i+1}", ["str", "float", "int", "bool"], index=0)
        field_desc = col3.text_input(f"Description {i+1}")
        if field_name:
            fields[field_name] = (eval(field_type), Field(description=field_desc))
    
    if st.button("Build Schema"):
        st.session_state.schema_model = create_model('DynamicSchema', **fields)
        st.success("Schema built successfully!")
        st.write(st.session_state.schema_model)

    if st.button("Apply Schema to Queue") and st.session_state.schema_model:
        for item in st.session_state.review_queue:
            results = parse(item['file_bytes'], extraction_model=st.session_state.schema_model)
            item['extracted'] = results[0].extraction.dict() if results else None
        st.success("Schema applied to queue")

    st.subheader('Suggested Schema from sample.pdf')
    # Update suggested schema to match sample.pdf form fields, focusing on handwritten parts
    suggested_fields = {
        'plaintiff_name': (str, Field(description='Handwritten name of the plaintiff')),
        'court_county': (str, Field(description='Handwritten county of the court')),
        'case_number': (str, Field(description='Handwritten case number')),
        'plaintiff_signature': (bool, Field(description='Whether there is a handwritten signature (True/False)')),
        'signature_date': (str, Field(description='Handwritten date of signature')),
        # Add more fields based on sample.pdf, e.g.,
        'authorized_signature': (bool, Field(description='Whether authorized signature is present')),
        'printed_name': (str, Field(description='Handwritten printed name')),
        'address': (str, Field(description='Handwritten address')),
        'city_state_zip': (str, Field(description='Handwritten city, state, zip')),
        'telephone': (str, Field(description='Handwritten telephone number')),
        'email': (str, Field(description='Handwritten email address')),
    }
    if st.button('Use Suggested Schema'):
        st.session_state.schema_model = create_model('SampleSchema', **suggested_fields)
        st.success('Loaded suggested schema')

with tabs[2]:
    st.header("Review Queue")
    if st.session_state.review_queue:
        current = st.session_state.review_queue[0]
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("PDF Preview")
                images = convert_from_bytes(current['file_bytes'])
                for img in images:
                    st.image(img, width=400)
            
            with col2:
                st.subheader("Extracted Data Table")
                if current['extracted']:
                    data = current['extracted']
                    # Handle blanks and bools
                    table_data = {}
                    for key, value in data.items():
                        if value is None or value == '':
                            table_data[key] = 'Blank'
                        elif isinstance(value, bool):
                            table_data[key] = 'Yes' if value else 'No'
                        else:
                            table_data[key] = value
                    # Create Markdown table
                    md_table = '| Field | Value |\n|-------|-------|\n'
                    for k, v in table_data.items():
                        md_table += f'| {k.replace("_", " ").title()} | {v} |\n'
                    st.markdown(md_table)
                else:
                    st.info('No schema applied yet. Build and apply a schema to extract fields into table.')
                    st.write(current['parsed'].markdown)
        
        notes = st.text_area("Review Notes")
        action = st.selectbox("Action", ["Approve", "Reject", "Flag"])
        if st.button("Submit Review"):
            category = action + "ed"
            doc_data = {
                'name': current['name'],
                'extracted': current['extracted'] if current['extracted'] else current['parsed'].markdown,
                'notes': notes
            }
            st.session_state.processed_docs[category].append(doc_data)
            st.session_state.review_queue.pop(0)
            st.success(f"Document {category}")
    else:
        st.info("No documents in queue")

with tabs[3]:  # Approved
    st.header("Approved Documents")
    if st.session_state.processed_docs['Approved']:
        df = pd.DataFrame(st.session_state.processed_docs['Approved'])
        st.dataframe(df)
        for idx, doc in enumerate(st.session_state.processed_docs['Approved']):
            csv_data = pd.DataFrame([doc['extracted']]).to_csv(index=False).encode('utf-8')
            st.download_button(f'Download {doc["name"]} CSV', csv_data, f'{doc["name"]}.csv', 'text/csv', key=idx)
        if st.button("Clear Approved"):
            st.session_state.processed_docs['Approved'] = []
            st.success("Cleared")

# Similar for Rejected and Flagged tabs
with tabs[4]:
    st.header("Rejected Documents")
    if st.session_state.processed_docs['Rejected']:
        df = pd.DataFrame(st.session_state.processed_docs['Rejected'])
        st.dataframe(df)

with tabs[5]:
    st.header("Flagged Documents")
    if st.session_state.processed_docs['Flagged']:
        df = pd.DataFrame(st.session_state.processed_docs['Flagged'])
        st.dataframe(df) 