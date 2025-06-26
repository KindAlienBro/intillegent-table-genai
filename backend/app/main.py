# backend/app/main.py

from fastapi import FastAPI, HTTPException, Form, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import fitz  # PyMuPDF
import json
import os
import sys
import google.generativeai as genai # Ensure this is used if GOOGLE_API_KEY is configured

# This allows importing from your existing llm_service.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# We will use the functions you already have
from llm_service import generate_json_response, clean_gemini_response

# -- Initialize the FastAPI App --
app = FastAPI(
    title="Intelligent Universal Prompt Table Generator API",
    version="1.5" # Version bump for robust AI Chaining!
)

# -- Configure CORS --
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to your frontend's origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================================
#        HELPER FUNCTIONS FOR THE NEW "AI CHAIN"
# ==========================================================

def get_pdf_columns_with_llm(raw_text: str) -> list[str]:
    """
    AI CHAIN STEP 1: Makes a small, fast AI call to identify column headers from the PDF.
    """
    if not raw_text.strip(): return []
    try:
        # A very focused prompt to extract only the column headers
        parsing_prompt = f"""Analyze the start of this text from a PDF and identify the column headers. Return a single, flat JSON array of strings with the header names. Example: ["SI. No.", "USN", "Name"]. Ignore document titles or any text that is clearly not a column header. Focus on typical tabular headers. Text: --- {raw_text[:1000]} ---""" # Increased context slightly
        model = genai.GenerativeModel("gemini-1.5-flash-latest") # Use a fast model for this simple task
        response = model.generate_content(
            parsing_prompt,
            generation_config=genai.types.GenerationConfig(response_mime_type="application/json"),
            request_options={"timeout": 30} # Timeout for this simpler call
        )
        cleaned_text = clean_gemini_response(response.text) # Clean before parsing
        headers = json.loads(cleaned_text)
        return headers if isinstance(headers, list) and all(isinstance(h, str) for h in headers) else []
    except Exception as e:
        print(f"Error getting columns from PDF: {e}")
        return []

def populate_data_with_llm(raw_text: str, schema: dict) -> list:
    """
    AI CHAIN STEP 3: Uses a focused AI call to parse the full PDF text against a final, correct schema.
    This function leverages the main generate_json_response for robustness.
    """
    if not raw_text.strip() or not schema or not schema.get("columns"): return [{}]
    try:
        # A focused prompt to parse data against a pre-defined schema
        # The system instructions for generate_json_response already handle the JSON output format.
        # We just need to give it the task description and data.
        user_task_prompt = f"""
        Parse the following 'Raw Text' and structure it into a JSON array of objects that fits the provided 'JSON Schema'.
        Map the data for each row to the correct column `id` from the schema.
        Ignore any header rows present in the 'Raw Text' itself, as the schema defines the headers.
        Ensure each object in the output array corresponds to one row of data.
        If a column's data is missing for a row in the text, use null or an empty string for that field.

        **JSON Schema to follow (use the 'id' fields for mapping):**
        ```json
        {json.dumps(schema, indent=2)}
        ```
        """
        # The generate_json_response expects the raw text to be part of the full_prompt
        # in a specific way if it's meant to be processed directly.
        # We construct the full prompt including the PDF text marker as expected by SYSTEM_INSTRUCTIONS
        full_parsing_prompt = f"{user_task_prompt}\n\n--- PDF TEXT ---\n{raw_text}"

        # We reuse your main llm_service function for this complex parsing task.
        # It will return a dict with "schema" and "tableData". We only need "tableData".
        result = generate_json_response(full_parsing_prompt)

        if "error" in result:
            print(f"Error from generate_json_response during data population: {result['error']} - {result.get('details')}")
            return [{}]
        
        # The AI is instructed to return schema and data, but for this step, we mainly care about tableData.
        # The schema generated here might be a re-interpretation, so we stick to the schema from Step 2.
        populated_data = result.get("tableData", [{}])
        return populated_data if isinstance(populated_data, list) else [{}]

    except Exception as e:
        print(f"Error populating data with LLM: {e}")
        return [{}]

# ==========================================================
#        MAIN ENDPOINT ORCHESTRATING THE AI CHAIN
# ==========================================================

@app.post("/generate-table")
async def generate_table_endpoint(
    prompt: str = Form(...),
    file: UploadFile = File(None)
):
    final_prompt_for_schema = prompt
    raw_pdf_text = ""
    # table_data = [{}] # Initialize to empty or placeholder if no PDF

    # --- PDF Pre-processing (if file is provided) ---
    if file and file.filename:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Invalid file type. Only PDF files are accepted.")
        try:
            pdf_stream = await file.read() # Use await for async file read
            doc = fitz.open(stream=pdf_stream, filetype="pdf")
            raw_pdf_text = "".join(page.get_text() for page in doc).strip()
            doc.close()
            
            # --- AI CHAIN STEP 1: (Optional) Get column hints from PDF text ---
            # This step is to augment the main prompt for schema generation
            if raw_pdf_text:
                pdf_columns = get_pdf_columns_with_llm(raw_pdf_text)
                if pdf_columns:
                    print(f"Detected columns from PDF to guide schema: {pdf_columns}")
                    columns_text = ", ".join(f'"{c}"' for c in pdf_columns)
                    # Modify the prompt to guide the LLM for schema generation
                    super_prompt_addition = (
                        f" The user has also uploaded a PDF. "
                        f"Based on an initial scan, it seems to contain columns like: {columns_text}. "
                        f"Please ensure your generated schema includes these, along with any columns "
                        f"described in the main prompt. Make the most appropriate column "
                        f"(e.g., USN, ID, Sl. No.) the primary key and non-editable if it comes from the PDF."
                    )
                    final_prompt_for_schema += "\n" + super_prompt_addition
        except Exception as e:
            print(f"Failed to read or analyze PDF: {e}")
            # Decide if this is a fatal error or if you can proceed without PDF data
            # For now, we'll just print and let schema generation proceed with original prompt
            raw_pdf_text = "" # Ensure it's empty if PDF processing failed

    # --- AI CHAIN STEP 2: Generate the Final, Complete Schema (and initial data if no PDF) ---
    print(f"Generating schema with prompt: '{final_prompt_for_schema[:200]}...'")

    # The main LLM call that uses SYSTEM_INSTRUCTIONS.
    # If raw_pdf_text is available, it will be used by the LLM as per SYSTEM_INSTRUCTIONS.
    # If no raw_pdf_text, it should generate schema and empty tableData.
    llm_full_prompt = f"{final_prompt_for_schema}"
    if raw_pdf_text:
        llm_full_prompt += f"\n\n--- PDF TEXT ---\n{raw_pdf_text}"
    
    # Use the robust generate_json_response
    response_json = generate_json_response(llm_full_prompt)
    
    if "error" in response_json or "schema" not in response_json:
        error_detail = response_json.get('details', 'Unknown LLM error.')
        if "raw_response" in response_json: # Log raw response if available
            print(f"LLM Raw Error Response: {response_json['raw_response']}")
        raise HTTPException(status_code=500, detail=f"Failed to generate table schema: {error_detail}")
    
    final_schema = response_json["schema"]
    # tableData from this call will be used if PDF processing was successful via SYSTEM_INSTRUCTIONS
    # or it will be an empty array if no PDF was provided / processed.
    table_data_from_step2 = response_json.get("tableData", [{}])


    # --- AI CHAIN STEP 3: (Refined) Populate Data if PDF was processed AND schema is robust ---
    # The initial `generate_json_response` (Step 2) is already designed to populate `tableData`
    # if `--- PDF TEXT ---` is present. So, `populate_data_with_llm` might be redundant
    # IF the `SYSTEM_INSTRUCTIONS` are followed perfectly by the LLM in the first call.
    # However, having a dedicated parsing step can be more robust if the initial parse isn't perfect
    # or if you want to apply different logic/prompting for data extraction vs. schema generation.

    # Current logic: The `generate_json_response` in Step 2 handles schema AND data from PDF.
    # `table_data_from_step2` should contain the data if `raw_pdf_text` was provided.
    # If you find `table_data_from_step2` is often poorly populated when a PDF is present,
    # then you might re-enable a more focused `populate_data_with_llm` call here,
    # passing `raw_pdf_text` and `final_schema`.

    # For now, we trust the `generate_json_response` with `SYSTEM_INSTRUCTIONS` to do both.
    final_table_data = table_data_from_step2
    if not final_table_data and raw_pdf_text: # If PDF was there but no data came back
        print("Initial LLM call didn't populate data from PDF, attempting focused population...")
        # This would be the place for a corrective call if needed.
        # For simplicity in this iteration, we'll rely on the first call.
        # If you were to add it back:
        # final_table_data = populate_data_with_llm(raw_pdf_text, final_schema)
        pass # Keeping it simple for now


    # Ensure tableData is an array, even if it's an array with one empty object
    if not isinstance(final_table_data, list):
        final_table_data = [{}]
    if not final_table_data: # If it's an empty list
         final_table_data = [{}]


    # --- Final Step: Return the complete, correct result ---
    return { "schema": final_schema, "tableData": final_table_data }

# Other endpoints remain the same
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    favicon_path = os.path.join(os.path.dirname(__file__), "static", "favicon.ico")
    if os.path.exists(favicon_path):
        return FileResponse(favicon_path)
    return {"status": "no favicon"} # Or raise 404

@app.get("/")
def root():
    return {"status": "ok", "message": "Welcome to the Intelligent Table Generator API!"}

# To run this app (save as main.py in an 'app' directory, with llm_service.py):
# Ensure .env file with GOOGLE_API_KEY is in the backend/app directory or parent.
# Install dependencies: fastapi uvicorn python-multipart python-dotenv google-generativeai PyMuPDF
# Run from backend directory: uvicorn app.main:app --reload