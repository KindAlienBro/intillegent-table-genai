 from fastapi import FastAPI, HTTPException, Form, UploadFile, File, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import fitz  # PyMuPDF
import json
import os
import sys
import re
import google.generativeai as genai

# This allows importing from your existing llm_service.py
# Ensure llm_service.py is in the same directory or a discoverable path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Assuming llm_service.py contains these functions
from llm_service import generate_json_response, clean_gemini_response

# -- Initialize the FastAPI App --
app = FastAPI(
    title="Intelligent Universal Prompt Table Generator API",
    version="1.6" # Version bump for Draft Feature
)

# -- Configure CORS --
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to your frontend's origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Directory for saving drafts ---
DRAFTS_DIR = os.path.join(os.path.dirname(__file__), "drafts")
os.makedirs(DRAFTS_DIR, exist_ok=True)


# ==========================================================
#                  PYDANTIC MODELS
# ==========================================================

class TableData(BaseModel):
    schema: Dict[str, Any]
    tableData: List[Dict[str, Any]]

class SaveDraftPayload(BaseModel):
    draftName: str = Field(..., min_length=1, max_length=50)
    content: TableData

# ==========================================================
#        HELPER FUNCTIONS FOR THE "AI CHAIN"
# ==========================================================

def get_pdf_columns_with_llm(raw_text: str) -> list[str]:
    """
    AI CHAIN STEP 1: Makes a small, fast AI call to identify column headers from the PDF.
    """
    if not raw_text.strip(): return []
    try:
        parsing_prompt = f"""Analyze the start of this text from a PDF and identify the column headers. Return a single, flat JSON array of strings with the header names. Example: ["SI. No.", "USN", "Name"]. Ignore document titles or any text that is clearly not a column header. Focus on typical tabular headers. Text: --- {raw_text[:1000]} ---"""
        model = genai.GenerativeModel("gemini-1.5-flash-latest")
        response = model.generate_content(
            parsing_prompt,
            generation_config=genai.types.GenerationConfig(response_mime_type="application/json"),
            request_options={"timeout": 30}
        )
        cleaned_text = clean_gemini_response(response.text)
        headers = json.loads(cleaned_text)
        return headers if isinstance(headers, list) and all(isinstance(h, str) for h in headers) else []
    except Exception as e:
        print(f"Error getting columns from PDF: {e}")
        return []

def populate_data_with_llm(raw_text: str, schema: dict) -> list:
    """
    AI CHAIN STEP 3: Uses a focused AI call to parse the full PDF text against a final, correct schema.
    """
    if not raw_text.strip() or not schema or not schema.get("columns"): return [{}]
    try:
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
        full_parsing_prompt = f"{user_task_prompt}\n\n--- PDF TEXT ---\n{raw_text}"
        result = generate_json_response(full_parsing_prompt)
        if "error" in result:
            print(f"Error from generate_json_response during data population: {result['error']} - {result.get('details')}")
            return [{}]
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
    
    if file and file.filename:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Invalid file type. Only PDF files are accepted.")
        try:
            pdf_stream = await file.read()
            doc = fitz.open(stream=pdf_stream, filetype="pdf")
            raw_pdf_text = "".join(page.get_text() for page in doc).strip()
            doc.close()
            
            if raw_pdf_text:
                pdf_columns = get_pdf_columns_with_llm(raw_pdf_text)
                if pdf_columns:
                    print(f"Detected columns from PDF to guide schema: {pdf_columns}")
                    columns_text = ", ".join(f'"{c}"' for c in pdf_columns)
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
            raw_pdf_text = ""

    print(f"Generating schema with prompt: '{final_prompt_for_schema[:200]}...'")
    llm_full_prompt = f"{final_prompt_for_schema}"
    if raw_pdf_text:
        llm_full_prompt += f"\n\n--- PDF TEXT ---\n{raw_pdf_text}"
    
    response_json = generate_json_response(llm_full_prompt)
    
    if "error" in response_json or "schema" not in response_json:
        error_detail = response_json.get('details', 'Unknown LLM error.')
        if "raw_response" in response_json:
            print(f"LLM Raw Error Response: {response_json['raw_response']}")
        raise HTTPException(status_code=500, detail=f"Failed to generate table schema: {error_detail}")
    
    final_schema = response_json["schema"]
    table_data_from_step2 = response_json.get("tableData", [{}])
    final_table_data = table_data_from_step2

    # This block seems to be a placeholder, if you intend to re-populate data, you'd call populate_data_with_llm here
    if not final_table_data and raw_pdf_text:
        print("Initial LLM call didn't populate data from PDF, attempting focused population...")
        # To make this functional, you would add:
        # final_table_data = populate_data_with_llm(raw_pdf_text, final_schema)
        pass

    if not isinstance(final_table_data, list):
        final_table_data = [{}]
    if not final_table_data:
         final_table_data = [{}]

    return {"schema": final_schema, "tableData": final_table_data}


# ==========================================================
#               NEW DRAFT ENDPOINTS
# ==========================================================

def sanitize_filename(name: str) -> str:
    """Sanitizes a string to be a safe filename."""
    name = re.sub(r'[^\w\s-]', '', name).strip()
    name = re.sub(r'[-\s]+', '-', name)
    return name

@app.post("/save-draft")
async def save_draft(payload: SaveDraftPayload):
    """Saves the current table schema and data as a JSON file."""
    try:
        draft_name = payload.draftName
        sanitized_name = sanitize_filename(draft_name)
        if not sanitized_name:
            raise HTTPException(status_code=400, detail="Invalid draft name. Name must contain alphanumeric characters.")
        
        file_path = os.path.join(DRAFTS_DIR, f"{sanitized_name}.json")
        
        # content is already a Pydantic model, so we can convert it to a dict
        draft_content = payload.content.dict()

        with open(file_path, "w") as f:
            json.dump(draft_content, f, indent=4)
            
        return {"status": "success", "message": f"Draft '{draft_name}' saved successfully."}
    except Exception as e:
        print(f"Error saving draft: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save draft. Reason: {e}")

@app.get("/drafts")
async def get_drafts_list():
    """Returns a list of all available draft filenames."""
    try:
        files = [f.replace(".json", "") for f in os.listdir(DRAFTS_DIR) if f.endswith(".json")]
        return {"drafts": files}
    except Exception as e:
        print(f"Error listing drafts: {e}")
        raise HTTPException(status_code=500, detail="Could not retrieve draft list.")

@app.get("/drafts/{draft_id}")
async def load_draft(draft_id: str = Path(..., description="The ID of the draft to load.")):
    """Loads a specific draft by its ID (filename without extension)."""
    try:
        sanitized_id = sanitize_filename(draft_id)
        file_path = os.path.join(DRAFTS_DIR, f"{sanitized_id}.json")
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Draft not found.")
            
        with open(file_path, "r") as f:
            content = json.load(f)
        
        return content
    except HTTPException as e:
        raise e # Re-raise HTTP exceptions
    except Exception as e:
        print(f"Error loading draft {draft_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Could not load draft '{draft_id}'.")


# ==========================================================
#                 OTHER ENDPOINTS
# ==========================================================

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    # Assuming you have a 'static' folder next to your 'backend' folder
    # The path might need adjustment based on your project structure
    favicon_path = os.path.join(os.path.dirname(__file__), "static", "favicon.ico")
    if os.path.exists(favicon_path):
        return FileResponse(favicon_path)
    return JSONResponse(status_code=404, content={"status": "no favicon"})

@app.get("/")
def root():
    return {"status": "ok", "message": "Welcome to the Intelligent Table Generator API!"}
