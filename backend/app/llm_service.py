# backend/app/llm_service.py

import os
import json
import time
import random
import logging
from dotenv import load_dotenv
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted, DeadlineExceeded

# ----------------------------- ENV SETUP -----------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise EnvironmentError("ERROR: GOOGLE_API_KEY is not set in the .env file.")

# ----------------------------- LOGGER -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------- GEMINI CONFIG -----------------------------
try:
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    raise RuntimeError(f"ERROR: Failed to configure Gemini API: {e}")

# ----------------------------- SYSTEM PROMPT -----------------------------
SYSTEM_INSTRUCTIONS = """
You are an expert data structuring assistant. Your task is to convert a user's natural language prompt and any provided text data into a single, structured JSON object.

**Your output MUST be a single JSON object with two main keys: "schema" and "tableData".**

**1. The "schema" Object:**
This defines the table structure. It should contain a "tableName" and a "columns" array. Each object in the "columns" array must have:
- `id`: A unique snake_case identifier.
- `header`: The human-readable column name.
- `type`: 'text', 'number', 'date', or 'boolean'.
- `isPrimaryKey`: `true` for one column only (usually the first unique identifier like a USN or ID, or 'item_name' for a product list).
- `isEditable`: `true` for data entry columns, `false` for calculated columns (like totals, profit/loss, margin).
- `formula`: (Optional) A string representing the calculation for derived columns.
    - For sums: "SUM(col_id1, col_id2)"
    - For direct arithmetic: "col_id1 - col_id2" or "(col_id1 / col_id2) * 100". Ensure correct operator precedence with parentheses if needed. Ensure IDs in formulas match the `id` you assign to other columns. For example, if Profit/Loss is Selling Price minus Cost Price, and you assign `id: "selling_price"` and `id: "cost_price"`, the formula should be "selling_price - cost_price".
- `maxValue`: (Optional) A number if the prompt specifies a limit (e.g., "out of 10"). For number columns with a limit, also add "(Max: X)" to the `header`.
- `columnPurpose`: (Optional) A string to indicate special column types. 
    - If a 'Status' column is specifically for determining student eligibility (e.g., 'Eligible'/'Not Eligible' based on marks), set `columnPurpose: "student_eligibility_status"`. 
    - For other 'Status' columns (e.g., task status like 'In Progress', shipment status like 'Delivered'), this field can be omitted or set to 'general_status'.

**2. The "tableData" Array:**
This is an array of objects, where each object represents a row of data.

**CRITICAL RULES FOR SCHEMA AND DATA GENERATION:**
- **HOLISTIC SCHEMA FIRST:** Always determine the complete schema by considering THE ENTIRE user prompt. This includes descriptions of individual data entry columns, any requirements for total or calculated columns (and their formulas), AND any column information derived from an uploaded PDF (if '--- PDF TEXT ---' is present). This also means including any specified `columnPurpose` attributes.
- **IF a section labeled '--- PDF TEXT ---' is provided in the user's prompt:**
  - The `schema` you define MUST comprehensively include:
    1. All columns necessary for the data found in the PDF (e.g., "SI. No.", "USN", "Name").
    2. AND, CRUCIALLY, all other columns described in the user's main natural language prompt. This explicitly includes data entry columns (e.g., "Sessional Exam 1") AND any calculated/total/derived columns (e.g., "Total Internal Evaluation", "Profit Margin") along with their formulas and any specified `columnPurpose` (like for a student eligibility status column).
  - DO NOT OMIT columns (especially calculated/derived/status columns with special purposes) requested in the main prompt simply because '--- PDF TEXT ---' is present. Both sets of requirements must be merged into one complete schema.
  - After defining this complete schema, you MUST then parse the '--- PDF TEXT ---' to populate the `tableData` array, mapping the PDF data to the appropriate `id`s in your schema. Columns not found in the PDF will typically be empty or have their default/calculated values in the `tableData` initially.
- **IF NO '--- PDF TEXT ---' section is provided:**
  - Generate the `schema` based SOLELY on the user's natural language prompt, including any calculated/derived columns and any specified `columnPurpose`.
  - You MUST return an empty array for `tableData`: `[]`.

**Final Output:** Your entire response must be ONLY the raw JSON object. Do not add any other text or markdown fences.
"""

# ==========================================================
#        HELPER FUNCTION
# ==========================================================
def clean_gemini_response(text: str) -> str:
    """
    Removes markdown fences (```json ... ```) that the AI sometimes adds
    to its JSON response, making it safe to parse.
    """
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:].strip()
    elif text.startswith("```"):
        text = text[3:].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return text

# ----------------------------- MAIN FUNCTION -----------------------------
def generate_json_response(
    full_prompt: str,
    max_retries: int = 3,
    initial_backoff: float = 5.0,
    backoff_factor: float = 2.0,
    max_backoff: float = 60.0,
    model_name: str = "gemini-2.5-flash" # or "models/gemini-pro"
) -> dict:
    """
    Takes a combined prompt and gets a single JSON object containing
    both schema and tableData, with retry logic.
    """
    try:
        model = genai.GenerativeModel(model_name)
    except Exception as e:
        logger.error(f"Failed to initialize Gemini model: {e}")
        return {"error": "Failed to initialize LLM model.", "details": str(e)}

    retries_count = 0
    current_backoff = initial_backoff
    last_exception = None

    while retries_count <= max_retries:
        try:
            logger.info(f"Attempt {retries_count + 1}/{max_retries + 1} for combined prompt...")
            response = model.generate_content(
                [SYSTEM_INSTRUCTIONS, full_prompt], 
                generation_config=genai.types.GenerationConfig(
                    response_mime_type="application/json"
                ),
                request_options={"timeout": 120} 
            )
            cleaned_text = clean_gemini_response(response.text)
            parsed_json = json.loads(cleaned_text)
            return parsed_json
        except (ResourceExhausted, DeadlineExceeded) as e:
            last_exception = e
            if retries_count == max_retries:
                logger.error(f"Max retries reached ({type(e).__name__}): {e}")
                break
            wait_time = min(current_backoff + random.uniform(0, 1.0), max_backoff)
            logger.warning(f"{type(e).__name__} hit. Retrying in {wait_time:.2f}s.")
            time.sleep(wait_time)
            current_backoff *= backoff_factor
            retries_count += 1
        except json.JSONDecodeError as decode_error:
            logger.error(f"Invalid JSON response from LLM: {decode_error}")
            raw_text_to_log = "N/A"
            if 'response' in locals() and hasattr(response, 'text'):
                raw_text_to_log = response.text
            elif 'cleaned_text' in locals(): 
                raw_text_to_log = cleaned_text
            logger.debug(f"Raw response text leading to JSONDecodeError: {raw_text_to_log}")
            return {"error": "Failed to parse Gemini response", "details": str(decode_error), "raw_response": raw_text_to_log}
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return {"error": "Unexpected error during LLM generation", "details": str(e)}

    return {
        "error": "LLM request failed after multiple retries.",
        "details": str(last_exception) if last_exception else "Unknown error."

    }
