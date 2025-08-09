import os
import json
import google.generativeai as genai
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI()

@app.post("/api/")
async def process_questions(
    questions_txt: UploadFile = File(alias="questions.txt"),
    image_png: UploadFile = File(alias="image.png", default=None),
    data_csv: UploadFile = File(alias="data.csv", default=None)
):
    # Get Gemini API key from environment
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY environment variable not set")
    
    # Configure Gemini API
    genai.configure(api_key=api_key)
    
    # Read questions.txt content
    try:
        content = await questions_txt.read()
        questions_text = content.decode('utf-8')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading questions.txt: {str(e)}")
    
    # Read other files if provided
    prompt_parts = [f"Questions: {questions_text}"]
    
    if image_png:
        try:
            image_content = await image_png.read()
            prompt_parts.append(f"Image file provided: {image_png.filename}")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error reading image.png: {str(e)}")
    
    if data_csv:
        try:
            csv_content = await data_csv.read()
            csv_text = csv_content.decode('utf-8')
            prompt_parts.append(f"CSV Data: {csv_text}")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error reading data.csv: {str(e)}")
    
    # Create the full prompt with better context
    full_prompt = """You are a data analyst. Analyze the provided data carefully and respond with only a single valid JSON array or object containing the requested results. Be precise with calculations and use the actual data provided.

IMPORTANT: 
- Perform actual calculations, don't make assumptions
- Use exact values from the data
- Return only valid JSON, no explanations or extra text
- If you need to compute correlations, use the actual numerical data
- If generating visualizations, create them based on the real data patterns

Data to analyze:
""" + "\n\n".join(prompt_parts)
    
    # Generate response using Gemini with better configuration
    try:
        model = genai.GenerativeModel('gemini-2.0-flash-lite')
        
        # Configure generation parameters for better accuracy
        generation_config = genai.types.GenerationConfig(
            temperature=0.1,  # Lower temperature for more consistent results
            top_p=0.8,
            top_k=20,
            max_output_tokens=8192,
        )
        
        response = model.generate_content(
            full_prompt,
            generation_config=generation_config
        )
        
        if not response.text:
            raise HTTPException(status_code=500, detail="No response from Gemini API")
        
        generated_text = response.text.strip()
        
        # Parse the JSON response from Gemini
        try:
            json_response = json.loads(generated_text)
            return json_response
        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract JSON from the text
            text = generated_text
            start_idx = text.find('{')
            end_idx = text.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                try:
                    json_response = json.loads(text[start_idx:end_idx])
                    return json_response
                except json.JSONDecodeError:
                    pass
            
            # Check for array format
            start_idx = text.find('[')
            end_idx = text.rfind(']') + 1
            if start_idx != -1 and end_idx > start_idx:
                try:
                    json_response = json.loads(text[start_idx:end_idx])
                    return json_response
                except json.JSONDecodeError:
                    pass
            
            raise HTTPException(status_code=500, detail="Gemini did not return valid JSON")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calling Gemini API: {str(e)}")

@app.get("/")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))